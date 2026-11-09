;; Llama2 Inference in CHICKEN Scheme: array-morphisms implementation.
;;
;; This module computes the transformer forward pass with the
;; array-morphisms library.  Weight matrices are wrapped as concrete
;; array morphisms and multiplied through the array-morphisms BLAS
;; dispatch layer, which forwards to the registered BLAS backend; by
;; default this is the crunch backend from array-morphisms-crunch.
;;
;; The parts of a layer whose shapes do not depend on the token position
;; are written as array morphism graphs and run with the array-morphisms
;; trace/replay mechanism: each graph is compiled once into an SSA
;; program, traced in a morphism context that plans its buffers, and from
;; then on replayed into the same buffers for every token.  Each layer has
;; two such blocks:
;;
;;   qkv block:  rmsnorm, then the fused Q/K/V projection
;;   out block:  attention output projection and residual, rmsnorm,
;;               SwiGLU feed-forward network and residual
;;
;; and a final block computes rmsnorm and the classifier logits.  The
;; inputs of every block are morphisms that wrap the state vectors x and
;; xb, so each replay reads their current contents.
;;
;; Rotary embedding and attention over the key/value cache stay outside
;; the blocks, because the number of cached positions changes with every
;; token.  They call the crunch BLAS, softmax and rotary embedding kernels
;; directly, writing into preallocated vectors.
;;
;; The BLAS kernels read every operand from index 0, so each layer's
;; weights, and each head's key/value cache, are kept in separate vectors
;; instead of being addressed at an offset inside one large vector.

(module llama-am

        ( run make-am-inference-backend register-crunch-backend!
          transformer-weights->am-weights
          make-am-state am-state-logits
          am-transformer
          )

        (import
         scheme
         (only (scheme base) vector-map)
         (chicken base)
         (chicken format)
         (chicken memory)
         (chicken number-vector)
         (srfi 4)
         datatype
         (except yasos object object-with-ancestors operate-as)
         array-morphisms-core
         array-morphisms-basic-ops
         array-morphisms-blas-exec
         array-morphisms-activation-exec
         array-morphisms-context
         array-morphisms-ssa
         (only array-morphisms-grad make-var)
         array-morphisms-crunch-conv-backend
         array-morphisms-crunch-activations
         array-morphisms-crunch-threads
         (only array-morphisms-crunch-codegen register-crunch-codegen-backend!)
         llama-common
         )


;; Weights in the layout used by this module.  Layered tensors are Scheme
;; vectors indexed by layer.  Matrices are concrete array morphisms of
;; shape (out-dim, in-dim) and norm weights are (dim x 1) columns.  wqkv
;; holds Wq, Wk and Wv stacked into one (3 dim x dim) matrix.
(define-record-type am-weights
  (make-am-weights token-embedding-table
                   rms-att wqkv wo
                   rms-ffn w1 w2 w3
                   rms-final freq-cis-real freq-cis-imag
                   wcls)
  am-weights?
  (token-embedding-table am-weights-token-embedding-table)
  (rms-att am-weights-rms-att)
  (wqkv am-weights-wqkv)
  (wo am-weights-wo)
  (rms-ffn am-weights-rms-ffn)
  (w1 am-weights-w1)
  (w2 am-weights-w2)
  (w3 am-weights-w3)
  (rms-final am-weights-rms-final)
  (freq-cis-real am-weights-freq-cis-real)
  (freq-cis-imag am-weights-freq-cis-imag)
  (wcls am-weights-wcls))

;; A morphism graph compiled into an SSA program, together with the
;; morphism context holding its buffer plan.
(define-record-type compiled-block
  (make-compiled-block program context)
  compiled-block?
  (program compiled-block-program)
  (context compiled-block-context))

;; Activations, key/value cache, compiled blocks, and the BLAS kernels
;; taken from the active backend.  The morphisms x-m and xb-m wrap the
;; f32vectors x and xb without copying, as (dim x 1) columns; they are the
;; inputs of the compiled blocks.
(define-record-type am-state
  (%make-am-state x xb q k v logits
                  x-m xb-m
                  key-cache value-cache qh att outh
                  qkv-blocks out-blocks final-block
                  gemv gemm)
  am-state?
  (x am-state-x)
  (xb am-state-xb)
  (q am-state-q)
  (k am-state-k)
  (v am-state-v)
  (logits am-state-logits)
  (x-m am-state-x-m)
  (xb-m am-state-xb-m)
  (key-cache am-state-key-cache)       ; [layer][head] -> seq-len*head-size
  (value-cache am-state-value-cache)   ; [layer][head] -> seq-len*head-size
  (qh am-state-qh)                     ; [head] -> head-size
  (att am-state-att)                   ; [head] -> seq-len
  (outh am-state-outh)                 ; [head] -> head-size
  (qkv-blocks am-state-qkv-blocks)     ; [layer] -> compiled-block
  (out-blocks am-state-out-blocks)     ; [layer] -> compiled-block
  (final-block am-state-final-block)
  (gemv am-state-gemv)
  (gemm am-state-gemm))


;; Vector of n fresh f32vectors of the given length.
(define (make-f32vectors n len)
  (let ((v (make-vector n)))
    (do ((i 0 (+ i 1)))
        ((= i n) v)
      (vector-set! v i (make-f32vector len 0.0)))))

;; Copies n elements of src starting at src-offset into dst at dst-offset.
(define (f32-copy! dst dst-offset src src-offset n)
  (move-memory! src dst (* 4 n) (* 4 src-offset) (* 4 dst-offset)))

;; The data vector of a concrete array.
(define (concrete-data m)
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis) data)
    (else (error 'concrete-data "not a concrete array" m))))


;; Compiles the morphism graph rooted at root and runs it once in trace
;; mode, so that the context can plan the buffers, then once more to
;; compile the replay plan.  The graph is evaluated with whatever the
;; input vectors hold at this time.
(define (compile-block root)
  (let ((program (morphism-to-ssa (make-var root #f)))
        (ctx (make-morphism-context)))
    (ssa-realize/ctx ctx program)
    (finalize-context! ctx)
    (reset-context! ctx)
    (ssa-realize/ctx ctx program)
    (make-compiled-block program ctx)))

;; Replays a compiled block and returns the data vector of its result.
;; The vector is a context buffer that the next replay overwrites.
(define (run-block! block)
  (reset-context! (compiled-block-context block))
  (concrete-data (car (ssa-realize/ctx (compiled-block-context block)
                                       (compiled-block-program block)))))


;; Morphism graphs of the compiled blocks.  All vectors are (n x 1)
;; columns.

;; weight * x / sqrt(mean(x^2) + eps)
(define (rmsnorm-morphism x weight eps minus-half)
  (morph* (morph* x (morph-pow (morph+ (morph-reduce 'mean (morph* x x) '(0) #t) eps)
                               minus-half))
          weight))

;; Wqkv rmsnorm(x): Q, K and V stacked in one (3 dim x 1) column.
(define (qkv-morphism weights l x eps minus-half)
  (morph-matmul (vector-ref (am-weights-wqkv weights) l)
                (rmsnorm-morphism x (vector-ref (am-weights-rms-att weights) l)
                                  eps minus-half)))

;; With h = x + Wo att, the next residual stream
;;   h + W2 (silu(W1 n) * W3 n),  n = rmsnorm(h)
(define (out-morphism weights l x att eps minus-half)
  (let* ((h (morph+ x (morph-matmul (vector-ref (am-weights-wo weights) l) att)))
         (n (rmsnorm-morphism h (vector-ref (am-weights-rms-ffn weights) l)
                              eps minus-half))
         (a (morph-matmul (vector-ref (am-weights-w1 weights) l) n))
         (b (morph-matmul (vector-ref (am-weights-w3 weights) l) n)))
    (morph+ h (morph-matmul (vector-ref (am-weights-w2 weights) l)
                            (morph* (morph* a (morph-sigmoid a)) b)))))

;; Classifier logits Wcls rmsnorm(x).
(define (final-morphism weights x eps minus-half)
  (morph-matmul (am-weights-wcls weights)
                (rmsnorm-morphism x (am-weights-rms-final weights) eps minus-half)))


;; Allocates the run state for a configuration and compiles the blocks
;; for the given weights.  Must be called after the crunch backends have
;; been registered (see register-crunch-backend!), because the BLAS
;; kernels are taken from the active backend and the compiled blocks are
;; planned for it.
(define (make-am-state config weights)
  (let* ((dim (config-dim config))
         (n-layers (config-n-layers config))
         (n-heads (config-n-heads config))
         (seq-len (config-seq-len config))
         (head-size (quotient dim n-heads))
         (backend (or (active-blas-backend)
                      (error 'make-am-state "no BLAS backend is registered")))
         (x (make-f32vector dim 0.0))
         (xb (make-f32vector dim 0.0))
         (x-m (make-morphism x (list dim 1) 'f32))
         (xb-m (make-morphism xb (list dim 1) 'f32))
         (eps (make-morphism (f32vector 1e-5) (list 1 1) 'f32))
         (minus-half (make-morphism (f32vector -0.5) (list 1 1) 'f32))
         (per-layer (lambda (f)
                      (let ((v (make-vector n-layers)))
                        (do ((l 0 (+ l 1)))
                            ((= l n-layers) v)
                          (vector-set! v l (f l))))))
         (cache (lambda ()
                  (per-layer (lambda (l) (make-f32vectors n-heads (* seq-len head-size)))))))
    (%make-am-state x xb
                    (make-f32vector dim 0.0)
                    (make-f32vector dim 0.0)
                    (make-f32vector dim 0.0)
                    (make-f32vector (config-vocab-size config) 0.0)
                    x-m xb-m
                    (cache) (cache)
                    (make-f32vectors n-heads head-size)
                    (make-f32vectors n-heads seq-len)
                    (make-f32vectors n-heads head-size)
                    (per-layer (lambda (l)
                                 (compile-block (qkv-morphism weights l x-m eps minus-half))))
                    (per-layer (lambda (l)
                                 (compile-block (out-morphism weights l x-m xb-m eps minus-half))))
                    (compile-block (final-morphism weights x-m eps minus-half))
                    (blas-backend-gemv-f32 backend)
                    (blas-backend-gemm-f32 backend))))


;; Converts weights in either the flat or the per-layer layout (see
;; checkpoint-init-weights) into am-weights.
(define (transformer-weights->am-weights config weights)
  (let* ((dim (config-dim config))
         (hidden-dim (config-hidden-dim config))
         (n-layers (config-n-layers config))
         (vocab-size (config-vocab-size config))
         (per-layer
          (lambda (t size)
            (if (vector? t)
                t
                (let ((v (make-vector n-layers)))
                  (do ((l 0 (+ l 1)))
                      ((= l n-layers) v)
                    (vector-set! v l (subf32vector t (* l size) (* (+ l 1) size))))))))
         (matrices
          (lambda (t rows cols)
            (vector-map (lambda (m) (make-morphism m (list rows cols) 'f32))
                        (per-layer t (* rows cols)))))
         (columns
          (lambda (t)
            (vector-map (lambda (w) (make-morphism w (list dim 1) 'f32))
                        (per-layer t dim))))
         (stacked-qkv
          (let ((wq (per-layer (transformer-weights-wq weights) (* dim dim)))
                (wk (per-layer (transformer-weights-wk weights) (* dim dim)))
                (wv (per-layer (transformer-weights-wv weights) (* dim dim))))
            (vector-map (lambda (q k v)
                          (let ((m (make-f32vector (* 3 dim dim))))
                            (f32-copy! m 0 q 0 (* dim dim))
                            (f32-copy! m (* dim dim) k 0 (* dim dim))
                            (f32-copy! m (* 2 dim dim) v 0 (* dim dim))
                            (make-morphism m (list (* 3 dim) dim) 'f32)))
                        wq wk wv))))
    (make-am-weights
     (transformer-weights-token-embedding-table weights)
     (columns (transformer-weights-rms-att-weight weights))
     stacked-qkv
     (matrices (transformer-weights-wo weights) dim dim)
     (columns (transformer-weights-rms-ffn-weight weights))
     (matrices (transformer-weights-w1 weights) hidden-dim dim)
     (matrices (transformer-weights-w2 weights) dim hidden-dim)
     (matrices (transformer-weights-w3 weights) hidden-dim dim)
     (make-morphism (transformer-weights-rms-final-weight weights) (list dim 1) 'f32)
     (transformer-weights-freq-cis-real weights)
     (transformer-weights-freq-cis-imag weights)
     (let ((wcls (transformer-weights-wcls weights)))
       (make-morphism (if (= (f32vector-length wcls) (* vocab-size dim))
                          wcls
                          (subf32vector wcls 0 (* vocab-size dim)))
                      (list vocab-size dim) 'f32)))))


;; Transformer components outside the compiled blocks

;; Rotary position embedding applied in place to q and k.
(define (am-rope! state weights config pos)
  (let* ((dim (config-dim config))
         (head-size (quotient dim (config-n-heads config)))
         (base (* pos (quotient head-size 2)))
         (fr (am-weights-freq-cis-real weights))
         (fi (am-weights-freq-cis-imag weights)))
    (crunch-rope-f32 dim head-size base fr fi (am-state-q state))
    (crunch-rope-f32 dim head-size base fr fi (am-state-k state))))

;; Stores k and v for position pos, writes the attention output of every
;; head into xb.
(define (am-attention! state layer-idx pos config)
  (let* ((n-heads (config-n-heads config))
         (head-size (quotient (config-dim config) n-heads))
         (n-pos (+ pos 1))
         (scale (/ 1.0 (sqrt head-size)))
         (kc (vector-ref (am-state-key-cache state) layer-idx))
         (vc (vector-ref (am-state-value-cache state) layer-idx))
         (gemv (am-state-gemv state))
         (gemm (am-state-gemm state)))
    (do ((h 0 (+ h 1)))
        ((= h n-heads))
      (let ((off (* h head-size))
            (kh (vector-ref kc h))
            (vh (vector-ref vc h))
            (qh (vector-ref (am-state-qh state) h))
            (att (vector-ref (am-state-att state) h))
            (outh (vector-ref (am-state-outh state) h)))
        ;; Cache this position's key and value for head h
        (f32-copy! kh (* pos head-size) (am-state-k state) off head-size)
        (f32-copy! vh (* pos head-size) (am-state-v state) off head-size)
        (f32-copy! qh 0 (am-state-q state) off head-size)
        ;; att[0..pos] := scale * K[0..pos] qh
        (gemv n-pos head-size scale kh qh 0.0 att)
        (crunch-softmax-rows-f32 1 n-pos att att)
        ;; outh := att[0..pos] V[0..pos], a (1 x n-pos) by (n-pos x head-size) product
        (gemm 1 head-size n-pos 1.0 att vh 0.0 outh)
        (f32-copy! (am-state-xb state) off outh 0 head-size)))))

(define (am-transformer-layer! state weights l pos config)
  (let ((dim (config-dim config)))
    ;; Q, K and V from the compiled qkv block
    (let ((qkv (run-block! (vector-ref (am-state-qkv-blocks state) l))))
      (f32-copy! (am-state-q state) 0 qkv 0 dim)
      (f32-copy! (am-state-k state) 0 qkv dim dim)
      (f32-copy! (am-state-v state) 0 qkv (* 2 dim) dim))
    (am-rope! state weights config pos)
    (am-attention! state l pos config)
    ;; Output projection and feed-forward network from the compiled out block
    (f32-copy! (am-state-x state) 0
               (run-block! (vector-ref (am-state-out-blocks state) l)) 0 dim)))

;; Runs the model on one token at position pos; returns the logits vector.
(define (am-transformer token pos config state weights)
  (let ((dim (config-dim config)))
    (f32-copy! (am-state-x state) 0 (am-weights-token-embedding-table weights) (* token dim) dim)
    (do ((l 0 (+ l 1)))
        ((= l (config-n-layers config)))
      (am-transformer-layer! state weights l pos config))
    (f32-copy! (am-state-logits state) 0 (run-block! (am-state-final-block state))
               0 (config-vocab-size config))
    (am-state-logits state)))


;; Makes crunch the active array-morphisms BLAS backend and element-wise
;; (activation) backend.  When threads is given it also sets the number of
;; threads used by the crunch kernels.  When codegen is true (the default)
;; it also registers the segop code generator of array-morphisms-crunch,
;; which compiles kernels for the segops that no catalogue or BLAS kernel
;; runs; the environment variable AM_CRUNCH_CODEGEN=0 turns it off.
(define (register-crunch-backend! #!key (threads #f) (codegen #t))
  (register-blas-backend! (make-crunch-blas-backend))
  (register-activation-backend! (make-crunch-activation-backend))
  (when codegen (register-crunch-codegen-backend!))
  (when threads (crunch-thread-count threads)))

;; Inference backend running on array-morphisms with the crunch BLAS
;; backend.  When threads is given it sets the number of threads used by
;; the multithreaded crunch kernels; otherwise crunch uses AM_CRUNCH_THREADS
;; or the number of processors, at most 8.
(define (make-am-inference-backend #!key (threads #f))
  (let ((config #f)
        (weights #f)
        (state #f))
    (operations ()
     ((inference-backend? self) #t)
     ((backend-name self) 'crunch)
     ((backend-prepare! self conf port file-size)
      (register-crunch-backend! threads: threads)
      (set! config conf)
      (set! weights
        (transformer-weights->am-weights
         conf
         (checkpoint-init-weights conf port (config-shared-weights conf) file-size
                                  layout: 'per-layer)))
      (set! state (make-am-state conf weights)))
     ((backend-forward! self token pos)
      (am-transformer token pos config state weights)))))

;; Main runtime function
(define (run args #!key (threads #f))
  (run-with-backend args (make-am-inference-backend threads: threads)))
)
