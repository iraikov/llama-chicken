;; Llama2 Inference in CHICKEN Scheme
;; Based on OCaml llama2 implementation
;;
;; This module implements the transformer forward pass directly on the
;; CHICKEN blas egg.  Backend-independent components (records, checkpoint
;; and tokenizer readers, sampling, generation loop) live in llama-common
;; and are re-exported here.

(module llama


        ( run make-blas-inference-backend
          make-run-state run-state-logits
          run-state-xb run-state-xb2 run-state-x run-state-hb run-state-hb2
          run-state-q run-state-k run-state-v
          run-state-key-cache run-state-value-cache
          rmsnorm softmax matmul scopy accum
          transformer
          ; transformer components
          token-embedding-lookup
          get-rope-frequencies
          attention-rmsnorm
          compute-qkv
          apply-rope
          cache-kv
          compute-attention
          attention-output
          ffn-rmsnorm
          compute-ffn-w1w3
          apply-swiglu
          ffn-output
          final-rmsnorm
          compute-logits
          process-transformer-layer
          )

        (import
         scheme
         (chicken base)
         (chicken module)
         (chicken flonum)
         (chicken format)
         (chicken number-vector)
         (srfi 4)
         (except yasos object object-with-ancestors operate-as)
         blas
         llama-common
         )

        (reexport
         (only llama-common
               make-args args? args-checkpoint args-tokenizer args-temperature
               args-steps args-prompt args-seed
               make-config config-dim config-n-layers make-transformer-weights
               bpe-encode argmax verify-checkpoint-data))


(define-record-type run-state
  (make-run-state x xb q k v att key-cache value-cache xb2 hb hb2 logits)
  run-state?
  (x run-state-x run-state-x-set!)
  (xb run-state-xb run-state-xb-set!)
  (q run-state-q run-state-q-set!)
  (k run-state-k run-state-k-set!)
  (v run-state-v run-state-v-set!)
  (att run-state-att run-state-att-set!)
  (key-cache run-state-key-cache run-state-key-cache-set!)
  (value-cache run-state-value-cache run-state-value-cache-set!)
  (xb2 run-state-xb2 run-state-xb2-set!)
  (hb run-state-hb run-state-hb-set!)
  (hb2 run-state-hb2 run-state-hb2-set!)
  (logits run-state-logits run-state-logits-set!))


;; Vector subset
(define (dsub x start len)
  (subf64vector x start (+ start len)))
(define (ssub x start len)
  (subf32vector x start (+ start len)))

;; Vector copy with offset
(define (dblit out x #!key (Xoffset 0) (Yoffset 0) (size #f))
  (let ((size1 (or size (- (f64vector-length x) Xoffset))))
    (dicopy size1 x y: out offsetX: Xoffset offsetY: Yoffset)))

;; Vector copy with offset
(define (sblit out x #!key (Xoffset 0) (Yoffset 0) (size #f))
  (let ((size1 (or size (- (f32vector-length x) Xoffset))))
    (sicopy size1 x y: out offsetX: Xoffset offsetY: Yoffset)))

;; RMS normalization using BLAS
(define (rmsnorm out x weight)
  (let* ((epsilon 1e-5)
         (size (f32vector-length x))
         (x-mean-sqsum (fp/ (sdot size x x) (exact->inexact size)))
         (rms (sqrt (+ epsilon x-mean-sqsum))))
    ;; Use scal to scale vector by rms
    (sblit out x)
    (sscal! size (/ 1.0 rms) out)
    ;; element-wise multiply with weight
    (do ((i 0 (+ i 1)))
        ((= i size))
      (f32vector-set! out i (* (f32vector-ref out i) (f32vector-ref weight i)))))
  out)

;; Matrix-vector multiplication with BLAS gemv
(define (matmul out x w n d)
  ;; BLAS dgemv: y = alpha*A*x + beta*y
  (sgemv! RowMajor NoTrans d n 1.0 w x 0.0 out)
  out)

;; Dot product using BLAS dot
(define (dot-product arr1 arr2)
  (sdot (f32vector-length arr1) arr1 arr2))

;; Vector scaling and addition with BLAS axpy
(define (accum a b)
  ;; BLAS axpy: y = alpha*x + y
  (saxpy! (f32vector-length a) 1.0 b a)
  a)

(define (softmax out x size)
  (let* ((n (f32vector-length x))
         (max-val (f32vector-fold (lambda (elem acc) (max acc elem)) (f32vector-ref x 0) x)))

  (let ((exp-sum
         ;; Calculate exp and sum
         (let loop ((i 0) (exp-sum 0.0))
           (if (= i size)
               exp-sum
               (let ((exp-val (exp (- (f32vector-ref x i) max-val))))
                 (f32vector-set! out i exp-val)
                 (loop (+ i 1) (+ exp-sum exp-val))))
           ))
        )
    ;; Normalize by exp-sum
    (sscal! size (/ 1.0 exp-sum) out))
    
  ;; Copy the rest of the array if any
  (if (< size n)
      (sblit out x Xoffset: size Yoffset: size size: (- n size)))
    
  out)
  )


;; Transformer Components

;; Token Embedding Lookup
(define (token-embedding-lookup state weights token)
  "Extract token embedding and copy into state x vector"
  (let ((dim (f32vector-length (run-state-x state))))
    (sblit
     (run-state-x state)
     (transformer-weights-token-embedding-table weights)
     Xoffset: (* token dim)
     size: dim)))

;; Get RoPE frequency rows
(define (get-rope-frequencies weights pos head-size)
  "Extract RoPE frequency rows for given position"
  (let ((freq-cis-real-row
         (ssub
          (transformer-weights-freq-cis-real weights)
          (* pos (/ head-size 2))
          (/ head-size 2)))
        (freq-cis-imag-row
         (ssub 
          (transformer-weights-freq-cis-imag weights)
          (* pos (/ head-size 2))
          (/ head-size 2))))
    (values freq-cis-real-row freq-cis-imag-row)))

;; Attention RMS normalization
(define (attention-rmsnorm state weights layer-idx config)
  "Apply RMS normalization for attention layer"
  (let* ((dim (config-dim config))
         (rms-att-weight-l (ssub 
                            (transformer-weights-rms-att-weight weights)
                            (* layer-idx dim) dim)))
    (rmsnorm (run-state-xb state) (run-state-x state) rms-att-weight-l)
    ))
             
              
;; Compute QKV matrices
(define (compute-qkv state weights layer-idx config)
  "Compute Query, Key, Value matrices for given layer"
  (let* ((dim (config-dim config))
         (wq-l (ssub (transformer-weights-wq weights)
                     (* layer-idx dim dim)
                     (* dim dim)))
         (wk-l (ssub (transformer-weights-wk weights)
                     (* layer-idx dim dim)
                     (* dim dim)))
         (wv-l (ssub (transformer-weights-wv weights)
                     (* layer-idx dim dim)
                     (* dim dim))))
    
    (matmul (run-state-q state) (run-state-xb state) wq-l dim dim)
    (matmul (run-state-k state) (run-state-xb state) wk-l dim dim)
    (matmul (run-state-v state) (run-state-xb state) wv-l dim dim)))

;; Apply RoPE rotation
(define (apply-rope state config freq-cis-real-row freq-cis-imag-row)
  "Apply Rotary Position Embedding to Q and K vectors"
  (let ((head-size (/ (config-dim config) (config-n-heads config))))
    
    (do ((h 0 (+ h 1)))
        ((= h (config-n-heads config)))

      (let (
            (q (ssub (run-state-q state) (* h head-size) head-size))
            (k (ssub (run-state-k state) (* h head-size) head-size))
            )

        ;; Rotate q and k
        (let loop ((i 0))
          (when (< i head-size)
            (let ((q0 (f32vector-ref q i))
                  (q1 (f32vector-ref q (+ i 1)))
                  (k0 (f32vector-ref k i))
                  (k1 (f32vector-ref k (+ i 1)))
                  (fcr (f32vector-ref freq-cis-real-row (/ i 2)))
                  (fci (f32vector-ref freq-cis-imag-row (/ i 2))))
              
              (f32vector-set! q i (- (* q0 fcr) (* q1 fci)))
              (f32vector-set! q (+ i 1) (+ (* q0 fci) (* q1 fcr)))
              (f32vector-set! k i (- (* k0 fcr) (* k1 fci)))
              (f32vector-set! k (+ i 1) (+ (* k0 fci) (* k1 fcr)))
              
              (loop (+ i 2)))))
      
        ;; Copy back to state
        (sblit (run-state-q state) q Yoffset: (* h head-size))
        (sblit (run-state-k state) k Yoffset: (* h head-size))
        ))
    ))


;; Cache key-value pairs
(define (cache-kv state layer-idx pos config)
  "Store current key and value vectors in cache"
  (let* ((dim (config-dim config))
         (seq-len (config-seq-len config))
         (loff (* layer-idx seq-len dim))
         (offset (+ loff (* pos dim))))
    (sblit (run-state-key-cache state) (run-state-k state) Yoffset: offset)
    (sblit (run-state-value-cache state) (run-state-v state) Yoffset: offset)))

;; Compute multi-head attention
(define (compute-attention state layer-idx pos config)
  "Compute multi-head attention scores and apply to values"
  (let* ((head-size (/ (config-dim config) (config-n-heads config)))
         (seq-len (config-seq-len config))
         (dim (config-dim config))
         (att-softmax (make-f32vector seq-len)))
    
    (do ((h 0 (+ h 1)))
        ((= h (config-n-heads config)))
      
      (let ((q (ssub (run-state-q state) (* h head-size) head-size))
            (att (ssub (run-state-att state) (* h seq-len) seq-len)))
      
        ;; Compute attention scores
        (do ((t 0 (+ t 1)))
            ((> t pos))
          
          (let* ((loff (* layer-idx seq-len dim))
                 (k (ssub (run-state-key-cache state) (+ loff (* t dim) (* h head-size)) head-size)))
            
            (let ((score (/ (dot-product q k) (sqrt head-size))))
              (f32vector-set! att t score))))
        
        ;; Softmax the scores
        (let ((att-softmax (softmax att-softmax att (+ pos 1))))
          
          ;; Weighted sum of values
          (let ((xb-ptr (* h head-size))
                (xb-end (* (+ h 1) head-size)))
            ;; zero out xb
            (let loop ((i xb-ptr))
              (if (< i xb-end)
                  (begin
                    (f32vector-set! (run-state-xb state) i 0)
                    (loop (+ 1 i)))))
            
            (do ((t pos (- t 1)))
                ((< t 0))
              (let* ((loff (* layer-idx seq-len dim))
                     (v (ssub (run-state-value-cache state)
                              (+ loff (* t dim) (* h head-size))
                              head-size))
                     (a (f32vector-ref att-softmax t)))
                
                ;; Use axpy for accumulation: y := a*x + y
                (siaxpy! head-size a v 
                         (run-state-xb state)
                         #:incx 1 #:incy 1 #:offy xb-ptr))))))))
  )

;; Attention output projection
(define (attention-output state weights layer-idx config)
  "Apply final linear transformation to attention output"
  (let* ((dim (config-dim config))
         (wo-l (ssub (transformer-weights-wo weights)
                     (* layer-idx dim dim) 
                     (* dim dim))))
    (matmul (run-state-xb2 state) (run-state-xb state) wo-l dim dim)))

;; FFN RMS normalization  
(define (ffn-rmsnorm state weights layer-idx config)
  "Apply RMS normalization for feed-forward network"
  (let* ((dim (config-dim config))
         (rms-ffn-weight-l 
          (ssub (transformer-weights-rms-ffn-weight weights)
                (* layer-idx dim) dim)))
    (rmsnorm (run-state-xb state) (run-state-x state) rms-ffn-weight-l)))

;; Compute FFN first part (W1 and W3)
(define (compute-ffn-w1w3 state weights layer-idx config)
  "Compute first part of FFN: W1(x) and W3(x)"
  (let* ((dim (config-dim config))
         (hidden-dim (config-hidden-dim config))
         (w1-l (ssub (transformer-weights-w1 weights)
                       (* layer-idx dim hidden-dim)
                       (* dim hidden-dim)))
         (w3-l (ssub (transformer-weights-w3 weights)
                       (* layer-idx dim hidden-dim)
                       (* dim hidden-dim))))
    
    (matmul (run-state-hb state) (run-state-xb state) w1-l dim hidden-dim)
    (matmul (run-state-hb2 state) (run-state-xb state) w3-l dim hidden-dim)))

;; Apply SwiGLU activation
(define (apply-swiglu state config)
  "Apply SwiGLU activation: SiLU(W1(x)) * W3(x)"
  (let* ((hidden-dim (config-hidden-dim config))
         (hb (run-state-hb state))
         (hb2 (run-state-hb2 state)))

    ;; Apply SiLU (Swish) to W1(x)
    (do ((i 0 (+ i 1)))
        ((= i hidden-dim))
      (f32vector-set! hb i (* (f32vector-ref hb i)
                              (/ 1.0 (+ 1.0 (exp (- (f32vector-ref hb i))))))))
    
    ;; Elementwise multiply with W3(x)
    (do ((i 0 (+ i 1)))
        ((= i hidden-dim))
      (f32vector-set! hb i (* (f32vector-ref hb i) (f32vector-ref hb2 i))))))

;; FFN output projection
(define (ffn-output state weights layer-idx config)
  "Apply final FFN linear transformation W2"
  (let* ((dim (config-dim config))
         (hidden-dim (config-hidden-dim config))
         (w2-l (ssub (transformer-weights-w2 weights)
                       (* layer-idx hidden-dim dim)
                       (* hidden-dim dim))))
    ;(printf "ffn-output: xb = ~A hb = ~A w2-l = ~A\n"
    ;        (run-state-xb state) (run-state-hb state) w2-l)
    
    (matmul (run-state-xb state) (run-state-hb state) w2-l hidden-dim dim)))

;; Final RMS Normalization
(define (final-rmsnorm state weights)
  "Apply final RMS normalization before classification"
  (rmsnorm (run-state-x state) (run-state-x state) 
           (transformer-weights-rms-final-weight weights)))

;; Compute Final Logits
(define (compute-logits state weights config)
  "Compute final classification logits"
  (matmul (run-state-logits state)
          (run-state-x state)
          (transformer-weights-wcls weights)
          (config-dim config)
          (config-vocab-size config)
          ))

;; Process Single Transformer Layer
(define (process-transformer-layer state weights layer-idx pos config freq-cis-real-row freq-cis-imag-row)
  "Process a single transformer layer"
  ;; Attention block
  (attention-rmsnorm state weights layer-idx config)
  (compute-qkv state weights layer-idx config)
  (apply-rope state config freq-cis-real-row freq-cis-imag-row)
  (cache-kv state layer-idx pos config)
  (compute-attention state layer-idx pos config)
  (attention-output state weights layer-idx config)
  (accum (run-state-x state) (run-state-xb2 state))
  
  ;; FFN block
  (ffn-rmsnorm state weights layer-idx config)
  (compute-ffn-w1w3 state weights layer-idx config)
  (apply-swiglu state config)
  
  (ffn-output state weights layer-idx config)
  (accum (run-state-x state) (run-state-xb state))
  )

;; Main Transformer implementation
(define (transformer token pos conf state weights)

  (let* ((head-size (/ (config-dim conf) (config-n-heads conf))))
    
    ;; Token embedding lookup
    (token-embedding-lookup state weights token)
    
    ;; Get RoPE frequencies for this position
    (let-values (((freq-cis-real-row freq-cis-imag-row) 
                  (get-rope-frequencies weights pos head-size)))
      
      ;; Process all layers
      (do ((l 0 (+ l 1)))
          ((= l (config-n-layers conf)))
        (process-transformer-layer state weights l pos conf 
                                   freq-cis-real-row freq-cis-imag-row))
      
      ;; Final processing
      (final-rmsnorm state weights)
      (compute-logits state weights conf)
      
      state)))

;; Allocates a zeroed run state for the given configuration.
(define (config->run-state config)
  (let ((dim (config-dim config))
        (hidden-dim (config-hidden-dim config))
        (cache-size (* (config-n-layers config)
                       (config-seq-len config)
                       (config-dim config))))
    (make-run-state
     (make-f32vector dim 0.0)
     (make-f32vector dim 0.0)
     (make-f32vector dim 0.0)
     (make-f32vector dim 0.0)
     (make-f32vector dim 0.0)
     (make-f32vector (* (config-n-heads config) (config-seq-len config)) 0.0)
     (make-f32vector cache-size 0.0)
     (make-f32vector cache-size 0.0)
     (make-f32vector dim 0.0)
     (make-f32vector hidden-dim 0.0)
     (make-f32vector hidden-dim 0.0)
     (make-f32vector (config-vocab-size config) 0.0))))

;; Inference backend that runs the forward pass on the blas egg, with the
;; weights in the flat layout.
(define (make-blas-inference-backend)
  (let ((config #f)
        (weights #f)
        (state #f))
    (operations ()
     ((inference-backend? self) #t)
     ((backend-name self) 'blas)
     ((backend-prepare! self conf port file-size)
      (set! config conf)
      (set! weights (checkpoint-init-weights conf port
                                             (config-shared-weights conf)
                                             file-size))
      (set! state (config->run-state conf)))
     ((backend-forward! self token pos)
      (transformer token pos config state weights)
      (run-state-logits state)))))

;; Main runtime function
(define (run args)
  (run-with-backend args (make-blas-inference-backend)))
)
