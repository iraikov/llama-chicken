;; Llama2 Inference in CHICKEN Scheme
;; Based on OCaml llama2 implementation

(module llama


        ( run make-args args? args-checkpoint args-tokenizer args-temperature args-steps args-prompt args-seed
          make-config config-dim config-n-layers make-transformer-weights make-run-state run-state-logits
          run-state-xb run-state-xb2 run-state-x run-state-hb run-state-hb2
          run-state-q run-state-k run-state-v
          run-state-key-cache run-state-value-cache
          bpe-encode rmsnorm softmax argmax matmul scopy accum
          verify-checkpoint-data
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
         (chicken io)
         (chicken base)
         (chicken bitwise)
         (chicken condition)
         (chicken flonum)
         (chicken process-context)
         (chicken foreign)
         (chicken port)
         (chicken time)
         (chicken file)
         (chicken file posix)
         (chicken format)
         (chicken string)
         (chicken sort)
         (chicken blob)
         (chicken memory)
         (srfi 1)   ; List library
         (srfi 4)   ; Numeric vectors
         (srfi 69)  ; Hash tables
         (srfi 42)  ; Comprehensions
         (endian-blob)
         (endian-port)
         (prefix random-mtzig random:)
         vector-lib
         blas
         )

        

(define-record-type args
  (make-args checkpoint tokenizer temperature steps prompt seed)
  args?
  (checkpoint args-checkpoint args-checkpoint-set!)
  (tokenizer args-tokenizer args-tokenizer-set!)
  (temperature args-temperature args-temperature-set!)
  (steps args-steps args-steps-set!)
  (prompt args-prompt args-prompt-set!)
  (seed args-seed)
  )

(define-record-type config
  (make-config dim hidden-dim n-layers n-heads n-kv-heads vocab-size seq-len shared-weights)
  config?
  (dim config-dim)
  (hidden-dim config-hidden-dim)
  (n-layers config-n-layers)
  (n-heads config-n-heads)
  (n-kv-heads config-n-kv-heads)
  (vocab-size config-vocab-size)
  (seq-len config-seq-len)
  (shared-weights config-shared-weights))

(define-record-type transformer-weights
  (make-transformer-weights token-embedding-table
                            rms-att-weight
                            wq wk wv wo
                            rms-ffn-weight
                            w1 w2 w3
                            rms-final-weight
                            freq-cis-real
                            freq-cis-imag
                            wcls)
  transformer-weights?
  (token-embedding-table transformer-weights-token-embedding-table transformer-weights-token-embedding-table-set!)
  (rms-att-weight transformer-weights-rms-att-weight transformer-weights-rms-att-weight-set!)
  (wq transformer-weights-wq transformer-weights-wq-set!)
  (wk transformer-weights-wk transformer-weights-wk-set!)
  (wv transformer-weights-wv transformer-weights-wv-set!)
  (wo transformer-weights-wo transformer-weights-wo-set!)
  (rms-ffn-weight transformer-weights-rms-ffn-weight transformer-weights-rms-ffn-weight-set!)
  (w1 transformer-weights-w1 transformer-weights-w1-set!)
  (w2 transformer-weights-w2 transformer-weights-w2-set!)
  (w3 transformer-weights-w3 transformer-weights-w3-set!)
  (rms-final-weight transformer-weights-rms-final-weight transformer-weights-rms-final-weight-set!)
  (freq-cis-real transformer-weights-freq-cis-real transformer-weights-freq-cis-real-set!)
  (freq-cis-imag transformer-weights-freq-cis-imag transformer-weights-freq-cis-imag-set!)
  (wcls transformer-weights-wcls transformer-weights-wcls-set!))

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


(define (f64vector-fold f x0 v . rest)
    (let ((n   (f64vector-length v))
	  (vs  (cons v rest)))
      (fold-ec x0 (:range i 0 n)
	       (map (lambda (v) (f64vector-ref v i)) vs)
	       (lambda (x ax) (apply f (append x (list ax)))))))

(define (f32vector-fold f x0 v . rest)
    (let ((n   (f32vector-length v))
	  (vs  (cons v rest)))
      (fold-ec x0 (:range i 0 n)
	       (map (lambda (v) (f32vector-ref v i)) vs)
	       (lambda (x ax) (apply f (append x (list ax)))))))

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

(define (read-config port checkpoint)
  (let* ((size 28))  ; bytes in int * 7 = 28
    (handle-exceptions exn
      (begin
        (printf "Couldn't open file ~A: ~A\n" checkpoint exn)
        (exit 1))
      (set-littlendian! port)
      (let* ((dim (read-uint4 port))
             (hidden-dim (read-uint4 port))
             (n-layers (read-uint4 port))
             (n-heads (read-uint4 port))
             (n-kv-heads (read-uint4 port))
             (vocab-size (read-uint4 port))
             (seq-len (read-uint4 port))
             (shared-weights (> vocab-size 0)))
        (make-config dim hidden-dim n-layers n-heads n-kv-heads 
                     (abs vocab-size) seq-len shared-weights))))
  )

;; Helper function to compute vector statistics
(define (vector-stats vec name)
  (if (= (f32vector-length vec) 0)
      (printf "  ~A: EMPTY VECTOR\n" name)
      (let* ((len (f32vector-length vec))
             (min-val (f32vector-fold (lambda (x ax) (min x ax)) +inf.0 vec))
             (max-val (f32vector-fold (lambda (x ax) (max x ax)) -inf.0 vec))
             (sum (f32vector-fold (lambda (x ax) (+ x ax)) 0.0 vec))
             (mean (/ sum (exact->inexact len)))
             (variance (/ (f32vector-fold (lambda (x ax) (+ ax (expt (- x mean) 2))) 0.0 vec) (exact->inexact len)))
             (std-dev (sqrt variance)))
        (printf "  ~A: [~A] min=~A max=~A mean=~A std=~A\n" 
                name len min-val max-val mean std-dev))))

;; Print sample values from different parts of a vector
(define (print-sample-values vec name num-samples)
  (let ((len (f32vector-length vec)))
    (when (> len 0)
      (printf "    ~A samples: [" name)
      (let ((indices (cond
                      ((= len 1) '(0))
                      ((< len num-samples) (iota len))
                      (else (let ((step (max 1 (quotient len (- num-samples 1)))))
                              (append (iota (- num-samples 1) 0 step) 
                                     (list (- len 1))))))))
        (for-each (lambda (i idx)
                    (printf "~A" (f32vector-ref vec idx))
                    (unless (= i (- (length indices) 1))
                      (printf " ")))
                  (iota (length indices)) indices))
      (printf "]\n"))))

;; Print matrix-like weights with proper dimension interpretation
(define (print-matrix-weights vec name rows cols layer-info)
  (printf "  ~A (~A):\n" name layer-info)
  (vector-stats vec "stats")
  (print-sample-values vec name 8)
  
  ;; For matrices, show some actual matrix structure
  (when (and (> rows 0) (> cols 0) (< (* rows cols) 50))
    (printf "    Matrix structure (~Ax~A):\n" rows cols)
    (do ((r 0 (+ r 1)))
        ((or (= r rows) (> r 4)))  ; Limit to first 5 rows
      (printf "      [")
      (do ((c 0 (+ c 1)))
          ((or (= c cols) (> c 7)))  ; Limit to first 8 columns
        (printf "~A" (f32vector-ref vec (+ (* c rows) r)))  ; Column-major
        (unless (or (= c (- cols 1)) (= c 7))
          (printf " ")))
      (when (> cols 8) (printf " ..."))
      (printf "]\n"))
    (when (> rows 5) (printf "      ...\n"))))

;; Main pretty printer for transformer weights
(define (print-transformer-weights weights config #!optional (detailed #f))
  (let* ((dim (config-dim config))
         (hidden-dim (config-hidden-dim config))
         (n-layers (config-n-layers config))
         (n-heads (config-n-heads config))
         (vocab-size (config-vocab-size config))
         (seq-len (config-seq-len config))
         (head-size (/ dim n-heads)))
    
    (printf "===============================================\n")
    (printf "TRANSFORMER WEIGHTS SUMMARY\n")
    (printf "===============================================\n")
    (printf "Model Configuration:\n")
    (printf "  Dimensions: ~A\n" dim)
    (printf "  Hidden Dim: ~A\n" hidden-dim)
    (printf "  Layers: ~A\n" n-layers)
    (printf "  Heads: ~A\n" n-heads)
    (printf "  Vocab Size: ~A\n" vocab-size)
    (printf "  Sequence Length: ~A\n" seq-len)
    (printf "  Head Size: ~A\n" head-size)
    (printf "\n")
    
    ;; Token Embedding Table
    (printf "TOKEN EMBEDDING TABLE:\n")
    (print-matrix-weights (transformer-weights-token-embedding-table weights)
                         "token-embedding-table"
                         dim vocab-size
                         (sprintf "~A tokens x ~A dims" vocab-size dim))
    (printf "\n")
    
    ;; Layer-wise weights
    (printf "LAYER WEIGHTS:\n")
    (do ((l 0 (+ l 1)))
        ((= l n-layers))
      (printf "  Layer ~A:\n" l)
      
      ;; Attention weights for this layer
      (let ((att-start (* l dim))
            (att-end (* (+ l 1) dim)))
        (printf "    Attention RMS Norm:\n")
        (vector-stats (subf32vector (transformer-weights-rms-att-weight weights) 
                                    att-start att-end)
                     "rms-att-weight"))
      
      ;; QKV weights for this layer  
      (let ((qkv-start (* l dim dim))
            (qkv-end (* (+ l 1) dim dim)))
        (printf "    Query/Key/Value Matrices:\n")
        (vector-stats (subf32vector (transformer-weights-wq weights) qkv-start qkv-end) "wq")
        (vector-stats (subf32vector (transformer-weights-wk weights) qkv-start qkv-end) "wk") 
        (vector-stats (subf32vector (transformer-weights-wv weights) qkv-start qkv-end) "wv")
        (vector-stats (subf32vector (transformer-weights-wo weights) qkv-start qkv-end) "wo"))
      
      ;; FFN weights for this layer
      (let ((ffn-rms-start (* l dim))
            (ffn-rms-end (* (+ l 1) dim))
            (w1-start (* l dim hidden-dim))
            (w1-end (* (+ l 1) dim hidden-dim))
            (w2-start (* l hidden-dim dim))
            (w2-end (* (+ l 1) hidden-dim dim)))
        (printf "    FFN Weights:\n")
        (vector-stats (subf32vector (transformer-weights-rms-ffn-weight weights)
                                   ffn-rms-start ffn-rms-end)
                     "rms-ffn-weight")
        (vector-stats (subf32vector (transformer-weights-w1 weights) w1-start w1-end) "w1")
        (vector-stats (subf32vector (transformer-weights-w2 weights) w2-start w2-end) "w2")
        (vector-stats (subf32vector (transformer-weights-w3 weights) w1-start w1-end) "w3"))
      
      (printf "\n"))
    
    ;; Final normalization
    (printf "FINAL WEIGHTS:\n")
    (vector-stats (transformer-weights-rms-final-weight weights) "rms-final-weight")
    (printf "\n")
    
    ;; RoPE frequencies
    (printf "ROPE FREQUENCIES:\n")
    (vector-stats (transformer-weights-freq-cis-real weights) "freq-cis-real")
    (vector-stats (transformer-weights-freq-cis-imag weights) "freq-cis-imag")
    (printf "\n")
    
    ;; Output classifier
    (printf "OUTPUT CLASSIFIER:\n")
    (if (eq? (transformer-weights-wcls weights) 
             (transformer-weights-token-embedding-table weights))
        (printf "  wcls: SHARED with token-embedding-table\n")
        (print-matrix-weights (transformer-weights-wcls weights)
                             "wcls"
                             dim vocab-size
                             (sprintf "~A tokens x ~A dims" vocab-size dim)))
    (printf "\n")
    
    ;; Detailed breakdown if requested
    (when detailed
      (printf "DETAILED WEIGHT ANALYSIS:\n")
      (printf "===============================================\n")
      
      ;; Show actual sample values for first layer
      (printf "LAYER 0 DETAILED SAMPLES:\n")
      (print-sample-values (subf32vector (transformer-weights-wq weights) 0 (* dim dim))
                          "WQ[0]" 10)
      (print-sample-values (subf32vector (transformer-weights-wk weights) 0 (* dim dim))
                          "WK[0]" 10)
      (print-sample-values (subf32vector (transformer-weights-wv weights) 0 (* dim dim))
                          "WV[0]" 10)
      
      ;; Show frequency patterns
      (printf "\nROPE FREQUENCY PATTERNS:\n")
      (print-sample-values (transformer-weights-freq-cis-real weights) "cos frequencies" 12)
      (print-sample-values (transformer-weights-freq-cis-imag weights) "sin frequencies" 12)
      
      ;; Show embedding patterns for first few tokens
      (printf "\nTOKEN EMBEDDING SAMPLES:\n")
      (do ((token 0 (+ token 1)))
          ((or (= token vocab-size) (> token 4)))
        (let ((start (* token dim)))
          (printf "  Token ~A: [" token)
          (do ((i 0 (+ i 1)))
              ((or (= i dim) (> i 7)))
            (printf "~A" (f32vector-ref (transformer-weights-token-embedding-table weights) 
                                          (+ start i)))
            (unless (or (= i (- dim 1)) (= i 7))
              (printf " ")))
          (when (> dim 8) (printf " ..."))
          (printf "]\n"))))
    
    (printf "===============================================\n")))

;; Function to load and print weights from a checkpoint file
(define (verify-checkpoint-data checkpoint-file #!optional (detailed #f))

  (printf "Loading checkpoint: ~A\n" checkpoint-file)
  
  (let ((port (open-endian-port 'read checkpoint-file)))
    (set-littlendian! port)
    (let* ((config (read-config port checkpoint-file))
           (file-size (file-size checkpoint-file))
           (weights (checkpoint-init-weights config port 
                                           (config-shared-weights config)
                                           file-size)))
      (close-endian-port port)
      
      (print-transformer-weights weights config detailed)
      
      ;; Quick sanity checks
      (printf "Checkpoint validation:\n")
      (printf "  All weights finite? ")
      (let ((all-finite? 
             (and (f32vector-fold (lambda (x ax) (and (finite? x) ax)) #t
                                  (transformer-weights-token-embedding-table weights))
                  (f32vector-fold (lambda (x ax) (and (finite? x) ax)) #t
                                  (transformer-weights-rms-final-weight weights)))))
        (printf "~A\n" (if all-finite? "YES" "NO")))
      
      ;(printf "  Non-zero weights? ")
      ;(let ((has-non-zero?
      ;       (or (any (lambda (x) (not (= x 0.0)))
      ;               (f32vector->list (transformer-weights-token-embedding-table weights)))
      ;           (any (lambda (x) (not (= x 0.0)))
      ;               (f32vector->list (transformer-weights-rms-final-weight weights))))))
      ;  (printf "~A\n" (if has-non-zero? "YES" "NO")))
      
      weights)))


;; Checkpoint and Tokenizer Functions
(define (checkpoint-init-weights conf port shared-weights file-size)
  
  (let* (
         (vocab-size (config-vocab-size conf))
         (dim (config-dim conf))
         (n-layers (config-n-layers conf))
         (hidden-dim (config-hidden-dim conf))
         (seq-len (config-seq-len conf))
         (n-heads (config-n-heads conf)))

    (let* (
          (token-embedding-table
           (read-ieee-float32-vector port (* vocab-size dim)))

          (rms-att-weight 
           (read-ieee-float32-vector port (* n-layers dim)))
    
          (wq 
           (read-ieee-float32-vector port (* n-layers dim dim)))
    
          (wk 
           (read-ieee-float32-vector port (* n-layers dim dim)))
    
          (wv 
           (read-ieee-float32-vector port (* n-layers dim dim)))
    
          (wo 
           (read-ieee-float32-vector port (* n-layers dim dim)))
    
          (rms-ffn-weight 
           (read-ieee-float32-vector port (* n-layers dim)))
    
          (w1 
           (read-ieee-float32-vector port (* n-layers dim hidden-dim)))
    
          (w2 
           (read-ieee-float32-vector port (* n-layers hidden-dim dim)))
    
          (w3 
           (read-ieee-float32-vector port (* n-layers dim hidden-dim)))
    
          (rms-final-weight 
           (read-ieee-float32-vector port dim))
    
          (freq-cis-real 
           (read-ieee-float32-vector port (* seq-len (/ (/ dim n-heads) 2))))
    
          (freq-cis-imag 
           (read-ieee-float32-vector port (* seq-len (/ (/ dim n-heads) 2))))
          )
      (let (
            (wcls
             (if shared-weights
                 token-embedding-table
                 (read-ieee-float32-vector port (/ (- file-size (file-position (endian-port-fileno port))) 4))))
            )
        (make-transformer-weights token-embedding-table
                                  rms-att-weight
                                  wq wk wv wo
                                  rms-ffn-weight
                                  w1 w2 w3
                                  rms-final-weight
                                  freq-cis-real
                                  freq-cis-imag
                                  wcls)
        ))
    ))

(define (read-bytes eport count)
  (let* ([buf (make-blob count)]
         [ret (file-read (endian-port-fileno eport) count buf)])
    (and (= (cadr ret) count) (car ret))))

(define (tokenizer-init conf port)
  (let ((max-token-length (read-uint4 port)))
    (let loop ((i 0)
               (vocab '())
               (vocab-scores '()))
      (if (= i (config-vocab-size conf))
          (values
           (reverse vocab)
           (reverse vocab-scores)
           max-token-length)
          (let* ((score (read-ieee-float32 port))
                 (len (read-uint4 port))
                 (blob (read-bytes port len))
                 (blob-str (blob->string blob)))
            (loop (+ i 1)
                  (cons blob-str vocab)
                  (cons score vocab-scores)
                  )))
      ))
  )

;; BPE Encoding
(define (bpe-encode text vocab vocab-scores)
  (define (index-opt elem lst)
    (let loop ((idx 0) (lst lst))
      (if (null? lst)
          #f
          (if (equal? (car lst) elem)
              idx
              (loop (+ idx 1) (cdr lst))))))
  
  (define (str-lookup str vocab)
    (let ((idx (index-opt str vocab)))
      (if idx idx -1)))
  
  ;; Encode individual characters
  (let* ((chars (string->list text))
         (tokens
          (let loop ((i 0) (chars chars) (tokens '()))
            (if (null? chars)
                (reverse tokens)
                (let* ((char (car chars))
                       (string (string char))
                       (id (str-lookup string vocab)))
                  (if (= id -1)
                      (begin
                        (printf "unable to determine token id for prompt at pos ~A\n" i)
                        (exit 1))
                      (loop (+ i 1) (cdr chars) (cons id tokens))))))))
    
    (let ((vocab-a (list->vector vocab))
          (vocab-scores-a (list->vector vocab-scores)))
      
      (define (merge-tokens tokens)
        (define (find-best-pair tokens)
          (let loop ((tokens tokens) (best-score -1e10) (best-id -1) (best-index 0) (i 0))
            (if (or (null? tokens) (null? (cdr tokens)))
                (values best-id best-index)
                (let* ((token1 (car tokens))
                       (token2 (cadr tokens))
                       (string (string-append (vector-ref vocab-a token1) 
                                             (vector-ref vocab-a token2)))
                       (id (str-lookup string vocab)))
                  (if (= id -1)
                      (loop (cdr tokens) best-score best-id best-index (+ i 1))
                      (let ((score (vector-ref vocab-scores-a id)))
                        (if (< score best-score)
                            (loop (cdr tokens) best-score best-id best-index (+ i 1))
                            (loop (cdr tokens) score id i (+ i 1)))))))))
        
        (let-values (((best-id best-index) (find-best-pair tokens)))
          (if (= best-id -1)
              tokens
              (merge-tokens (append (take tokens best-index)
                                   (cons best-id (drop tokens (+ best-index 2))))))))
      
      (merge-tokens tokens))))

;; RMS normalization using BLAS
(define (rmsnorm out x weight)
  (let* ((epsilon 1e-5)
         (size (f32vector-length x))
         (x-mean-sqsum (fp/ (sdot size x x out) (exact->inexact size)))
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

(define (argmax v)
  (let loop ((i 1) (max-i 0) (max-p (f32vector-ref v 0)))
    (if (= i (f32vector-length v))
        max-i
        (let ((val (f32vector-ref v i)))
          (if (> val max-p)
              (loop (+ i 1) i val)
              (loop (+ i 1) max-i max-p))))))

(define (sample probabilities st)
  (let* ((n (f32vector-length probabilities))
         (r (random:randu! st)))
    (let loop ((cdf 0.0) (i 0))
      (if (= i n)
          (- n 1)  ;; In case of rounding errors
          (let ((new-cdf (+ cdf (f32vector-ref probabilities i))))
            (if (>= new-cdf r)
                i
                (loop new-cdf (+ i 1))))))))

(define (trim-left s)
  (let ((is-whitespace 
         (lambda (c) (member c (list #\space #\page #\newline #\return #\tab)))))
    (let loop ((i 0))
      (if (and (< i (string-length s))
               (is-whitespace (string-ref s i)))
          (loop (+ i 1))
          (if (= i (string-length s))
              ""
              (substring s i (string-length s)))))))

;; Main runtime function
(define (run args)
  (let* ((checkpoint (args-checkpoint args))
         (tokenizer (args-tokenizer args))
         (temperature (args-temperature args))
         (steps (args-steps args))
         (prompt (args-prompt args))
         (seed (or (args-seed args) (current-seconds)))
         ;; Initialize random seed
         (random-state (random:init seed)))
    
    ;; Open checkpoint file
    (let ((port (open-endian-port 'read checkpoint)))
      (set-littlendian! port)
      (let* ((config (read-config port checkpoint))
             (file-size (file-size checkpoint))
             ;; Initialize weights from checkpoint
             (weights
              (checkpoint-init-weights config port 
                                       (config-shared-weights config)
                                       file-size))
             )
        (close-endian-port port)
        
        ;; Adjust steps if needed
        (let ((steps (if (or (<= steps 0) (> steps (config-seq-len config)))
                         (config-seq-len config)
                         steps)))
          
          ;; Load tokenizer
          (let-values (((vocab vocab-scores max-token-length)
                        (let ((port (open-endian-port 'read tokenizer)))
                          (set-littlendian! port)
                          (let-values (((vocab vocab-scores max-token-length)
                                        (tokenizer-init config port)))
                            (close-endian-port port)
                            (values vocab vocab-scores max-token-length)))))

            (let ((vocab-a (list->vector vocab))
                  
                  ;; Initialize model state
                  (state (make-run-state 
                          (make-f32vector (config-dim config) 0.0)
                          (make-f32vector (config-dim config) 0.0)
                          (make-f32vector (config-dim config) 0.0)
                          (make-f32vector (config-dim config) 0.0)
                          (make-f32vector (config-dim config) 0.0)
                          (make-f32vector (* (config-n-heads config)
                                             (config-seq-len config)) 0.0)
                          (make-f32vector (* (config-n-layers config)
                                             (config-seq-len config)
                                             (config-dim config)) 0.0)
                          (make-f32vector (* (config-n-layers config)
                                             (config-seq-len config)
                                             (config-dim config)) 0.0)
                          (make-f32vector (config-dim config) 0.0)
                          (make-f32vector (config-hidden-dim config) 0.0)
                          (make-f32vector (config-hidden-dim config) 0.0)
                          (make-f32vector (config-vocab-size config) 0.0)))

                  (logits-softmax (make-f32vector (config-vocab-size config) 0.0))
                  
                  )
              
              ;; Tokenize prompt
              (let ((start-time (current-seconds))
                    (prompt-tokens
                     (if (> (string-length prompt) 0)
                         (bpe-encode prompt vocab vocab-scores)
                         '())))

                ;; Main generation loop
                (let loop ((token 1)   ;; BOS token
                           (pos 0)
                           (n steps))
                  
                    (if (> n 0)
                        (begin

                          ;; Run the transformer
                          (transformer token pos config state weights)
                          
                          ;; Get next token
                          (let ((next-token
                                 (if (< pos (length prompt-tokens))
                                     (list-ref prompt-tokens pos)
                                     (if (= temperature 0.0)
                                         (argmax (run-state-logits state))
                                         (let ((logits (run-state-logits state)))
                                           ;; Apply temperature
                                           (do ((i 0 (+ i 1)))
                                               ((= i (f32vector-length logits)))
                                             (f32vector-set! 
                                              logits i
                                              (/ (f32vector-ref logits i) temperature)))
                                           (sample (softmax logits-softmax logits 
                                                            (config-vocab-size config))
                                                   random-state))))))
                            
                            ;; Print the token
                            (let ((token-str 
                                   (if (and (= token 1)
                                            (char=? (string-ref (vector-ref vocab-a next-token) 0)
                                                   #\space))
                                       (trim-left (vector-ref vocab-a next-token))
                                       (vector-ref vocab-a next-token))))
                              (display token-str)
                              (flush-output))

                            (loop next-token (+ pos 1) (- n 1))))))
              
              ;; Print timing info
              (let* ((end-time (current-seconds))
                     (time-delta (- end-time start-time))
                     (achieved-toks-per-sec (/ (exact->inexact (- steps 1)) time-delta)))
                (printf "\nachieved tok/s: ~A\n" achieved-toks-per-sec))))))))))
)
