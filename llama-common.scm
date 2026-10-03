;; Llama2 Inference in CHICKEN Scheme: common components.
;;
;; This module contains the parts of the inference program that do not
;; depend on the numerical backend: configuration and weight records,
;; checkpoint and tokenizer readers, byte-pair encoding, sampling, and
;; token generation loop.  A numerical backend is a Yasos object that
;; implements the inference-backend protocol defined below; the
;; generation loop drives any such object.

(module llama-common

        ( make-args args? args-checkpoint args-tokenizer args-temperature
          args-steps args-prompt args-seed

          make-config config? config-dim config-hidden-dim config-n-layers
          config-n-heads config-n-kv-heads config-vocab-size config-seq-len
          config-shared-weights

          make-transformer-weights transformer-weights?
          transformer-weights-token-embedding-table
          transformer-weights-rms-att-weight
          transformer-weights-wq transformer-weights-wk
          transformer-weights-wv transformer-weights-wo
          transformer-weights-rms-ffn-weight
          transformer-weights-w1 transformer-weights-w2 transformer-weights-w3
          transformer-weights-rms-final-weight
          transformer-weights-freq-cis-real transformer-weights-freq-cis-imag
          transformer-weights-wcls

          f32vector-fold f64vector-fold
          read-config checkpoint-init-weights tokenizer-init
          bpe-encode argmax sample trim-left softmax-f32!
          print-transformer-weights verify-checkpoint-data

          ;; inference-backend protocol
          inference-backend? backend-name backend-prepare! backend-forward!
          run-with-backend
          )

        (import
         scheme
         (chicken base)
         (chicken condition)
         (chicken time)
         (chicken foreign)
         (chicken file)
         (chicken file posix)
         (chicken format)
         (chicken string)
         (chicken bytevector)
         (chicken number-vector)
         (srfi 1)
         (srfi 4)
         (endian-sequence)
         (endian-port)
         (prefix random-mtzig random:)
         (except yasos object object-with-ancestors operate-as)
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

;; Model weights.  With the flat layout every layered tensor is a single
;; f32vector holding all layers one after another.  With the per-layer
;; layout every layered tensor is a Scheme vector of n-layers f32vectors.
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
  (token-embedding-table transformer-weights-token-embedding-table)
  (rms-att-weight transformer-weights-rms-att-weight)
  (wq transformer-weights-wq)
  (wk transformer-weights-wk)
  (wv transformer-weights-wv)
  (wo transformer-weights-wo)
  (rms-ffn-weight transformer-weights-rms-ffn-weight)
  (w1 transformer-weights-w1)
  (w2 transformer-weights-w2)
  (w3 transformer-weights-w3)
  (rms-final-weight transformer-weights-rms-final-weight)
  (freq-cis-real transformer-weights-freq-cis-real)
  (freq-cis-imag transformer-weights-freq-cis-imag)
  (wcls transformer-weights-wcls))


;; Inference backend protocol.
;;
;; (backend-name b)                         => symbol naming the backend
;; (backend-prepare! b config port file-size)
;;     reads the weights from the checkpoint port (positioned just after
;;     the configuration header) and allocates the run state
;; (backend-forward! b token pos)           => logits f32vector
(define-predicate inference-backend?)
(define-operation (backend-name b))
(define-operation (backend-prepare! b config port file-size))
(define-operation (backend-forward! b token pos))


;; Wall-clock time in seconds from a monotonic clock, used to report the
;; generation rate.
(foreign-declare "#include <time.h>")
(define wall-clock-seconds
  (foreign-lambda* double ()
    "struct timespec ts;
     clock_gettime(CLOCK_MONOTONIC, &ts);
     C_return((double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec);
"))


;; Folds over the elements of one or more numeric vectors of equal length.
(define (f64vector-fold f x0 v . rest)
  (let ((n (f64vector-length v))
        (vs (cons v rest)))
    (let loop ((i 0) (ax x0))
      (if (= i n)
          ax
          (loop (+ i 1)
                (apply f (append (map (lambda (v) (f64vector-ref v i)) vs)
                                 (list ax))))))))

(define (f32vector-fold f x0 v . rest)
  (let ((n (f32vector-length v))
        (vs (cons v rest)))
    (let loop ((i 0) (ax x0))
      (if (= i n)
          ax
          (loop (+ i 1)
                (apply f (append (map (lambda (v) (f32vector-ref v i)) vs)
                                 (list ax))))))))


(define (read-config port checkpoint)
  (handle-exceptions exn
    (begin
      (printf "Couldn't open file ~A: ~A\n" checkpoint exn)
      (exit 1))
    (set-littlendian! port)
    ;; A negative vocabulary size marks a checkpoint whose classifier
    ;; weights are stored separately from the token embedding table.
    (let* ((dim (read-uint4 port))
           (hidden-dim (read-uint4 port))
           (n-layers (read-uint4 port))
           (n-heads (read-uint4 port))
           (n-kv-heads (read-uint4 port))
           (vocab-size (read-int4 port))
           (seq-len (read-uint4 port))
           (shared-weights (> vocab-size 0)))
      (make-config dim hidden-dim n-layers n-heads n-kv-heads
                   (abs vocab-size) seq-len shared-weights))))


;; Reads the weights that follow the configuration header.  The checkpoint
;; stores each layered tensor for all layers in turn, so the per-layer
;; layout is obtained by reading the same total amount in n-layers chunks.
(define (checkpoint-init-weights conf port shared-weights file-size
                                 #!key (layout 'flat))
  (let* ((vocab-size (config-vocab-size conf))
         (dim (config-dim conf))
         (n-layers (config-n-layers conf))
         (hidden-dim (config-hidden-dim conf))
         (seq-len (config-seq-len conf))
         (n-heads (config-n-heads conf))
         (read-layered
          (case layout
            ((flat)
             (lambda (size)
               (read-ieee-float32-vector port (* n-layers size))))
            ((per-layer)
             (lambda (size)
               (let ((v (make-vector n-layers)))
                 (do ((l 0 (+ l 1)))
                     ((= l n-layers) v)
                   (vector-set! v l (read-ieee-float32-vector port size))))))
            (else
             (error 'checkpoint-init-weights "unknown weight layout" layout)))))

    (let* ((token-embedding-table (read-ieee-float32-vector port (* vocab-size dim)))
           (rms-att-weight (read-layered dim))
           (wq (read-layered (* dim dim)))
           (wk (read-layered (* dim dim)))
           (wv (read-layered (* dim dim)))
           (wo (read-layered (* dim dim)))
           (rms-ffn-weight (read-layered dim))
           (w1 (read-layered (* dim hidden-dim)))
           (w2 (read-layered (* hidden-dim dim)))
           (w3 (read-layered (* dim hidden-dim)))
           (rms-final-weight (read-ieee-float32-vector port dim))
           (freq-cis-real
            (read-ieee-float32-vector port (* seq-len (/ (/ dim n-heads) 2))))
           (freq-cis-imag
            (read-ieee-float32-vector port (* seq-len (/ (/ dim n-heads) 2))))
           (wcls
            (if shared-weights
                token-embedding-table
                (read-ieee-float32-vector
                 port
                 (quotient (- file-size (file-position (endian-port-fileno port))) 4)))))
      (make-transformer-weights token-embedding-table
                                rms-att-weight
                                wq wk wv wo
                                rms-ffn-weight
                                w1 w2 w3
                                rms-final-weight
                                freq-cis-real
                                freq-cis-imag
                                wcls))))


(define (read-bytes eport count)
  (let* ((buf (make-bytevector count))
         (ret (file-read (endian-port-fileno eport) count buf)))
    (and (= (cadr ret) count) (car ret))))

;; Reads the vocabulary.  Token strings are stored as UTF-8 bytes.
(define (tokenizer-init conf port)
  (let ((max-token-length (read-uint4 port)))
    (let loop ((i 0)
               (vocab '())
               (vocab-scores '()))
      (if (= i (config-vocab-size conf))
          (values (reverse vocab) (reverse vocab-scores) max-token-length)
          (let* ((score (read-ieee-float32 port))
                 (len (read-uint4 port))
                 (bytes (read-bytes port len)))
            (loop (+ i 1)
                  (cons (utf8->string bytes) vocab)
                  (cons score vocab-scores)))))))


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

;; Softmax of the first size elements of x, written into out.  Elements of
;; x past size are copied to out unchanged.
(define (softmax-f32! out x size)
  (let ((n (f32vector-length x))
        (max-val (let loop ((i 1) (m (f32vector-ref x 0)))
                   (if (= i size) m (loop (+ i 1) (max m (f32vector-ref x i)))))))
    (let ((exp-sum
           (let loop ((i 0) (s 0.0))
             (if (= i size)
                 s
                 (let ((e (exp (- (f32vector-ref x i) max-val))))
                   (f32vector-set! out i e)
                   (loop (+ i 1) (+ s e)))))))
      (do ((i 0 (+ i 1)))
          ((= i size))
        (f32vector-set! out i (/ (f32vector-ref out i) exp-sum))))
    (do ((i size (+ i 1)))
        ((>= i n))
      (f32vector-set! out i (f32vector-ref x i)))
    out))


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

;; Main pretty printer for transformer weights (flat layout)
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

      weights)))


;; Token generation loop.  Loads the checkpoint into the given backend,
;; reads the tokenizer, and generates up to (args-steps args) tokens,
;; printing each one as it is produced.
(define (run-with-backend args backend)
  (let* ((checkpoint (args-checkpoint args))
         (tokenizer (args-tokenizer args))
         (temperature (args-temperature args))
         (steps (args-steps args))
         (prompt (args-prompt args))
         (seed (or (args-seed args) (current-seconds)))
         (random-state (random:init seed)))

    ;; Read the configuration and let the backend load the weights
    (let* ((port (open-endian-port 'read checkpoint))
           (config (begin (set-littlendian! port)
                           (read-config port checkpoint))))
      (backend-prepare! backend config port (file-size checkpoint))
      (close-endian-port port)

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

          (let* ((vocab-a (list->vector vocab))
                 (vocab-size (config-vocab-size config))
                 (probs (make-f32vector vocab-size 0.0))
                 (prompt-tokens
                  (list->vector
                   (if (> (string-length prompt) 0)
                       (bpe-encode prompt vocab vocab-scores)
                       '())))
                 (start-time (wall-clock-seconds)))

            ;; Main generation loop
            (let loop ((token 1)   ;; BOS token
                       (pos 0)
                       (n steps))
              (when (> n 0)
                (let* ((logits (backend-forward! backend token pos))
                       (next-token
                        (cond
                         ((< pos (vector-length prompt-tokens))
                          (vector-ref prompt-tokens pos))
                         ((= temperature 0.0)
                          (argmax logits))
                         (else
                          (do ((i 0 (+ i 1)))
                              ((= i vocab-size))
                            (f32vector-set! logits i (/ (f32vector-ref logits i) temperature)))
                          (sample (softmax-f32! probs logits vocab-size) random-state))))
                       (token-str
                        (if (and (= token 1)
                                 (> (string-length (vector-ref vocab-a next-token)) 0)
                                 (char=? (string-ref (vector-ref vocab-a next-token) 0)
                                         #\space))
                            (trim-left (vector-ref vocab-a next-token))
                            (vector-ref vocab-a next-token))))
                  (display token-str)
                  (flush-output)
                  (loop next-token (+ pos 1) (- n 1)))))

            ;; Print timing info
            (let* ((elapsed (- (wall-clock-seconds) start-time))
                   (toks-per-sec (if (> elapsed 0.0)
                                     (/ (exact->inexact (- steps 1)) elapsed)
                                     0.0)))
              (printf "\nachieved tok/s: ~A\n" toks-per-sec))))))))
)
