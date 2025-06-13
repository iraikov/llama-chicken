;; Test Cases for Llama2 CHICKEN Scheme Implementation
;; Tests for: bpe-encode, rmsnorm, softmax, and transformer

(import scheme
        (chicken base)
        (chicken format)
        (chicken time)
        (srfi 1)
        (srfi 4)
        (test)
        blas
        llama)

;; Helper function to compare f32vectors with tolerance
(define (f32vector-approx-equal? v1 v2 tolerance)
  (and (= (f32vector-length v1) (f32vector-length v2))
       (let loop ((i 0))
         (cond
          ((= i (f32vector-length v1)) #t)
          ((> (abs (- (f32vector-ref v1 i) (f32vector-ref v2 i))) tolerance) #f)
          (else (loop (+ i 1)))))))

;; Tests for bpe-encode
(test-group "bpe-encode tests"
  
  ;; Test 1: Basic character encoding
  (test "simple character encoding"
        '(0 1 2)
        (bpe-encode "abc" 
                    '("a" "b" "c" "ac" "cb")
                    '(0.0 0.0 0.0 0.5 0.5)))
  
  ;; Test 2: Single character
  (test "single character"
        '(0)
        (bpe-encode "a"
                    '("a" "b" "c")
                    '(0.0 0.0 0.0)))
  
  ;; Test 3: Empty string
  (test "empty string"
        '()
        (bpe-encode ""
                    '("a" "b" "c")
                    '(0.0 0.0 0.0)))
  
  ;; Test 4: Merge with higher score
  (test "merge tokens with higher score"
        '(3)  ; Should merge "a" + "b" -> "ab" (index 3)
        (bpe-encode "ab"
                    '("a" "b" "c" "ab")
                    '(0.0 0.0 0.0 1.0)))
  
  ;; Test 5: Complex merging scenario
  (test "complex merging"
        '(5)  ; Should merge to "abc" (highest score)
        (bpe-encode "abc"
                    '("a" "b" "c" "ab" "bc" "abc")
                    '(0.0 0.0 0.0 0.5 0.5 2.0)))
  )

;; Tests for rmsnorm
(test-group "rmsnorm tests"
  
  ;; Test 1: Basic normalization
  (let ((out (make-f32vector 3 0.0))
        (x (f32vector 1.0 2.0 3.0))
        (weight (f32vector 1.0 1.0 1.0)))
    (rmsnorm out x weight)
    (test-assert "rmsnorm basic test"
                 (f32vector-approx-equal? out 
                                         (f32vector 0.4629 0.9258 1.3887)
                                         0.001)))
  
  ;; Test 2: Zero vector input
  (let ((out (make-f32vector 3 0.0))
        (x (f32vector 0.0 0.0 0.0))
        (weight (f32vector 1.0 1.0 1.0)))
    (rmsnorm out x weight)
    (test-assert "rmsnorm zero input"
                 (f32vector-approx-equal? out 
                                         (f32vector 0.0 0.0 0.0)
                                         0.001)))
  
  ;; Test 3: Different weights
  (let ((out (make-f32vector 3 0.0))
        (x (f32vector 1.0 1.0 1.0))
        (weight (f32vector 2.0 0.5 1.0)))
    (rmsnorm out x weight)
    (test-assert "rmsnorm with different weights"
                 (> (f32vector-ref out 0) (f32vector-ref out 1))))
  
  ;; Test 4: Single element
  (let ((out (make-f32vector 1 0.0))
        (x (f32vector 5.0))
        (weight (f32vector 2.0)))
    (rmsnorm out x weight)
    (test-assert "rmsnorm single element"
                 (f32vector-approx-equal? out 
                                         (f32vector 2.0)
                                         0.001)))
  )

;; Tests for softmax
(test-group "softmax tests"
  
  ;; Test 1: Basic softmax
  (let* ((x (f32vector 1.0 2.0 3.0 0.0))
         (out (make-f32vector 4 0.0))
         (result (softmax out x 3)))
    (test-assert "softmax basic test"
                 (f32vector-approx-equal? result
                                         (f32vector 0.0900 0.2447 0.6652 0.0)
                                         0.001)))
  
  ;; Test 2: Uniform input
  (let* ((x (f32vector 1.0 1.0 1.0))
         (out (make-f32vector 3 0.0))
         (result (softmax out x 3)))
    (test-assert "softmax uniform input"
                 (f32vector-approx-equal? result
                                         (f32vector 0.3333 0.3333 0.3333)
                                         0.001)))
  
  ;; Test 3: Large values (numerical stability)
  (let* ((x (f32vector 1000.0 1001.0 1002.0))
         (out (make-f32vector 3 0.0))
         (result (softmax out x 3)))
    (test-assert "softmax large values"
                 (and (< (f32vector-ref result 0) (f32vector-ref result 1))
                      (< (f32vector-ref result 1) (f32vector-ref result 2)))))
  
  ;; Test 4: Single element
  (let* ((x (f32vector 5.0))
         (out (make-f32vector 1 0.0))
         (result (softmax out x 1)))
    (test-assert "softmax single element"
                 (f32vector-approx-equal? result
                                         (f32vector 1.0)
                                         0.001)))
  
  ;; Test 5: Zero input
  (let* ((x (f32vector 0.0 0.0 0.0))
         (out (make-f32vector 3 0.0))
         (result (softmax out x 3)))
    (test-assert "softmax zero input"
                 (f32vector-approx-equal? result
                                         (f32vector 0.3333 0.3333 0.3333)
                                         0.001))))

(test-group "matmul tests"
  
  ;; Test 1: Basic 2x2 matrix-vector multiplication
  ;; Matrix: [[1, 2], [3, 4]] 
  ;; Vector: [5, 6]
  ;; Expected: [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
  (let ((out (make-f32vector 2 0.0))
        (x (f32vector 5.0 6.0))
        (w (f32vector 1.0 2.0 3.0 4.0))) 
    (matmul out x w 2 2)
    (test-assert "matmul 2x2 basic test"
                 (f32vector-approx-equal? out 
                                         (f32vector 17.0 39.0)
                                         0.001)))
  
  ;; Test 2: Identity matrix
  ;; 3x3 identity matrix with vector [1, 2, 3]
  ;; Expected: [1, 2, 3] (unchanged)
  (let ((out (make-f32vector 3 0.0))
        (x (f32vector 1.0 2.0 3.0))
        (w (f32vector 1.0 0.0 0.0 
                      0.0 1.0 0.0 
                      0.0 0.0 1.0)))
    (matmul out x w 3 3)
    (test-assert "matmul identity matrix"
                 (f32vector-approx-equal? out 
                                         (f32vector 1.0 2.0 3.0)
                                         0.001)))
  
  ;; Test 3: Zero matrix
  ;; 2x2 zero matrix with any vector should give zero output
  (let ((out (make-f32vector 2 0.0))
        (x (f32vector 5.0 7.0))
        (w (f32vector 0.0 0.0 0.0 0.0)))
    (matmul out x w 2 2)
    (test-assert "matmul zero matrix"
                 (f32vector-approx-equal? out 
                                         (f32vector 0.0 0.0)
                                         0.001)))
  
  ;; Test 4: Zero vector
  ;; Any matrix with zero vector input should give zero output
  (let ((out (make-f32vector 2 0.0))
        (x (f32vector 0.0 0.0))
        (w (f32vector 1.0 2.0 3.0 4.0)))
    (matmul out x w 2 2)
    (test-assert "matmul zero vector"
                 (f32vector-approx-equal? out 
                                         (f32vector 0.0 0.0)
                                         0.001)))
  
  ;; Test 5: Rectangular matrix (3x2)
  ;; Matrix: [[1, 2], [3, 4], [5, 6]]
  ;; Vector: [7, 8, 9]
  (let ((out (make-f32vector 2 0.0))
        (x (f32vector 7.0 8.0 9.0))
        (w (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)))
    (matmul out x w 3 2)
    (test-assert "matmul rectangular matrix 3x2"
                 (f32vector-approx-equal? out 
                                         (f32vector 50.0 122.0)
                                         0.001)))
  
  ;; Test 6: Rectangular matrix (2x3)
  ;; Matrix: [[1, 2, 3], [4, 5, 6]]
  ;; Vector: [7, 8, 9]
  ;; Expected: [1*7 + 2*8 + 3*9, 4*7 + 5*8 + 6*9] = [50, 122]
  (let ((out (make-f32vector 3 0.0))
        (x (f32vector 7.0 8.0))
        (w (f32vector 1.0 2.0 3.0 4.0 5.0 6.0)))
    (matmul out x w 2 3)
    (test-assert "matmul rectangular matrix 2x3"
                 (f32vector-approx-equal? out 
                                         (f32vector 23.0 53.0 83.0)
                                         0.001)))
  
  ;; Test 7: Single element (1x1 matrix)
  ;; Matrix: [[5]] with vector [3]
  ;; Expected: [15]
  (let ((out (make-f32vector 1 0.0))
        (x (f32vector 3.0))
        (w (f32vector 5.0)))
    (matmul out x w 1 1)
    (test-assert "matmul 1x1 matrix"
                 (f32vector-approx-equal? out 
                                         (f32vector 15.0)
                                         0.001)))
  
  ;; Test 8: Negative values
  ;; Matrix: [[-1, 2], [3, -4]]
  ;; Vector: [5, -6]
  ;; Expected: [(-1)*5 + 2*(-6), 3*5 + (-4)*(-6)] = [-17, 39]
  (let ((out (make-f32vector 2 0.0))
        (x (f32vector 5.0 -6.0))
        (w (f32vector -1.0 2.0 3.0 -4.0)))
    (matmul out x w 2 2)
    (test-assert "matmul with negative values"
                 (f32vector-approx-equal? out 
                                         (f32vector -17.0 39.0)
                                         0.001)))
  
  ;; Test 9: Fractional values
  ;; Matrix: [[0.5, 1.5], [2.5, 3.5]]
  ;; Vector: [2.0, 4.0]
  ;; Expected: [0.5*2 + 1.5*4, 2.5*2 + 3.5*4] = [7.0, 19.0]
  (let ((out (make-f32vector 2 0.0))
        (x (f32vector 2.0 4.0))
        (w (f32vector 0.5 1.5 2.5 3.5)))
    (matmul out x w 2 2)
    (test-assert "matmul with fractional values"
                 (f32vector-approx-equal? out 
                                         (f32vector 7.0 19.0)
                                         0.001)))
  
  ;; Test 10: Large dimensions (4x4)
  ;; Test that matmul works with larger matrices
  (let ((out (make-f32vector 4 0.0))
        (x (f32vector 1.0 1.0 1.0 1.0))
        (w (f32vector 1.0 1.0 1.0 1.0 
                      2.0 2.0 2.0 2.0 
                      3.0 3.0 3.0 3.0 
                      4.0 4.0 4.0 4.0)))
    (matmul out x w 4 4)
    (test-assert "matmul 4x4 matrix"
                 (f32vector-approx-equal? out 
                                         (f32vector 4.0 8.0 12.0 16.0)
                                         0.001)))

  )


;; Tests for transformer
(test-group "transformer tests"
  
  ;; Setup minimal config and weights for testing
  (define test-config
    (make-config 4     ; dim
                 8     ; hidden-dim  
                 2     ; n-layers
                 2     ; n-heads
                 2     ; n-kv-heads
                 10    ; vocab-size
                 8     ; seq-len
                 #t))  ; shared-weights
  
  (define test-weights
    (make-transformer-weights
     (list->f32vector (iota 40 0.1 0.1))    ; token-embedding-table (vocab-size * dim)
     (make-f32vector 8 1.0)     ; rms-att-weight (n-layers * dim)
     (make-f32vector 32 0.1)    ; wq (n-layers * dim * dim)
     (make-f32vector 32 0.1)    ; wk
     (make-f32vector 32 0.1)    ; wv
     (make-f32vector 32 0.1)    ; wo
     (make-f32vector 8 1.0)     ; rms-ffn-weight
     (make-f32vector 64 0.1)    ; w1 (n-layers * dim * hidden-dim)
     (make-f32vector 64 0.1)    ; w2 (n-layers * hidden-dim * dim)
     (make-f32vector 64 0.1)    ; w3
     (make-f32vector 4 1.0)     ; rms-final-weight
     (make-f32vector 8 0.1)     ; freq-cis-real (seq-len * head-size/2)
     (make-f32vector 8 0.1)     ; freq-cis-imag
     (list->f32vector (iota 40 0.1 0.1))))  ; wcls
  
  (define test-state
    (make-run-state
     (make-f32vector 4 0.0)     ; x
     (make-f32vector 4 0.0)     ; xb
     (make-f32vector 4 0.0)     ; q
     (make-f32vector 4 0.0)     ; k
     (make-f32vector 4 0.0)     ; v
     (make-f32vector 16 0.0)    ; att (n-heads * seq-len)
     (make-f32vector 128 0.0)   ; key-cache (n-layers * seq-len * dim)
     (make-f32vector 128 0.0)   ; value-cache
     (make-f32vector 4 0.0)     ; xb2
     (make-f32vector 8 0.0)     ; hb
     (make-f32vector 8 0.0)     ; hb2
     (make-f32vector 10 0.0)))  ; logits

  (let ((initial-logits (scopy (run-state-logits test-state))))

    ;; Test 1: Basic transformer execution
    (test-assert "transformer executes without error"
                 (begin
                   (transformer 1 0 test-config test-state test-weights)
                 #t))
  
  ;; Test 2: Check logits are modified
    (transformer 1 0 test-config test-state test-weights)
    (test-assert "transformer modifies logits"
                 (not (f32vector-approx-equal? initial-logits 
                                              (run-state-logits test-state)
                                              0.001)))
  
  ;; Test 3: Different tokens produce different results
  (let ((state1 (make-run-state
                 (make-f32vector 4 0.0) (make-f32vector 4 0.0)
                 (make-f32vector 4 0.0) (make-f32vector 4 0.0)
                 (make-f32vector 4 0.0) (make-f32vector 16 0.0)
                 (make-f32vector 128 0.0) (make-f32vector 128 0.0)
                 (make-f32vector 4 0.0) (make-f32vector 8 0.0)
                 (make-f32vector 8 0.0) (make-f32vector 10 0.0)))
        (state2 (make-run-state
                 (make-f32vector 4 0.0) (make-f32vector 4 0.0)
                 (make-f32vector 4 0.0) (make-f32vector 4 0.0)
                 (make-f32vector 4 0.0) (make-f32vector 16 0.0)
                 (make-f32vector 128 0.0) (make-f32vector 128 0.0)
                 (make-f32vector 4 0.0) (make-f32vector 8 0.0)
                 (make-f32vector 8 0.0) (make-f32vector 10 0.0))))
    
    (transformer 1 0 test-config state1 test-weights)
    (transformer 2 0 test-config state2 test-weights)
    
    (test-assert "different tokens produce different logits"
                 (not (f32vector-approx-equal? (run-state-logits state1)
                                               (run-state-logits state2)
                                               0.001))))
  
  ;; Test 4: Sequential positions affect state
  (let ((state (make-run-state
                (make-f32vector 4 0.0) (make-f32vector 4 0.0)
                (make-f32vector 4 0.0) (make-f32vector 4 0.0)
                (make-f32vector 4 0.0) (make-f32vector 16 0.0)
                (make-f32vector 128 0.0) (make-f32vector 128 0.0)
                (make-f32vector 4 0.0) (make-f32vector 8 0.0)
                (make-f32vector 8 0.0) (make-f32vector 10 0.0))))
    
    ;; Run transformer at position 0, then 1
    (transformer 1 0 test-config state test-weights)
    (let ((logits-pos0 (subf32vector (run-state-logits state) 0 (f32vector-length (run-state-logits state)))))
      (transformer 2 1 test-config state test-weights)
      (test-assert "sequential positions produce different results"
                   (not (f32vector-approx-equal? logits-pos0
                                                (run-state-logits state)
                                                0.001)))))
  
  ;; Test 5: Key/value cache is populated
  (let ((state (make-run-state
                (make-f32vector 4 0.0) (make-f32vector 4 0.0)
                (make-f32vector 4 0.0) (make-f32vector 4 0.0)
                (make-f32vector 4 0.0) (make-f32vector 16 0.0)
                (make-f32vector 128 0.0) (make-f32vector 128 0.0)
                (make-f32vector 4 0.0) (make-f32vector 8 0.0)
                (make-f32vector 8 0.0) (make-f32vector 10 0.0))))
    
    (transformer 1 0 test-config state test-weights)
    (test-assert "key cache is populated"
                 (not (every (lambda (x) (= x 0.0))
                            (f32vector->list (run-state-key-cache state)))))
    (test-assert "value cache is populated"
                 (not (every (lambda (x) (= x 0.0))
                             (f32vector->list (run-state-value-cache state))))))
  ))

;; Unit Tests for Transformer Components

;; Helper function to create test configuration
(define (make-test-config)
  (make-config 4     ; dim
               8     ; hidden-dim  
               2     ; n-layers
               2     ; n-heads
               2     ; n-kv-heads
               10    ; vocab-size
               8     ; seq-len
               #t))  ; shared-weights

;; Helper function to create test weights
(define (make-test-weights)
  (make-transformer-weights
   (list->f32vector (iota 40 0.1 0.1))    ; token-embedding-table (vocab-size * dim)
   (make-f32vector 8 1.0)     ; rms-att-weight (n-layers * dim)
   (make-f32vector 32 0.1)    ; wq (n-layers * dim * dim)
   (make-f32vector 32 0.1)    ; wk
   (make-f32vector 32 0.1)    ; wv
   (make-f32vector 32 0.1)    ; wo
   (make-f32vector 8 1.0)     ; rms-ffn-weight
   (list->f32vector (iota 64 0.1 0.001))    ; w1 (n-layers * dim * hidden-dim)
   (list->f32vector (iota 64 0.1 0.001))    ; w2 (n-layers * hidden-dim * dim)
   (make-f32vector 64 0.1)    ; w3
   (make-f32vector 4 1.0)     ; rms-final-weight
   (list->f32vector (iota 8 0.1 0.1))     ; freq-cis-real (seq-len * head-size/2)
   (list->f32vector (iota 8 0.1 0.1))     ; freq-cis-imag
   (list->f32vector (iota 40 0.1 0.1))))  ; wcls

;; Helper function to create test state
(define (make-test-state)
  (make-run-state
   (make-f32vector 4 0.0)     ; x
   (make-f32vector 4 0.0)     ; xb
   (make-f32vector 4 0.0)     ; q
   (make-f32vector 4 0.0)     ; k
   (make-f32vector 4 0.0)     ; v
   (make-f32vector 16 0.0)    ; att (n-heads * seq-len)
   (make-f32vector 128 0.0)   ; key-cache (n-layers * seq-len * dim)
   (make-f32vector 128 0.0)   ; value-cache
   (make-f32vector 4 0.0)     ; xb2
   (make-f32vector 8 0.0)     ; hb
   (make-f32vector 8 0.0)     ; hb2
   (make-f32vector 10 0.0)))  ; logits

;; Token Embedding Lookup
(test-group "token-embedding-lookup tests"
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Test token 0 embedding
    (token-embedding-lookup state weights 0)
    (test-assert "token 0 embedding lookup"
                 (f32vector-approx-equal? (run-state-x state)
                                         (f32vector 0.1 0.2 0.3 0.4)
                                         0.001))
    
    ;; Test token 1 embedding
    (token-embedding-lookup state weights 1)
    (test-assert "token 1 embedding lookup"
                 (f32vector-approx-equal? (run-state-x state)
                                         (f32vector 0.5 0.6 0.7 0.8)
                                         0.001))
    
    ;; Test token 2 embedding
    (token-embedding-lookup state weights 2)
    (test-assert "token 2 embedding lookup"
                 (f32vector-approx-equal? (run-state-x state)
                                         (f32vector 0.9 1.0 1.1 1.2)
                                         0.001))))

;; RoPE Frequency Extraction
(test-group "get-rope-frequencies tests"
  (let ((weights (make-test-weights))
        (head-size 2))
    
    ;; Test position 0
    (let-values (((real-freq imag-freq) (get-rope-frequencies weights 0 head-size)))
      (test-assert "rope frequencies pos 0 real"
                   (f32vector-approx-equal? real-freq (f32vector 0.1) 0.001))
      (test-assert "rope frequencies pos 0 imag"
                   (f32vector-approx-equal? imag-freq (f32vector 0.1) 0.001)))

    ;; Test position 1
    (let-values (((real-freq imag-freq) (get-rope-frequencies weights 1 head-size)))
      (test-assert "rope frequencies pos 1 real"
                   (f32vector-approx-equal? real-freq (f32vector 0.2) 0.001))
      (test-assert "rope frequencies pos 1 imag"
                   (f32vector-approx-equal? imag-freq (f32vector 0.2) 0.001)))))

;; Attention RMS Normalization
(test-group "attention-rmsnorm tests"
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Set up test input
    (f32vector-set! (run-state-x state) 0 1.0)
    (f32vector-set! (run-state-x state) 1 2.0)
    (f32vector-set! (run-state-x state) 2 3.0)
    (f32vector-set! (run-state-x state) 3 4.0)
    
    ;; Test layer 0 normalization
    (attention-rmsnorm state weights 0 config)
    (test-assert "attention rmsnorm layer 0"
                 (not (f32vector-approx-equal? (run-state-xb state)
                                              (f32vector 0.0 0.0 0.0 0.0)
                                              0.001)))
    
    ;; Test that normalization changes the values
    (test-assert "attention rmsnorm modifies values"
                 (not (f32vector-approx-equal? (run-state-xb state)
                                              (run-state-x state)
                                              0.001)))))

;; QKV Computation
(test-group "compute-qkv tests"
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Set up normalized input
    (f32vector-set! (run-state-xb state) 0 0.5)
    (f32vector-set! (run-state-xb state) 1 0.5)
    (f32vector-set! (run-state-xb state) 2 0.5)
    (f32vector-set! (run-state-xb state) 3 0.5)
    
    ;; Compute QKV for layer 0
    (compute-qkv state weights 0 config)
    
    ;; Check that Q, K, V are computed and non-zero
    (test-assert "Q vector is computed"
                 (not (every (lambda (x) (= x 0.0))
                            (f32vector->list (run-state-q state)))))
    (test-assert "K vector is computed"
                 (not (every (lambda (x) (= x 0.0))
                            (f32vector->list (run-state-k state)))))
    (test-assert "V vector is computed"
                 (not (every (lambda (x) (= x 0.0))
                            (f32vector->list (run-state-v state)))))))

;; RoPE Application
(test-group "apply-rope tests"
  (let ((config (make-test-config))
        (state (make-test-state))
        (freq-real (f32vector 1.0))
        (freq-imag (f32vector 0.0)))

    
    ;; Set up Q and K vectors
    (f32vector-set! (run-state-q state) 0 1.0)
    (f32vector-set! (run-state-q state) 1 0.0)
    (f32vector-set! (run-state-q state) 2 1.0)
    (f32vector-set! (run-state-q state) 3 0.0)
    
    (f32vector-set! (run-state-k state) 0 0.0)
    (f32vector-set! (run-state-k state) 1 1.0)
    (f32vector-set! (run-state-k state) 2 0.0)
    (f32vector-set! (run-state-k state) 3 1.0)
    
    (let ((original-q (scopy (run-state-q state)))
          (original-k (scopy (run-state-k state))))
      
      ;; Apply RoPE
      (apply-rope state config freq-real freq-imag)

      ;; Check that vectors are NOT modified (identity rotation)
      (test-assert "Identity RoPE does not modify Q vector"
                   (f32vector-approx-equal? (run-state-q state)
                                           original-q
                                           0.001))
      (test-assert "Identity RoPE does not modify K vector"
                   (f32vector-approx-equal? (run-state-k state)
                                           original-k
                                           0.001))))

    (let ((config (make-test-config))
          (state (make-test-state))
          ;; Use frequencies that create actual rotation (cos(pi/4), sin(pi/4))
          (freq-real (f32vector 0.7071))  ; cos(pi/4) ≈ 0.7071
          (freq-imag (f32vector 0.7071))) ; sin(pi/4) ≈ 0.7071
    
    ;; Set up Q and K vectors
    (f32vector-set! (run-state-q state) 0 1.0)
    (f32vector-set! (run-state-q state) 1 0.0)
    (f32vector-set! (run-state-q state) 2 1.0)
    (f32vector-set! (run-state-q state) 3 0.0)
    
    (f32vector-set! (run-state-k state) 0 0.0)
    (f32vector-set! (run-state-k state) 1 1.0)
    (f32vector-set! (run-state-k state) 2 0.0)
    (f32vector-set! (run-state-k state) 3 1.0)
    
    (let ((original-q (scopy (run-state-q state)))
          (original-k (scopy (run-state-k state))))
      
      ;; Apply RoPE
      (apply-rope state config freq-real freq-imag)

      ;; Check that vectors are modified
      (test-assert "RoPE modifies Q vector"
                   (not (f32vector-approx-equal? (run-state-q state)
                                                 original-q
                                                 0.001)))
      (test-assert "RoPE modifies K vector"
                   (not (f32vector-approx-equal? (run-state-k state)
                                                 original-k
                                                 0.001)))
      
      ;; Test specific rotation values for verification
      ;; For Q[0,1]: (1.0, 0.0) rotated by 45 deg should be (0.7071, 0.7071)
      (test-assert "RoPE applies correct rotation to Q"
                   (and (< (abs (- (f32vector-ref (run-state-q state) 0) 0.7071)) 0.01)
                        (< (abs (- (f32vector-ref (run-state-q state) 1) 0.7071)) 0.01)))
      
      ;; For K[0,1]: (0.0, 1.0) rotated by 45 deg should be (-0.7071, 0.7071)  
      (test-assert "RoPE applies correct rotation to K"
                   (and (< (abs (- (f32vector-ref (run-state-k state) 0) -0.7071)) 0.01)
                        (< (abs (- (f32vector-ref (run-state-k state) 1) 0.7071)) 0.01)))))

  )

;; Key-Value Caching
(test-group "cache-kv tests"
  (let ((config (make-test-config))
        (state (make-test-state)))
    
    ;; Set up K and V vectors
    (f32vector-set! (run-state-k state) 0 1.0)
    (f32vector-set! (run-state-k state) 1 2.0)
    (f32vector-set! (run-state-k state) 2 3.0)
    (f32vector-set! (run-state-k state) 3 4.0)
    
    (f32vector-set! (run-state-v state) 0 5.0)
    (f32vector-set! (run-state-v state) 1 6.0)
    (f32vector-set! (run-state-v state) 2 7.0)
    (f32vector-set! (run-state-v state) 3 8.0)

    
    ;; Cache at layer 0, position 0
    (cache-kv state 0 0 config)
    
    ;; Check that cache is populated
    (test-assert "key cache populated at correct position"
                 (and (= (f32vector-ref (run-state-key-cache state) 0) 1.0)
                      (= (f32vector-ref (run-state-key-cache state) 1) 2.0)
                      (= (f32vector-ref (run-state-key-cache state) 2) 3.0)
                      (= (f32vector-ref (run-state-key-cache state) 3) 4.0)))
    
    (test-assert "value cache populated at correct position"
                 (and (= (f32vector-ref (run-state-value-cache state) 0) 5.0)
                      (= (f32vector-ref (run-state-value-cache state) 1) 6.0)
                      (= (f32vector-ref (run-state-value-cache state) 2) 7.0)
                      (= (f32vector-ref (run-state-value-cache state) 3) 8.0)))))

;; Attention Computation
(test-group "compute-attention tests"
  (let ((config (make-test-config))
        (state (make-test-state)))
    
    ;; Set up Q vector and key cache
    (f32vector-set! (run-state-q state) 0 1.0)
    (f32vector-set! (run-state-q state) 1 0.0)
    (f32vector-set! (run-state-q state) 2 0.0)
    (f32vector-set! (run-state-q state) 3 1.0)
    
    ;; Set up key cache for position 0
    (f32vector-set! (run-state-key-cache state) 0 1.0)
    (f32vector-set! (run-state-key-cache state) 1 0.0)
    (f32vector-set! (run-state-key-cache state) 2 0.0)
    (f32vector-set! (run-state-key-cache state) 3 1.0)
    
    ;; Set up value cache
    (f32vector-set! (run-state-value-cache state) 0 0.5)
    (f32vector-set! (run-state-value-cache state) 1 0.5)
    (f32vector-set! (run-state-value-cache state) 2 0.5)
    (f32vector-set! (run-state-value-cache state) 3 0.5)
    
    ;; Compute attention at position 0
    (compute-attention state 0 0 config)
    
    ;; Check that xb is modified (attention applied)
    (test-assert "attention computation modifies xb"
                 (not (every (lambda (x) (= x 0.0))
                            (f32vector->list (run-state-xb state)))))))

;; Attention Output Projection
(test-group "attention-output tests"
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Set up attention output
    (f32vector-set! (run-state-xb state) 0 1.0)
    (f32vector-set! (run-state-xb state) 1 1.0)
    (f32vector-set! (run-state-xb state) 2 1.0)
    (f32vector-set! (run-state-xb state) 3 1.0)
    
    ;; Apply attention output projection
    (attention-output state weights 0 config)
    
    ;; Check that xb2 is computed
    (test-assert "attention output computed"
                 (not (every (lambda (x) (= x 0.0))
                            (f32vector->list (run-state-xb2 state)))))))


;; FFN RMS Normalization
(test-group "ffn-rmsnorm tests"
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Set up input
    (f32vector-set! (run-state-x state) 0 2.0)
    (f32vector-set! (run-state-x state) 1 4.0)
    (f32vector-set! (run-state-x state) 2 6.0)
    (f32vector-set! (run-state-x state) 3 8.0)
    
    ;; Apply FFN normalization
    (ffn-rmsnorm state weights 0 config)
    
    ;; Check that normalization is applied
    (test-assert "ffn rmsnorm modifies xb"
                 (not (f32vector-approx-equal? (run-state-xb state)
                                              (run-state-x state)
                                              0.001)))))

;; FFN W1/W3 Computation
(test-group "compute-ffn-w1w3 tests"
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Set up normalized input
    (f32vector-set! (run-state-xb state) 0 0.5)
    (f32vector-set! (run-state-xb state) 1 0.5)
    (f32vector-set! (run-state-xb state) 2 0.5)
    (f32vector-set! (run-state-xb state) 3 0.5)
    
    ;; Compute W1 and W3
    (compute-ffn-w1w3 state weights 0 config)
    
    ;; Check that hb and hb2 are computed
    (test-assert "W1 output computed in hb"
                 (not (every (lambda (x) (= x 0.0))
                            (f32vector->list (run-state-hb state)))))
    (test-assert "W3 output computed in hb2"
                 (not (every (lambda (x) (= x 0.0))
                            (f32vector->list (run-state-hb2 state)))))))

;; SwiGLU Activation
(test-group "apply-swiglu tests"
  (let ((config (make-test-config))
        (state (make-test-state)))
    
    ;; Set up hb and hb2 with known values
    (f32vector-set! (run-state-hb state) 0 1.0)   ; SiLU(1.0) ≈ 0.731
    (f32vector-set! (run-state-hb state) 1 0.0)   ; SiLU(0.0) = 0.0
    (f32vector-set! (run-state-hb state) 2 -1.0)  ; SiLU(-1.0) ≈ -0.269
    (f32vector-set! (run-state-hb state) 3 2.0)   ; SiLU(2.0) ≈ 1.761
    
    (f32vector-set! (run-state-hb2 state) 0 2.0)
    (f32vector-set! (run-state-hb2 state) 1 3.0)
    (f32vector-set! (run-state-hb2 state) 2 4.0)
    (f32vector-set! (run-state-hb2 state) 3 5.0)
    
    ;; Apply SwiGLU
    (apply-swiglu state config)
    
    ;; Check results (approximate because of SiLU computation)
    (test-assert "SwiGLU activation applied"
                 (and (> (abs (f32vector-ref (run-state-hb state) 0)) 0.1)
                      (= (f32vector-ref (run-state-hb state) 1) 0.0)  ; 0 * anything = 0
                      (< (f32vector-ref (run-state-hb state) 2) 0.0)  ; negative input
                      (> (f32vector-ref (run-state-hb state) 3) 1.0)))) ; large positive
  )

;; FFN Output Projection
(test-group "ffn-output tests"
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Set up FFN intermediate result
    (f32vector-set! (run-state-hb state) 0 1.0)
    (f32vector-set! (run-state-hb state) 1 1.0)
    (f32vector-set! (run-state-hb state) 2 1.0)
    (f32vector-set! (run-state-hb state) 3 1.0)
    
    ;; Apply FFN output projection
    (ffn-output state weights 0 config)
    
    ;; Check that xb is computed
    (test-assert "ffn output computed"
                 (not (every (lambda (x) (= x 0.0))
                            (f32vector->list (run-state-xb state)))))))

;; Transformer Final RMS Normalization
(test-group "transformer final-rmsnorm tests"
  (let ((weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Set up final input
    (f32vector-set! (run-state-x state) 0 10.0)
    (f32vector-set! (run-state-x state) 1 20.0)
    (f32vector-set! (run-state-x state) 2 30.0)
    (f32vector-set! (run-state-x state) 3 40.0)
    
    (let ((original-x (scopy (run-state-x state))))
      ;; Apply final normalization
      (final-rmsnorm state weights)
      
      ;; Check that x is normalized
      (test-assert "final rmsnorm modifies x"
                   (not (f32vector-approx-equal? (run-state-x state)
                                                original-x
                                                0.001))))))

;; Logits Computation
(test-group "compute-logits tests"
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Set up normalized final state
    (f32vector-set! (run-state-x state) 0 0.25)
    (f32vector-set! (run-state-x state) 1 0.25)
    (f32vector-set! (run-state-x state) 2 0.25)
    (f32vector-set! (run-state-x state) 3 0.25)
    
    ;; Compute logits
    (compute-logits state weights config)
    
    ;; Check that logits are computed
    (test-assert "logits computed"
                 (not (every (lambda (x) (= x 0.0))
                            (f32vector->list (run-state-logits state)))))))


;; Layer Processing Tests
(test-group "process-transformer-layer tests"
  
  ;; Multi-layer sequential processing
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Set up initial token embedding
    (token-embedding-lookup state weights 1)
    (let ((initial-x (scopy (run-state-x state))))
      (let ((states-after-layers
             (let loop ((layer 0)
                        (states-after-layers '()))
               ;; Process through all layers sequentially
               (if (< layer (config-n-layers config))
                   (let-values (((freq-real freq-imag) (get-rope-frequencies weights 0 2)))
                     (process-transformer-layer state weights layer 0 config freq-real freq-imag)
                     ;; Save state after each layer
                     (loop (+ 1 layer)
                           (cons (scopy (run-state-x state)) states-after-layers)))
                   (reverse states-after-layers)))))
      
      ;; Test that each layer produces different output
      (test-assert "layer 0 modifies initial state"
                   (not (f32vector-approx-equal? initial-x 
                                                (car states-after-layers) 0.001)))
      
      (test-assert "layer 1 modifies layer 0 output"
                   (not (f32vector-approx-equal? (car states-after-layers)
                                                 (cadr states-after-layers) 0.001)))
      
      ;; Test that final state is different from all intermediate states
      (test-assert "final state differs from initial"
                   (not (f32vector-approx-equal? initial-x
                                                (run-state-x state) 0.001))))))
  
  ;; Sequential position processing with attention
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Process positions sequentially
    (let ((position-states '())
          (attention-scores '()))
      
      ;; Process position 0 (can only attend to itself)
      (token-embedding-lookup state weights 1)
      (let-values (((freq-real freq-imag) (get-rope-frequencies weights 0 2)))
        (process-transformer-layer state weights 0 0 config freq-real freq-imag))
      (set! position-states (cons (scopy (run-state-x state)) position-states))
      
      ;; Process position 1 (can attend to pos 0 and 1)
      (token-embedding-lookup state weights 2)
      (let-values (((freq-real freq-imag) (get-rope-frequencies weights 1 2)))
        (process-transformer-layer state weights 0 1 config freq-real freq-imag))
      (set! position-states (cons (scopy (run-state-x state)) position-states))
      
      ;; Process position 2 (can attend to pos 0, 1, and 2)
      (token-embedding-lookup state weights 3)
      (let-values (((freq-real freq-imag) (get-rope-frequencies weights 2 2)))
        (process-transformer-layer state weights 0 2 config freq-real freq-imag))
      (set! position-states (cons (scopy (run-state-x state)) position-states))
      
      (set! position-states (reverse position-states))
      
      ;; Test that different positions produce different outputs
      (test-assert "position 0 and 1 produce different states"
                   (not (f32vector-approx-equal? (car position-states)
                                                 (cadr position-states) 0.001)))
      
      (test-assert "position 1 and 2 produce different states"
                   (not (f32vector-approx-equal? (cadr position-states)
                                                 (caddr position-states) 0.001)))
      
      ;; Test that key-value cache is populated correctly
      (let ((cache-layer-0-pos-0 (subf32vector (run-state-key-cache state) 0 4))
            (cache-layer-0-pos-1 (subf32vector (run-state-key-cache state) 4 8))
            (cache-layer-0-pos-2 (subf32vector (run-state-key-cache state) 8 12)))
        
        (test-assert "key cache populated for different positions"
                     (and (not (every (lambda (x) (= x 0.0)) (f32vector->list cache-layer-0-pos-0)))
                          (not (every (lambda (x) (= x 0.0)) (f32vector->list cache-layer-0-pos-1)))
                          (not (every (lambda (x) (= x 0.0)) (f32vector->list cache-layer-0-pos-2)))))
        
        (test-assert "different positions have different cached keys"
                     (not (f32vector-approx-equal? cache-layer-0-pos-0 cache-layer-0-pos-1 0.001))))))
  
  ;; Cross-layer attention consistency
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state1 (make-test-state))
        (state2 (make-test-state)))
    
    ;; Set up identical initial states
    (token-embedding-lookup state1 weights 1)
    (token-embedding-lookup state2 weights 1)
    
    ;; Process through layers in different orders to test independence
    ;; State1: Layer 0 then Layer 1
    (let-values (((freq-real freq-imag) (get-rope-frequencies weights 0 2)))
      (process-transformer-layer state1 weights 0 0 config freq-real freq-imag)
      (process-transformer-layer state1 weights 1 0 config freq-real freq-imag))
    
    ;; State2: Process layer 1 on fresh embedding (should be different)
    (let-values (((freq-real freq-imag) (get-rope-frequencies weights 0 2)))
      (process-transformer-layer state2 weights 1 0 config freq-real freq-imag))
    
    ;; Test that different layer sequences produce different results
    (test-assert "different layer processing sequences produce different results"
                 (not (f32vector-approx-equal? (run-state-x state1)
                                               (run-state-x state2) 0.001))))
  
  ;; Layer weight isolation test
  (let ((config (make-test-config))
        (weights (make-test-weights)))
    
    ;; Process same input through different layers
    (let ((state-layer-0 (make-test-state))
          (state-layer-1 (make-test-state)))
      
      ;; Set up identical initial states
      (token-embedding-lookup state-layer-0 weights 1)
      (token-embedding-lookup state-layer-1 weights 1)
      
      ;; Process with layer 0
      (let-values (((freq-real freq-imag) (get-rope-frequencies weights 0 2)))
        (process-transformer-layer state-layer-0 weights 0 0 config freq-real freq-imag))
      
      ;; Process with layer 1  
      (let-values (((freq-real freq-imag) (get-rope-frequencies weights 0 2)))
        (process-transformer-layer state-layer-1 weights 1 0 config freq-real freq-imag))
      
      ;; Different layers should produce different results
      (test-assert "different layers produce different outputs for same input"
                   (not (f32vector-approx-equal? (run-state-x state-layer-0) 
                                                 (run-state-x state-layer-1) 0.001)))))
  
  ;; Attention causality test
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Process a sequence: pos 0, then pos 1
    ;; Position 1 should be able to attend to position 0, but not vice versa
    
    ;; Process position 0
    (token-embedding-lookup state weights 1)
    (let-values (((freq-real freq-imag) (get-rope-frequencies weights 0 2)))
      (process-transformer-layer state weights 0 0 config freq-real freq-imag))
    
    (let ((pos-0-state (scopy (run-state-x state))))
      
      ;; Process position 1 (should be influenced by cached pos 0)
      (token-embedding-lookup state weights 2)
      (let-values (((freq-real freq-imag) (get-rope-frequencies weights 1 2)))
        (process-transformer-layer state weights 0 1 config freq-real freq-imag))
      (let ((pos-1-with-context (scopy (run-state-x state))))
        
        ;; Now process position 1 in isolation (fresh state has no cached context)
        (let ((isolated-state (make-test-state)))
          (token-embedding-lookup isolated-state weights 2)
          (let-values (((freq-real freq-imag) (get-rope-frequencies weights 1 2)))
            (process-transformer-layer isolated-state weights 0 1 config freq-real freq-imag))
          
          ;; Position 1 with context should be different from position 1 in isolation
          ;; (This tests that attention is actually using the cached context)
          (test-assert "position 1 with context differs from position 1 in isolation"
                       (not (f32vector-approx-equal? pos-1-with-context
                                                    (run-state-x isolated-state) 0.001)))))))
  
  ;; Residual connection preservation
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Test that residual connections preserve some information from input
    (token-embedding-lookup state weights 1)
    (let ((initial-embedding (scopy (run-state-x state))))
      
      ;; Process through one layer
      (let-values (((freq-real freq-imag) (get-rope-frequencies weights 0 2)))
        (process-transformer-layer state weights 0 0 config freq-real freq-imag))
      
      ;; The output should contain some trace of the input due to residual connections
      ;; Calculate magnitude
      (let ((output-magnitude 
             (sqrt (let loop ((i 0) (sum 0.0))
                     (if (= i (f32vector-length (run-state-x state)))
                         sum
                         (let ((x (f32vector-ref (run-state-x state) i)))
                           (loop (+ i 1) (+ sum (* x x))))))))
            (input-magnitude 
             (sqrt (let loop ((i 0) (sum 0.0))
                     (if (= i (f32vector-length initial-embedding))
                         sum
                         (let ((x (f32vector-ref initial-embedding i)))
                           (loop (+ i 1) (+ sum (* x x)))))))))
        
        (test-assert "output magnitude is reasonable compared to input"
                     (and (> output-magnitude 0.0)
                          (< output-magnitude (* 10.0 input-magnitude)))))))
  
  ;; Multi-token sequence processing
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Process a sequence of tokens
    (let ((sequence-states
           (let loop ((pos 0)
                      (token-sequence '(1 2 3))
                      (sequence-states '()))
      
             ;; Process each token in sequence
             (if (null? token-sequence)
                 (reverse sequence-states)
                 (begin
                   ;; Set up token embedding
                   (token-embedding-lookup state weights (car token-sequence))
        
                   ;; Process through all layers for this position
                   (do ((layer 0 (+ layer 1)))
                       ((= layer (config-n-layers config)))
                     (let-values (((freq-real freq-imag) (get-rope-frequencies weights pos 2)))
                       (process-transformer-layer state weights layer pos config freq-real freq-imag)))
        
                   ;; Save the final state for this position
                   (loop (+ pos 1)
                         (cdr  token-sequence)
                         (cons (scopy (run-state-x state)) sequence-states))
                   ))
             )))
      
      ;; Test that the sequence produces increasingly complex representations
      (test-assert "sequence processing: all positions produce different states"
                   (and (not (f32vector-approx-equal? (car sequence-states) 
                                                      (cadr sequence-states) 0.001))
                        (not (f32vector-approx-equal? (cadr sequence-states)
                                                      (caddr sequence-states) 0.001))))
      
      ;; Test that the final position can access information from all previous positions
      ;; by checking that the key-value cache is fully populated
      (let ((total-cache-entries (* (config-n-layers config) (length sequence-states) (config-dim config)))) 
        (let ((non-zero-cache-entries 
               (length (filter (lambda (x) (not (= x 0.0)))
                              (f32vector->list (run-state-key-cache state))))))

          (test-assert "key cache is substantially populated after sequence"
                       (>= non-zero-cache-entries (* 0.5 total-cache-entries))))))))

(test-group "transformer component integration tests"
  (let ((config (make-test-config))
        (weights (make-test-weights))
        (state (make-test-state)))
    
    ;; Test full pipeline step by step
    (token-embedding-lookup state weights 1)
    (test-assert "token embedding"
                 (not (every (lambda (x) (= x 0.0))
                            (f32vector->list (run-state-x state)))))
    
    (let-values (((freq-real freq-imag) (get-rope-frequencies weights 0 2)))
      (attention-rmsnorm state weights 0 config)
      (test-assert "attention rmsnorm"
                   (not (every (lambda (x) (= x 0.0))
                              (f32vector->list (run-state-xb state)))))
      
      (compute-qkv state weights 0 config)
      (test-assert "qkv computation"
                   (and (not (every (lambda (x) (= x 0.0))
                                   (f32vector->list (run-state-q state))))
                        (not (every (lambda (x) (= x 0.0))
                                   (f32vector->list (run-state-k state))))
                        (not (every (lambda (x) (= x 0.0))
                                   (f32vector->list (run-state-v state))))))
      
      (apply-rope state config freq-real freq-imag)
      (cache-kv state 0 0 config)
      (compute-attention state 0 0 config)
      (attention-output state weights 0 config)
      (accum (run-state-x state) (run-state-xb2 state))

      
      (test-assert "attention block complete"
                   (not (every (lambda (x) (= x 0.0))
                              (f32vector->list (run-state-x state)))))
      
      (ffn-rmsnorm state weights 0 config)
      (compute-ffn-w1w3 state weights 0 config)
      (apply-swiglu state config)
      (ffn-output state weights 0 config)
      (accum (run-state-x state) (run-state-xb state))
      
      (test-assert "ffn block complete"
                   (not (every (lambda (x) (= x 0.0))
                              (f32vector->list (run-state-x state)))))
      
      (final-rmsnorm state weights)
      (compute-logits state weights config)
      
      (test-assert "final processing complete"
                   (not (every (lambda (x) (= x 0.0))
                              (f32vector->list (run-state-logits state))))))))

;; Integration test: argmax and sample functions
(test-group "utility function tests"
  
  ;; Test argmax
  (test "argmax finds correct index"
        2
        (argmax (f32vector 0.1 0.3 0.8 0.2)))
  
  ;; Test argmax with single element
  (test "argmax single element"
        0
        (argmax (f32vector 5.0)))
  
  ;; Test argmax with equal values
  (test "argmax equal values returns first"
        0
        (argmax (f32vector 1.0 1.0 1.0))))

;; Run all tests
(test-exit)
