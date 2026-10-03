;; Llama2 Command Line Interface

(import scheme
        (chicken base)
        (chicken condition)
        (chicken process-context)
        (chicken format)
        (chicken string)
        (chicken file)
        getopt-long
        llama-common
        (only llama make-blas-inference-backend)
        (only llama-am make-am-inference-backend))

;; Inference backends selectable with --backend
(define backend-names '("blas" "crunch"))

;; Define command-line option grammar
(define option-grammar
  `((help
     "Show this help message and exit"
     (single-char #\h)
     (value #f))
    
    (checkpoint
     "Path to the Llama2 model checkpoint file (required)"
     (single-char #\c)
     (value #t))
    
    (tokenizer
     "Path to the tokenizer file (default tokenizer.bin)"
     (single-char #\k)
     (value
      (optional FILE)
      (default "tokenizer.bin"))
     (required #f))
    
    (verify-checkpoint
     (value #f))
     
    (temperature
     "Sampling temperature (0.0 = greedy, higher = more random)"
     (single-char #\t)
     (value
      (required TEMP)
      (predicate ,(lambda (s) 
                    (let ((n (string->number s)))
                      (and n (>= n 0.0)))))
      (transformer ,string->number)
      (default "0")))
    
    (steps
     "Number of tokens to generate"
     (single-char #\s)
     (value (required NUM)
            (predicate ,(lambda (s)
                          (let ((n (string->number s)))
                            (and n (integer? n) (> n 0)))))
            (transformer ,string->number)
            (default "256")))
    
    (prompt
     "Input prompt text"
     (single-char #\p)
     (value (optional PROMPT)
            (default "")))
    
    (backend
     "Numerical backend: blas or crunch (array-morphisms)"
     (single-char #\b)
     (value
      (required NAME)
      (predicate ,(lambda (s) (member s backend-names)))))

    (threads
     "Number of threads for the crunch backend"
     (single-char #\j)
     (value (required NUM)
            (predicate ,(lambda (s)
                          (let ((n (string->number s)))
                            (and n (integer? n) (exact? n) (> n 0)))))
            (transformer ,string->number)))

    (seed
     "Random seed"
     (single-char #\r)
     (value
            (required SEED)
            (predicate ,(lambda (s) 
                          (or (not s) (let ((n (string->number s))) n))))
            (transformer ,string->number)
            (default #f)))

    )
  )

;; Show usage information
(define (show-usage program-name)
  (printf "Usage: ~A [options]\n\n" program-name)
  (printf "Llama2 Text Generation\n\n")
  (printf "Options:\n")
  (printf "  -h, --help            Show this help message and exit\n")
  (printf "  -c, --checkpoint FILE Path to the Llama2 model checkpoint file (required)\n")
  (printf "  -k, --tokenizer FILE  Path to the tokenizer file (default tokenizer.bin)\n")
  (printf "  -t, --temperature NUM Sampling temperature (default: 0.0)\n")
  (printf "                        0.0 = greedy sampling (deterministic)\n")
  (printf "                        Higher values = more random/creative output\n")
  (printf "  -s, --steps NUM       Number of tokens to generate (default: 256)\n")
  (printf "  -p, --prompt TEXT     Input prompt text (default: empty)\n")
  (printf "  -r, --seed NUM        Random seed (default: current time)\n")
  (printf "  -b, --backend NAME    Numerical backend (default: blas)\n")
  (printf "                        blas   = CHICKEN blas egg\n")
  (printf "                        crunch = array-morphisms with the crunch backend\n")
  (printf "  -j, --threads NUM     Number of threads for the crunch backend\n")
  (printf "  --verify-checkpoint   Print a summary of the checkpoint weights and exit\n")
  (printf "\n")
  (printf "Long options take their value after '=', e.g. --steps=100.\n")
  (printf "\n")
  (printf "Examples:\n")
  (printf "  ~A -c model.bin -p \"Once upon a time\"\n" program-name)
  (printf "  ~A --checkpoint=model.bin --temperature=0.8 --steps=100\n" program-name)
  (printf "  ~A -c model.bin -t 0.5 -s 50 -p \"The meaning of life is\"\n" program-name)
  (printf "  ~A -c model.bin --backend=crunch --threads=4 -p \"Once upon a time\"\n" program-name))

;; Parse command line options and validate
(define (parse-options args)
  (handle-exceptions exn
    (begin
      (printf "Error parsing command line: ~A\n"
              ((condition-property-accessor 'exn 'message) exn))
      (show-usage "llama-cli")
      (exit 1))
    (getopt-long args option-grammar)
    ))

;; Extract option value with default
(define (get-option options key default)
  (let ((opt (assoc key options)))
    (if opt (cdr opt) default)))

;; Validate that required files exist
(define (validate-files checkpoint tokenizer)
  (unless (file-exists? checkpoint)
    (printf "Error: Checkpoint file does not exist: ~A\n" checkpoint)
    (exit 1))
  
  (unless (file-exists? tokenizer)
    (printf "Error: Tokenizer file does not exist: ~A\n" tokenizer)
    (exit 1)))

;; Main function
(define (main args)

  (if (null? args)
      (begin
        (show-usage "llama-cli")
        (exit 1))
      
      (let ((program-name (car args))
            (parsed-options (parse-options args)))
        
        ;; Check for help option
        (when (get-option parsed-options 'help #f)
          (show-usage "llama-cli")
          (exit 0))
        
        ;; Extract and validate options
        (let* ((checkpoint (get-option parsed-options 'checkpoint #f))
               (tokenizer (get-option parsed-options 'tokenizer "tokenizer.bin"))
               (temperature (get-option parsed-options 'temperature 0.0))
               (steps (get-option parsed-options 'steps 256))
               (prompt (get-option parsed-options 'prompt ""))
               (seed (get-option parsed-options 'seed #f))
               (verify-checkpoint (get-option parsed-options 'verify-checkpoint #f))
               (backend-name (get-option parsed-options 'backend "blas"))
               (threads (get-option parsed-options 'threads #f))
               )
         
          ;; Validate required options
          (unless checkpoint
            (printf "Error: Checkpoint file is required\n")
            (show-usage "llama-cli")
            (exit 1))
          
          ;; Validate file existence
          (validate-files checkpoint tokenizer)

          (when verify-checkpoint
            (verify-checkpoint-data checkpoint)
            (exit 0))

          ;; Validate numeric conversions
          (unless temperature
            (printf "Error: Invalid temperature value: ~A\n" temperature)
            (exit 1))
          
          (unless steps
            (printf "Error: Invalid steps value: ~A\n" steps)
            (exit 1))
          
          ;; Print configuration
          (printf "Llama2 Text Generation\n")
          (printf "======================\n")
          (printf "Checkpoint: ~A\n" checkpoint)
          (printf "Temperature: ~A\n" temperature)
          (printf "Steps: ~A\n" steps)
          (printf "Backend: ~A\n" backend-name)
          (printf "Prompt: \"~A\"\n" prompt)
          (printf "\nGenerating text...\n\n")
          
          ;; Create args and run the model
          (let ((llama-args (make-args checkpoint tokenizer temperature steps prompt seed))
                (backend (if (string=? backend-name "crunch")
                             (make-am-inference-backend threads: threads)
                             (make-blas-inference-backend))))
            (run-with-backend llama-args backend))
          ))
      ))


;; Entry point
(main (command-line-arguments))
