
# LLAMA CHICKEN Scheme

A high-performance LLAMA2 inference implementation in CHICKEN Scheme,
based on Andrej Karpathy's
[llama2.c](https://github.com/karpathy/llama2.c) and its OCaml port
[llama2.ml](https://github.com/jackpeck/llama2.ml).

### System Dependencies
- **CHICKEN Scheme 5.0+**: [The Scheme implementation](https://call-cc.org/)
- **BLAS Library**: For optimized linear algebra (OpenBLAS, Intel MKL, or system BLAS)
- **C Compiler**: GCC or Clang for compiling extensions


## 🛠️ Installation

### 1. Install CHICKEN Scheme
```bash
# Ubuntu/Debian
sudo apt-get install chicken-bin libchicken-dev

# macOS with Homebrew
brew install chicken

# From source
wget https://code.call-cc.org/releases/5.3.0/chicken-5.3.0.tar.gz
tar xzf chicken-5.3.0.tar.gz
cd chicken-5.3.0
make PLATFORM=linux PREFIX=/usr/local
sudo make PLATFORM=linux PREFIX=/usr/local install
```

### 2. Install BLAS Library
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# macOS with Homebrew
brew install openblas

# CentOS/RHEL
sudo yum install openblas-devel
```

### 3. Install Required CHICKEN Extensions
```bash
chicken-install llama
```

## Quick Start

### Basic Text Generation
```bash
# Generate text with default settings
llama-cli -c model.bin -p "Once upon a time"

# Creative generation with temperature
llama-cli -c model.bin -t 0.8 -s 100 -p "The meaning of life is"

# Deterministic generation
llama-cli -c model.bin -t 0.0 -s 50 -p "To be or not to be"
```

### Verify Model Checkpoint
```bash
llama-cli -c model.bin --verify-checkpoint
```

## API Documentation

### Core Data Types

#### `config`
Model configuration parameters.
```scheme
(make-config dim hidden-dim n-layers n-heads n-kv-heads vocab-size seq-len shared-weights)
```

**Fields:**
- `dim`: Model embedding dimension
- `hidden-dim`: FFN hidden layer dimension  
- `n-layers`: Number of transformer layers
- `n-heads`: Number of attention heads
- `n-kv-heads`: Number of key-value heads
- `vocab-size`: Vocabulary size
- `seq-len`: Maximum sequence length
- `shared-weights`: Whether to share input/output embeddings

#### `transformer-weights`
Container for all model parameters.
```scheme
(make-transformer-weights token-embedding-table rms-att-weight wq wk wv wo 
                         rms-ffn-weight w1 w2 w3 rms-final-weight 
                         freq-cis-real freq-cis-imag wcls)
```

#### `run-state`
Runtime state for transformer computation.
```scheme
(make-run-state x xb q k v att key-cache value-cache xb2 hb hb2 logits)
```

**Fields:**
- `x`: Current hidden state
- `xb`, `xb2`: Temporary buffers
- `q`, `k`, `v`: Query, Key, Value vectors
- `att`: Attention scores
- `key-cache`, `value-cache`: Attention caches
- `hb`, `hb2`: FFN hidden buffers
- `logits`: Output logits

### High-Level Functions

#### `(run args)`
Main inference function.
```scheme
(define args (make-args "model.bin" "tokenizer.bin" 0.8 100 "Hello world" #f))
(run args)
```

#### `(bpe-encode text vocab vocab-scores)`
Tokenize text using Byte-Pair Encoding.
```scheme
(bpe-encode "Hello world" vocab vocab-scores)
;; => (15496 1776)
```

#### `(transformer token pos config state weights)`
Run transformer forward pass.
```scheme
(transformer token-id position config state weights)
;; => updated state with new logits
```

### Transformer Components

The modular architecture provides fine-grained control over transformer computation:

#### Token Processing
```scheme
;; Load token embedding
(token-embedding-lookup state weights token-id)

;; Get positional frequencies
(let-values (((freq-real freq-imag) 
              (get-rope-frequencies weights position head-size)))
  ...)
```

#### Attention Components
```scheme
;; Attention normalization
(attention-rmsnorm state weights layer-idx config)

;; Compute Q, K, V matrices
(compute-qkv state weights layer-idx config)

;; Apply rotary position embedding
(apply-rope state config freq-real freq-imag)

;; Cache key-value pairs
(cache-kv state layer-idx position config)

;; Compute attention scores and apply
(compute-attention state layer-idx position config)

;; Output projection
(attention-output state weights layer-idx config)
```

#### Feed-Forward Network
```scheme
;; FFN normalization
(ffn-rmsnorm state weights layer-idx config)

;; Compute W1 and W3 projections
(compute-ffn-w1w3 state weights layer-idx config)

;; Apply SwiGLU activation
(apply-swiglu state config)

;; Final projection
(ffn-output state weights layer-idx config)
```

#### Layer Processing
```scheme
;; Process complete transformer layer
(process-transformer-layer state weights layer-idx position config 
                          freq-real freq-imag)
```

### Utility Functions

#### Vector Operations
```scheme
;; RMS normalization
(rmsnorm output input weights)

;; Matrix-vector multiplication
(matmul output input matrix rows cols)

;; Softmax activation
(softmax output input size)

;; Vector accumulation (residual connections)
(accum target source)
```

#### Sampling Functions
```scheme
;; Greedy sampling (argmax)
(argmax logits-vector)

;; Probabilistic sampling
(sample probability-vector random-state)
```

### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--help` | `-h` | Show help message | - |
| `--checkpoint` | `-c` | Model checkpoint file (required) | - |
| `--tokenizer` | `-k` | Tokenizer file | `tokenizer.bin` |
| `--temperature` | `-t` | Sampling temperature (0.0-2.0) | `0.0` |
| `--steps` | `-s` | Number of tokens to generate | `256` |
| `--prompt` | `-p` | Input prompt text | `""` |
| `--seed` | | Random seed for sampling | Random |
| `--verify-checkpoint` | | Verify checkpoint integrity | `false` |


## 🔧 Configuration

### Model Files
- **Checkpoint**: Binary file containing model weights (`.bin`)
- **Tokenizer**: Binary file containing vocabulary and BPE merge rules

### Temperature Guidelines
- **0.0**: Deterministic (greedy sampling)
- **0.1-0.3**: Focused, coherent output
- **0.5-0.8**: Balanced creativity and coherence
- **0.9-1.2**: Creative, diverse output
- **1.5+**: Highly random, experimental

## Examples

### Interactive REPL Usage
```scheme
(import llama)

;; Load model
(define config (make-config 512 2048 8 8 8 32000 2048 #t))
(define weights (load-checkpoint "model.bin"))
(define state (make-run-state ...))

;; Generate single token
(transformer 1 0 config state weights)
(argmax (run-state-logits state))

;; Custom sampling
(define probs (softmax (make-f32vector 32000) (run-state-logits state) 32000))
(sample probs random-state)
```

### Batch Processing
```scheme
;; Process multiple prompts
(define prompts '("Hello world" "The meaning of life" "Once upon a time"))

(for-each (lambda (prompt)
            (printf "Prompt: ~A\n" prompt)
            (let ((args (make-args "model.bin" "tokenizer.bin" 0.5 50 prompt #f)))
              (run args)
              (newline)))
          prompts)
```


## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Original LLAMA2 paper and implementation by Meta AI
- Andrej Karpathy's C implementation of LLAMA2 [llama2.c](https://github.com/karpathy/llama2.c)
- The LLAMA2 Common Lisp port [llama.cl](https://github.com/snunez1/llama.cl) 
- The LLAMA2 OCaml port [llama2.ml](https://github.com/jackpeck/llama2.ml) 
- BLAS library maintainers for high-performance linear algebra
- CHICKEN Scheme community for excellent libraries
 

## Original README.md

For instructions on conversions to/from .bin format, training and
other background, see the [original repo](https://github.com/karpathy/llama2.c).

