# Mathematical Foundations of RNNs

## Core RNN Equations

### Forward Pass

**At each timestep t:**

```
h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b_h)
y_t = W_out · h_t + b_out
```

Where:
- `x_t` ∈ ℝ^(input_dim) - Input at time t
- `h_t` ∈ ℝ^(hidden_dim) - Hidden state at time t
- `h_{t-1}` ∈ ℝ^(hidden_dim) - Previous hidden state
- `y_t` ∈ ℝ^(output_dim) - Output at time t
- `W_ih` ∈ ℝ^(hidden_dim × input_dim) - Input-to-hidden weights
- `W_hh` ∈ ℝ^(hidden_dim × hidden_dim) - Hidden-to-hidden weights (recurrent)
- `W_out` ∈ ℝ^(output_dim × hidden_dim) - Hidden-to-output weights

### Loss Calculation

**Cross-Entropy Loss for sequence:**

```
L = -∑_{t=1}^{T} ∑_{k=1}^{K} y*_{t,k} · log(ŷ_{t,k})
```

Where:
- T = sequence length
- K = number of classes (vocabulary size)
- y*_{t,k} = ground truth (one-hot)
- ŷ_{t,k} = softmax(y_t)_k = predicted probability

**Simplified for single correct class c_t:**

```
L = -∑_{t=1}^{T} log(ŷ_{t,c_t})
```

This is Negative Log-Likelihood (NLL) loss.

## Backpropagation Through Time (BPTT)

### The Challenge

Standard backprop flows through layers. BPTT must flow through **time** too!

**Unfolded computation graph:**
```
x_1 → [RNN] → h_1 → [RNN] → h_2 → ... → h_T → L
       ↑        ↓      ↑      ↓
      h_0      y_1    h_1    y_2
```

### Gradient Flow

**Output layer gradients (easy):**
```
∂L/∂W_out = ∑_t (∂L/∂y_t) · h_t^T
∂L/∂b_out = ∑_t (∂L/∂y_t)
```

**Hidden-to-hidden gradients (hard - through time!):**
```
∂L/∂W_hh = ∑_t (∂L/∂h_t) · h_{t-1}^T
```

But `∂L/∂h_t` depends on ALL future timesteps!

### Recursive Gradient Computation

**Key insight:** Gradient at h_t comes from two sources:
1. Direct loss contribution at time t
2. Indirect contribution through h_{t+1}

```
∂L/∂h_t = ∂L/∂y_t · ∂y_t/∂h_t + ∂L/∂h_{t+1} · ∂h_{t+1}/∂h_t
            ↑                      ↑
        Direct impact          Future impact (recursion!)
```

**Expanding:**
```
∂L/∂h_t = W_out^T · ∂L/∂y_t + W_hh^T · diag(1 - tanh²(h_{t+1})) · ∂L/∂h_{t+1}
```

### Full BPTT Algorithm

```python
# Forward pass: compute all h_t, y_t
for t in 1..T:
    h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_h)
    y_t = W_out @ h_t + b_out

# Compute loss
L = cross_entropy(y, targets)

# Backward pass: compute gradients from T to 1
∂L/∂h_T = W_out^T @ ∂L/∂y_T

for t in T-1..1:
    # Gradient from output
    grad_output = W_out^T @ ∂L/∂y_t

    # Gradient from future (RECURRENT!)
    grad_future = W_hh^T @ ∂L/∂h_{t+1} @ diag(1 - tanh²(h_t))

    # Total gradient
    ∂L/∂h_t = grad_output + grad_future

# Accumulate weight gradients
for t in 1..T:
    ∂L/∂W_hh += ∂L/∂h_t @ h_{t-1}^T
    ∂L/∂W_ih += ∂L/∂h_t @ x_t^T
    ∂L/∂W_out += ∂L/∂y_t @ h_t^T
```

## Vanishing Gradient Problem

### Mathematical Cause

Consider gradient at h_1 (start of sequence):

```
∂L/∂h_1 = ∂L/∂h_T · ∏_{t=2}^{T} ∂h_t/∂h_{t-1}
```

Each term in the product:
```
∂h_t/∂h_{t-1} = W_hh^T · diag(1 - tanh²(h_t))
```

Since `|tanh'(x)| ≤ 1` and typically much less:
```
‖∂h_t/∂h_{t-1}‖ ≤ ‖W_hh‖ · ‖diag(1 - tanh²(h_t))‖ < ‖W_hh‖
```

**If ‖W_hh‖ < 1:**
```
‖∂L/∂h_1‖ ≤ ‖∂L/∂h_T‖ · ‖W_hh‖^{T-1}
            ↓
          Exponential decay!
```

**Example:**
- T = 50 (sequence length)
- ‖W_hh‖ = 0.9
- Gradient multiplier: 0.9^49 ≈ 0.0053 (99.5% vanished!)

### Consequences

1. **Early timesteps don't learn**: Gradients too small to update weights
2. **Can't capture long dependencies**: Network "forgets" distant past
3. **Training instability**: Some weights learn, others don't

## Exploding Gradient Problem

### Mathematical Cause

**If ‖W_hh‖ > 1:**
```
‖∂L/∂h_1‖ ≥ ‖∂L/∂h_T‖ · ‖W_hh‖^{T-1}
            ↓
          Exponential growth!
```

**Example:**
- T = 50
- ‖W_hh‖ = 1.1
- Gradient multiplier: 1.1^49 ≈ 97.4 (97× explosion!)

### Gradient Clipping Solution

**Norm-based clipping:**
```
if ‖g‖ > threshold:
    g ← threshold · g / ‖g‖
```

Where g is the gradient vector.

**In code:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

This rescales gradients while preserving direction.

## Why tanh?

### Properties

1. **Output range:** tanh(x) ∈ [-1, 1]
2. **Derivative:** tanh'(x) = 1 - tanh²(x) ∈ [0, 1]
3. **Zero-centered:** Better than sigmoid (0, 1)

### Derivative visualization

```
tanh'(x) = 1 - tanh²(x)

At x = 0:  tanh'(0) = 1 (maximum gradient flow)
At |x| → ∞: tanh'(x) → 0 (gradient vanishing)
```

This is why:
- Initialization matters (keep activations near 0)
- Deep/long sequences are hard (multiple near-zero derivatives)

## Hidden State as Memory

### Information Capacity

Hidden state h_t ∈ ℝ^d compresses entire history into d dimensions:

```
h_t = f(x_t, h_{t-1})
    = f(x_t, f(x_{t-1}, h_{t-2}))
    = f(x_t, f(x_{t-1}, f(x_{t-2}, ...)))
```

**Theoretical capacity:** Can encode up to d bits of information

**Practical capacity:** Much less due to:
- Gradient problems (vanishing)
- Lossy compression (tanh saturation)
- Training difficulty (local minima)

### Comparison with Feedforward

**Feedforward (context window = k):**
```
y_t = f(x_{t-k+1}, ..., x_{t-1}, x_t)
```
Capacity: k × input_dim parameters

**RNN:**
```
h_t = f(x_t, h_{t-1})  ← Only depends on h_{t-1}, not all history explicitly
```
Capacity: hidden_dim parameters (fixed, regardless of history length!)

This is both a strength (constant parameters) and weakness (information bottleneck).

## Probability Modeling

### Language Modeling Objective

Model probability of sequence:
```
P(x_1, x_2, ..., x_T) = ∏_{t=1}^{T} P(x_t | x_1, ..., x_{t-1})
```

**RNN approximation:**
```
P(x_t | x_1, ..., x_{t-1}) ≈ P(x_t | h_{t-1})
                              ↑
                    Hidden state compresses history
```

### Temperature Sampling

**Standard softmax:**
```
P(x_t = k) = exp(z_k) / ∑_j exp(z_j)
```

**Temperature-scaled:**
```
P(x_t = k) = exp(z_k / T) / ∑_j exp(z_j / T)
```

Where:
- T = 1: Standard (no change)
- T < 1: Sharper distribution (more deterministic)
- T > 1: Flatter distribution (more random)

**Effect on entropy:**
```
H = -∑_k P(k) log P(k)

T → 0: H → 0 (deterministic, picks argmax)
T → ∞: H → log(K) (uniform, maximum entropy)
```

## Computational Complexity

### Forward Pass

**Per timestep:**
- Input-to-hidden: O(hidden_dim × input_dim)
- Hidden-to-hidden: O(hidden_dim²)
- Hidden-to-output: O(output_dim × hidden_dim)

**Total:** O(T × hidden_dim × (input_dim + hidden_dim + output_dim))

### Backward Pass (BPTT)

**Same complexity as forward pass!**
- Must compute gradients for all T timesteps
- Must store all intermediate activations

**Memory:** O(T × hidden_dim) for storing hidden states

### Comparison with Feedforward

**Feedforward:**
- Forward: O(T × hidden_dim × input_dim) - parallelizable across T!
- Backward: O(T × hidden_dim × input_dim) - parallelizable across T!

**RNN:**
- Forward: O(T × hidden_dim²) - sequential (not parallelizable)
- Backward: O(T × hidden_dim²) - sequential (not parallelizable)

This is why Transformers (parallel) largely replaced RNNs in modern NLP.

## Initialization Strategies

### Xavier/Glorot Initialization

For weight matrix W ∈ ℝ^(m × n):

```
W ~ Uniform(-√(6/(m+n)), √(6/(m+n)))
```

or equivalently:

```
W ~ Normal(0, √(2/(m+n)))
```

**Goal:** Preserve variance through layers

**Variance:** Var(Wx) ≈ Var(x) if weights initialized properly

### Why It Matters for RNNs

Bad initialization + long sequences = vanishing/exploding gradients

Good initialization helps early training but doesn't solve fundamental issues (need LSTM/GRU for that).

## Summary of Key Math

| Concept | Equation | Intuition |
|---------|----------|-----------|
| **RNN Cell** | h_t = tanh(W_ih x_t + W_hh h_{t-1} + b) | Current = f(input, previous) |
| **BPTT Gradient** | ∂L/∂h_t = ∂L/∂y_t ∂y_t/∂h_t + ∂L/∂h_{t+1} ∂h_{t+1}/∂h_t | Gradients flow from output + future |
| **Vanishing** | ‖∂L/∂h_1‖ ∝ ‖W_hh‖^{T-1} if ‖W_hh‖ < 1 | Exponential decay over time |
| **Exploding** | ‖∂L/∂h_1‖ ∝ ‖W_hh‖^{T-1} if ‖W_hh‖ > 1 | Exponential growth over time |
| **Gradient Clip** | g ← min(threshold/‖g‖, 1) · g | Cap magnitude, keep direction |
| **Temperature** | P(k) ∝ exp(z_k/T) | Controls sampling randomness |

---

**Next:** Study LSTM equations to see how gating solves vanishing gradients!
