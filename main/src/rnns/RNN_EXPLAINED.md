# Understanding RNNs vs Feedforward Networks

## Key Differences from Your Feedforward Model

### 1. **Sequential Processing with Hidden State**

**Feedforward:**
```python
# Processes entire sequence at once
embedding -> linear1 -> relu -> linear2 -> relu -> output
# No memory between positions
```

**RNN:**
```python
# Processes one timestep at a time, maintaining hidden state
for t in range(seq_len):
    h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)  # Recurrent connection!
    # h_{t-1} carries information from previous timesteps
```

The key innovation: **h_{t-1}** connects the past to the present. Each step depends on all previous steps through this hidden state.

### 2. **Architecture Comparison**

**Feedforward (Your Example):**
```
Input shape: (batch, seq_len, vocab)
         ↓
    [All tokens processed together]
         ↓
  Embedding: (batch, seq_len, emb_dim)
         ↓
  Linear + ReLU (independent per position)
         ↓
  Linear + ReLU (independent per position)
         ↓
  Output: (batch, seq_len, vocab)
```

**RNN (This Implementation):**
```
Input shape: (batch, seq_len)
         ↓
  Embedding: (batch, seq_len, emb_dim)
         ↓
    [Process sequentially]
    For each timestep t:
      x_t: (batch, emb_dim)
      h_t = RNNCell(x_t, h_{t-1})  ← RECURRENT CONNECTION
         ↓
  Collect all h_t
         ↓
  Output projection: (batch, seq_len, vocab)
```

### 3. **The RNN Cell (Core Component)**

```python
class RNNCell:
    def forward(self, x_t, h_prev):
        # x_t: current input (batch, input_size)
        # h_prev: previous hidden state (batch, hidden_size)

        # Combine current input and previous hidden state
        h_t = tanh(
            W_ih @ x_t +      # Input contribution
            W_hh @ h_prev +   # Memory contribution (THIS IS KEY!)
            b_h
        )
        return h_t
```

**Why tanh?** Keeps values in [-1, 1], preventing explosion. Alternative: ReLU (used in modern variants).

### 4. **Backpropagation Through Time (BPTT)**

**Feedforward:** Standard backprop through layers

**RNN:** Must backpropagate through time!

```
Forward pass (unfolded in time):
x_0 → h_0 → x_1 → h_1 → x_2 → h_2 → ... → loss

Backward pass (gradients flow backward through time):
∂L/∂x_0 ← ∂L/∂h_0 ← ∂L/∂h_1 ← ∂L/∂h_2 ← ... ← ∂L/∂loss
```

**Problem:** Gradients get multiplied by W_hh at each timestep
- If |W_hh| > 1: **Exploding gradients** (solution: gradient clipping)
- If |W_hh| < 1: **Vanishing gradients** (solution: LSTM/GRU)

### 5. **Training Loop Differences**

**Feedforward:**
```python
for epoch in range(num_epochs):
    logits = model(input)  # Simple forward pass
    loss = criterion(logits, targets)
    loss.backward()  # Standard backprop
    optimizer.step()
```

**RNN:**
```python
for epoch in range(num_epochs):
    logits, hidden = model(input)  # Returns hidden state too!
    loss = criterion(logits, targets)
    loss.backward()  # BPTT happens here
    clip_grad_norm_(model.parameters(), max_norm=5.0)  # Critical!
    optimizer.step()
```

### 6. **Generation Differences**

**Feedforward:**
```python
# Context is fixed window
context = last_n_tokens
next_token = model(context).argmax()
```

**RNN:**
```python
# Hidden state carries ALL previous context
hidden = model.init_hidden()
for char in seed_text:
    output, hidden = model(char, hidden)  # Build up context

# Now generate
for _ in range(length):
    output, hidden = model(current_char, hidden)  # State persists!
    current_char = sample(output)
```

The hidden state compresses entire history into a fixed-size vector!

## What Makes This "Not Too Simple"?

1. **Multi-layer RNN**: Stacks RNN cells vertically for hierarchical features
2. **Proper BPTT**: Full backpropagation through time, not truncated
3. **Gradient clipping**: Handles exploding gradients
4. **Temperature sampling**: Controlled randomness in generation
5. **Mini-batch training**: Efficient training with batched sequences
6. **Xavier initialization**: Better gradient flow at start

## Common RNN Challenges (Experienced in This Code)

### 1. Vanishing Gradients
As sequences get longer, gradients diminish exponentially:
```
∂L/∂h_0 = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_1/∂h_0
          ↑ Product of many terms < 1
```

**Solution in modern RNNs:** LSTM/GRU with gating mechanisms

### 2. Exploding Gradients
Opposite problem: gradients grow exponentially
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```
This clips gradient norm to prevent instability.

### 3. Computational Cost
- Feedforward: Parallelizes across sequence (fast)
- RNN: Sequential processing (slower, but captures dependencies)

## Next Steps

1. **Run the code:**
   ```bash
   python simple_rnn_text_gen.py
   ```

2. **Experiment:**
   - Try different `hidden_dim` values (128, 512, 1024)
   - Add more layers (`num_layers=3`)
   - Try different `temperature` values in generation
   - Remove gradient clipping and watch it fail!

3. **Understand the output:**
   - Early epochs: Gibberish (random weights)
   - Middle epochs: Learns common patterns
   - Later epochs: Coherent sentences

4. **Upgrade to LSTM/GRU:**
   The vanilla RNN struggles with long sequences. Next step is understanding LSTM gates!

## Visualization of Information Flow

```
Time:      t=0           t=1           t=2
Input:     "T"           "h"           "e"
           |             |             |
Embed:     x_0           x_1           x_2
           |             |             |
           ↓             ↓             ↓
Hidden:  [h_0] ------→ [h_1] ------→ [h_2]
           |             |             |
Output:    o_0           o_1           o_2
           |             |             |
Target:    "h"           "e"           " "

The → arrows are the recurrent connections (W_hh)
These carry information forward through time!
```

## Key Takeaway

**RNNs process sequences one step at a time, maintaining a hidden state that acts as memory.** This allows them to capture temporal dependencies that feedforward networks cannot. The trade-off is more complex training (BPTT) and slower inference.
