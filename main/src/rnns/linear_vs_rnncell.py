"""
Why can't we use nn.Linear instead of nn.RNNCell?

This demonstrates the fundamental difference between feedforward (Linear)
and recurrent (RNNCell) computation.
"""

import torch
import torch.nn as nn


# ==============================================================================
# THE KEY DIFFERENCE
# ==============================================================================

print("="*80)
print("UNDERSTANDING nn.Linear vs nn.RNNCell")
print("="*80)

# Setup
batch_size = 2
input_dim = 5
hidden_dim = 8

current_input = torch.randn(batch_size, input_dim)
previous_hidden = torch.randn(batch_size, hidden_dim)

print(f"\nInput shapes:")
print(f"  current_input (x_t): {current_input.shape}")
print(f"  previous_hidden (h_{{t-1}}): {previous_hidden.shape}")

# ==============================================================================
# Option 1: nn.Linear - DOESN'T WORK for RNN!
# ==============================================================================

print("\n" + "-"*80)
print("Option 1: Using nn.Linear")
print("-"*80)

linear = nn.Linear(input_dim, hidden_dim)

# Problem: Linear only takes ONE input!
# We need to incorporate BOTH current_input AND previous_hidden
try:
    # This only uses current input - IGNORES PREVIOUS HIDDEN STATE!
    hidden_linear = linear(current_input)
    print(f"✗ Linear output: {hidden_linear.shape}")
    print("  Problem: This ONLY uses current_input!")
    print("  The previous_hidden state is completely ignored!")
    print("  → No memory, no recurrence, no temporal dependency!")
except Exception as e:
    print(f"Error: {e}")

# What if we try to pass both?
try:
    # Can't do this - Linear expects single input
    hidden_linear = linear(current_input, previous_hidden)  # This will fail!
except TypeError as e:
    print(f"\n✗ Can't pass two inputs to Linear: {e}")

# ==============================================================================
# Option 2: nn.RNNCell - CORRECT for RNN
# ==============================================================================

print("\n" + "-"*80)
print("Option 2: Using nn.RNNCell")
print("-"*80)

rnn_cell = nn.RNNCell(input_dim, hidden_dim)

# RNNCell takes TWO inputs: current input AND previous hidden!
hidden_rnn = rnn_cell(current_input, previous_hidden)
print(f"✓ RNNCell output: {hidden_rnn.shape}")
print("  This combines BOTH current_input AND previous_hidden!")
print("  → Has memory, has recurrence, captures temporal dependency!")

# ==============================================================================
# Option 3: Building RNNCell from TWO nn.Linear layers
# ==============================================================================

print("\n" + "-"*80)
print("Option 3: Manually implement RNNCell using TWO nn.Linear layers")
print("-"*80)

class ManualRNNCell(nn.Module):
    """
    This shows what nn.RNNCell does internally.
    You NEED two separate Linear layers!
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Linear layer for current input x_t
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim, bias=False)

        # Linear layer for previous hidden state h_{t-1}
        # THIS IS THE RECURRENT CONNECTION!
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Bias (shared)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x_t, h_prev):
        """
        RNN equation: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
        """
        # Process current input
        input_contribution = self.input_to_hidden(x_t)

        # Process previous hidden state (RECURRENT CONNECTION!)
        hidden_contribution = self.hidden_to_hidden(h_prev)

        # Combine both + bias, then apply nonlinearity
        h_t = torch.tanh(input_contribution + hidden_contribution + self.bias)

        return h_t


manual_rnn = ManualRNNCell(input_dim, hidden_dim)
hidden_manual = manual_rnn(current_input, previous_hidden)
print(f"✓ Manual RNN output: {hidden_manual.shape}")
print("\nBreakdown:")
print("  1. input_to_hidden(x_t):    Uses current input")
print("  2. hidden_to_hidden(h_prev): Uses previous hidden (RECURRENCE!)")
print("  3. Combine: h_t = tanh(W_ih @ x_t + W_hh @ h_prev + b)")

# ==============================================================================
# VISUAL COMPARISON
# ==============================================================================

print("\n" + "="*80)
print("VISUAL COMPARISON")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ FEEDFORWARD (nn.Linear) - No Memory                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   x_t ──→ [Linear] ──→ h_t                                      │
│                                                                  │
│   Each timestep is INDEPENDENT!                                 │
│   h_t only depends on x_t                                       │
│   No connection to previous timesteps                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ RECURRENT (nn.RNNCell) - Has Memory                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌──────────┐                                 │
│                    │          │                                 │
│   x_t ──→ ┌────────▼──────┐  │                                 │
│           │               │  │                                 │
│  h_prev ──→   RNNCell    ├──┘──→ h_t                           │
│           │               │                                     │
│           └───────────────┘                                     │
│                                                                  │
│   Each timestep depends on BOTH x_t AND h_prev!                │
│   h_t = f(x_t, h_{t-1})                                        │
│   Recurrent connection carries information through time        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
""")

# ==============================================================================
# MATHEMATICAL FORMULAS
# ==============================================================================

print("\n" + "="*80)
print("MATHEMATICAL FORMULAS")
print("="*80)

print("""
nn.Linear:
──────────
    y = W @ x + b

    • ONE weight matrix: W ∈ ℝ^(output_dim × input_dim)
    • Takes ONE input: x
    • No temporal dependency


nn.RNNCell:
───────────
    h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)

    • TWO weight matrices:
        - W_ih ∈ ℝ^(hidden_dim × input_dim)   [input-to-hidden]
        - W_hh ∈ ℝ^(hidden_dim × hidden_dim)  [hidden-to-hidden] ← RECURRENT!
    • Takes TWO inputs: x_t and h_{t-1}
    • Temporal dependency through h_{t-1}


Manual RNNCell (using two Linears):
────────────────────────────────────
    h_t = tanh(Linear1(x_t) + Linear2(h_{t-1}) + b)

    • Linear1: input-to-hidden transformation
    • Linear2: hidden-to-hidden transformation (RECURRENT!)
    • You need BOTH to implement recurrence!
""")

# ==============================================================================
# DEMONSTRATE INFORMATION FLOW OVER TIME
# ==============================================================================

print("\n" + "="*80)
print("INFORMATION FLOW OVER TIME")
print("="*80)

# Create sequence
seq_length = 5
x_sequence = torch.randn(seq_length, batch_size, input_dim)

print(f"\nProcessing sequence of length {seq_length}...\n")

# Process with RNNCell (has memory)
print("With RNNCell (has memory):")
hidden = torch.zeros(batch_size, hidden_dim)
rnn_outputs = []

for t in range(seq_length):
    hidden = rnn_cell(x_sequence[t], hidden)  # hidden accumulates information!
    rnn_outputs.append(hidden.mean().item())
    print(f"  t={t}: hidden state mean = {hidden.mean().item():.4f} "
          f"(accumulated info from timesteps 0 to {t})")

# Process with Linear (no memory)
print("\nWith Linear (no memory):")
linear_outputs = []

for t in range(seq_length):
    output = linear(x_sequence[t])  # Each step is independent!
    linear_outputs.append(output.mean().item())
    print(f"  t={t}: output mean = {output.mean().item():.4f} "
          f"(ONLY info from timestep {t})")

# ==============================================================================
# THE ANSWER
# ==============================================================================

print("\n" + "="*80)
print("WHY CAN'T WE USE nn.Linear INSTEAD OF nn.RNNCell?")
print("="*80)

print("""
1. DIFFERENT NUMBER OF INPUTS:
   • nn.Linear: Takes ONE input (current x_t)
   • nn.RNNCell: Takes TWO inputs (current x_t AND previous h_{t-1})

2. DIFFERENT NUMBER OF WEIGHT MATRICES:
   • nn.Linear: ONE weight matrix (W)
   • nn.RNNCell: TWO weight matrices (W_ih and W_hh)

   The W_hh matrix is the RECURRENT CONNECTION - this is what makes it an RNN!

3. INFORMATION FLOW:
   • nn.Linear: Each timestep is independent
     - h_t depends ONLY on x_t
     - No memory of previous timesteps

   • nn.RNNCell: Each timestep depends on history
     - h_t depends on BOTH x_t AND h_{t-1}
     - h_{t-1} contains information from all previous timesteps
     - Has memory!

4. CAN YOU IMPLEMENT RNNCell USING Linear?
   YES! But you need TWO Linear layers:

   class RNNCell:
       def __init__(self):
           self.linear_input = nn.Linear(input_dim, hidden_dim)   # For x_t
           self.linear_hidden = nn.Linear(hidden_dim, hidden_dim) # For h_{t-1}

       def forward(self, x_t, h_prev):
           return tanh(self.linear_input(x_t) + self.linear_hidden(h_prev))

   A SINGLE nn.Linear cannot do this because it only takes one input!

BOTTOM LINE:
───────────
The recurrent connection (W_hh @ h_{t-1}) is what makes RNNs different.
nn.Linear doesn't have this - it only transforms its input, with no
connection to previous timesteps.

To implement recurrence, you need:
• A way to combine TWO inputs (current + previous)
• A separate weight matrix for the recurrent connection
• This is exactly what nn.RNNCell provides!
""")

# ==============================================================================
# BONUS: Show parameter counts
# ==============================================================================

print("\n" + "="*80)
print("PARAMETER COUNTS")
print("="*80)

linear_params = sum(p.numel() for p in linear.parameters())
rnn_params = sum(p.numel() for p in rnn_cell.parameters())
manual_params = sum(p.numel() for p in manual_rnn.parameters())

print(f"\nnn.Linear({input_dim}, {hidden_dim}):")
print(f"  Parameters: {linear_params:,}")
print(f"  Breakdown: W({hidden_dim}×{input_dim}) + b({hidden_dim}) = {hidden_dim * input_dim + hidden_dim}")

print(f"\nnn.RNNCell({input_dim}, {hidden_dim}):")
print(f"  Parameters: {rnn_params:,}")
print(f"  Breakdown: W_ih({hidden_dim}×{input_dim}) + W_hh({hidden_dim}×{hidden_dim}) + b({hidden_dim})")
print(f"            = {hidden_dim * input_dim} + {hidden_dim * hidden_dim} + {hidden_dim} = {rnn_params}")

print(f"\nManual RNNCell (two Linears):")
print(f"  Parameters: {manual_params:,}")
print(f"  Same as nn.RNNCell!")

print(f"\nRNNCell has {rnn_params - linear_params:,} MORE parameters than Linear")
print(f"This extra capacity comes from W_hh, the recurrent weight matrix!")
