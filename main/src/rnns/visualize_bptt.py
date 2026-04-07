"""
Visualization of Backpropagation Through Time (BPTT).
Shows how gradients flow backward through the unfolded RNN.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


class TinyRNN(nn.Module):
    """Minimal RNN for visualization purposes"""

    def __init__(self, input_size=3, hidden_size=4, output_size=3):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        self.W_out = nn.Parameter(torch.randn(output_size, hidden_size) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, input_size)
        Returns: outputs, hidden_states, all_hiddens
        """
        batch_size, seq_len, _ = x.shape

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size)

        outputs = []
        all_hiddens = [hidden]

        for t in range(seq_len):
            x_t = x[:, t, :]

            # RNN cell computation
            hidden = torch.tanh(
                x_t @ self.W_ih.t() +
                hidden @ self.W_hh.t() +
                self.b_h
            )

            # Output projection
            out = hidden @ self.W_out.t() + self.b_out
            outputs.append(out)
            all_hiddens.append(hidden)

        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden, all_hiddens


def trace_gradients_through_time(model, x, targets):
    """
    Perform forward and backward pass, tracking gradient flow.

    Returns gradient magnitudes at each timestep.
    """
    model.zero_grad()

    # Forward pass
    outputs, final_hidden, all_hiddens = model(x)

    # Loss (simplified)
    loss = ((outputs - targets) ** 2).mean()

    # Backward pass
    loss.backward()

    # Extract gradient information
    grad_info = {
        'W_ih_grad': model.W_ih.grad.norm().item(),
        'W_hh_grad': model.W_hh.grad.norm().item(),
        'W_out_grad': model.W_out.grad.norm().item(),
    }

    return grad_info, all_hiddens


def visualize_rnn_unfolding():
    """Visualize how RNN unfolds in time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Folded RNN
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Folded RNN (Compact Representation)', fontsize=14, fontweight='bold')

    # Draw folded RNN
    cell = FancyBboxPatch((3, 4), 4, 2, boxstyle="round,pad=0.1",
                          edgecolor='blue', facecolor='lightblue', linewidth=3)
    ax1.add_patch(cell)
    ax1.text(5, 5, 'RNN\nCell', ha='center', va='center', fontsize=12, fontweight='bold')

    # Input arrow
    ax1.arrow(5, 2, 0, 1.5, head_width=0.3, head_length=0.3, fc='green', ec='green', linewidth=2)
    ax1.text(5, 1.5, r'$x_t$', ha='center', fontsize=12)

    # Output arrow
    ax1.arrow(5, 6.5, 0, 1.5, head_width=0.3, head_length=0.3, fc='red', ec='red', linewidth=2)
    ax1.text(5, 8.5, r'$y_t$', ha='center', fontsize=12)

    # Recurrent connection (loop)
    loop = mpatches.FancyBboxPatch((6.5, 4.5), 2, 1, boxstyle="round,pad=0.1",
                                   edgecolor='orange', facecolor='none', linewidth=3)
    ax1.add_patch(loop)
    ax1.arrow(8.2, 5, -1, 0, head_width=0.3, head_length=0.3, fc='orange', ec='orange', linewidth=2)
    ax1.text(8.5, 5, r'$h_{t-1}$', ha='left', fontsize=11)

    # Right: Unfolded RNN
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Unfolded RNN (Backpropagation Through Time)', fontsize=14, fontweight='bold')

    timesteps = 3
    x_positions = [2, 6, 10]

    for i, x_pos in enumerate(x_positions):
        # Hidden state box
        cell = FancyBboxPatch((x_pos-0.8, 4), 1.6, 2, boxstyle="round,pad=0.05",
                              edgecolor='blue', facecolor='lightblue', linewidth=2)
        ax2.add_patch(cell)
        ax2.text(x_pos, 5, f'$h_{i}$', ha='center', va='center', fontsize=11, fontweight='bold')

        # Input
        ax2.arrow(x_pos, 2, 0, 1.5, head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=1.5)
        ax2.text(x_pos, 1.5, f'$x_{i}$', ha='center', fontsize=10)

        # Output
        ax2.arrow(x_pos, 6.5, 0, 1.5, head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=1.5)
        ax2.text(x_pos, 8.5, f'$y_{i}$', ha='center', fontsize=10)

        # Recurrent connection
        if i < timesteps - 1:
            arrow = FancyArrowPatch((x_pos + 0.8, 5), (x_positions[i+1] - 0.8, 5),
                                   arrowstyle='->', mutation_scale=20, linewidth=2,
                                   color='orange')
            ax2.add_patch(arrow)

        # Loss arrows (backward)
        ax2.arrow(x_pos, 8, 0, 0.5, head_width=0, head_length=0, fc='purple', ec='purple',
                 linewidth=1, linestyle='--', alpha=0.5)

    # Gradient flow annotation
    ax2.text(12, 8.5, 'Gradient\nFlow', ha='center', fontsize=10, color='purple',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))
    ax2.arrow(11, 8, -7, 0, head_width=0.3, head_length=0.3, fc='purple', ec='purple',
             linewidth=2, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('/Users/ashritkuma.samudrala/lnex/ex_llm_rag/main/src/rnn_tutorial/rnn_unfolding.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("RNN unfolding visualization saved to rnn_unfolding.png")


def demonstrate_gradient_flow():
    """
    Demonstrate how gradients flow through different sequence lengths.
    Shows the vanishing gradient problem.
    """
    model = TinyRNN(input_size=3, hidden_size=4, output_size=3)

    sequence_lengths = [5, 10, 20, 30, 50]
    gradient_norms = {'W_ih': [], 'W_hh': [], 'W_out': []}

    print("\n=== Gradient Flow Analysis ===")
    print(f"{'Seq Length':<12} {'W_ih grad':<12} {'W_hh grad':<12} {'W_out grad':<12}")
    print("-" * 50)

    for seq_len in sequence_lengths:
        # Create random input and target
        x = torch.randn(1, seq_len, 3)
        targets = torch.randn(1, seq_len, 3)

        # Trace gradients
        grad_info, _ = trace_gradients_through_time(model, x, targets)

        gradient_norms['W_ih'].append(grad_info['W_ih_grad'])
        gradient_norms['W_hh'].append(grad_info['W_hh_grad'])
        gradient_norms['W_out'].append(grad_info['W_out_grad'])

        print(f"{seq_len:<12} {grad_info['W_ih_grad']:<12.4f} "
              f"{grad_info['W_hh_grad']:<12.4f} {grad_info['W_out_grad']:<12.4f}")

    # Visualize gradient decay
    plt.figure(figsize=(10, 6))
    plt.plot(sequence_lengths, gradient_norms['W_ih'], 'o-', label='W_ih (input → hidden)', linewidth=2)
    plt.plot(sequence_lengths, gradient_norms['W_hh'], 's-', label='W_hh (hidden → hidden)', linewidth=2)
    plt.plot(sequence_lengths, gradient_norms['W_out'], '^-', label='W_out (hidden → output)', linewidth=2)

    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.title('Gradient Flow vs Sequence Length\n(Demonstrates Vanishing Gradient Problem)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Add annotation
    plt.annotate('Gradients shrink with longer sequences!\nThis is the vanishing gradient problem.',
                xy=(sequence_lengths[-1], gradient_norms['W_hh'][-1]),
                xytext=(35, 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig('/Users/ashritkuma.samudrala/lnex/ex_llm_rag/main/src/rnn_tutorial/gradient_flow.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nGradient flow visualization saved to gradient_flow.png")


def demonstrate_exploding_gradients():
    """Show what happens without gradient clipping"""
    print("\n=== Demonstrating Exploding Gradients ===")

    model = TinyRNN(input_size=3, hidden_size=4, output_size=3)

    # Initialize with larger weights (prone to explosion)
    nn.init.uniform_(model.W_hh, -2, 2)

    x = torch.randn(1, 20, 3)
    targets = torch.randn(1, 20, 3)

    # Without clipping
    model.zero_grad()
    outputs, _, _ = model(x)
    loss = ((outputs - targets) ** 2).mean()
    loss.backward()

    grad_before = model.W_hh.grad.norm().item()
    print(f"Gradient norm WITHOUT clipping: {grad_before:.2f}")

    # With clipping
    model.zero_grad()
    outputs, _, _ = model(x)
    loss = ((outputs - targets) ** 2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    grad_after = model.W_hh.grad.norm().item()
    print(f"Gradient norm WITH clipping (max=5.0): {grad_after:.2f}")
    print(f"Reduction: {grad_before / max(grad_after, 1e-8):.2f}x")

    print("\n💡 Key Insight: Gradient clipping prevents unstable training!")


def visualize_hidden_state_evolution():
    """
    Visualize how hidden state evolves over time.
    Shows information accumulation.
    """
    model = TinyRNN(input_size=3, hidden_size=4, output_size=3)

    # Create a simple pattern
    seq_len = 15
    x = torch.sin(torch.linspace(0, 4*np.pi, seq_len)).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 3)

    with torch.no_grad():
        _, _, all_hiddens = model(x)

    # Convert to numpy
    hidden_states = torch.stack(all_hiddens[1:]).squeeze().numpy()  # (seq_len, hidden_dim)

    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Hidden state heatmap
    im = ax1.imshow(hidden_states.T, aspect='auto', cmap='RdBu', interpolation='nearest')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Hidden Dimension', fontsize=12)
    ax1.set_title('Hidden State Evolution Through Time', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Activation Value')

    # Hidden state magnitude over time
    hidden_norms = np.linalg.norm(hidden_states, axis=1)
    ax2.plot(hidden_norms, linewidth=2, marker='o')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Hidden State Magnitude', fontsize=12)
    ax2.set_title('How Information Accumulates in Hidden State', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ashritkuma.samudrala/lnex/ex_llm_rag/main/src/rnn_tutorial/hidden_state_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Hidden state evolution visualization saved to hidden_state_evolution.png")


def main():
    print("="*60)
    print("RNN BACKPROPAGATION THROUGH TIME VISUALIZATION")
    print("="*60)

    # 1. Show RNN unfolding
    print("\n1. Generating RNN unfolding diagram...")
    visualize_rnn_unfolding()

    # 2. Demonstrate gradient flow
    print("\n2. Analyzing gradient flow through different sequence lengths...")
    demonstrate_gradient_flow()

    # 3. Show exploding gradients
    demonstrate_exploding_gradients()

    # 4. Visualize hidden state
    print("\n4. Visualizing hidden state evolution...")
    visualize_hidden_state_evolution()

    print("\n" + "="*60)
    print("SUMMARY OF KEY CONCEPTS")
    print("="*60)
    print("""
1. UNFOLDING: RNN is "unrolled" through time for BPTT
   - Each timestep becomes a layer in the computation graph
   - Gradients flow backward through all timesteps

2. VANISHING GRADIENTS:
   - Gradients shrink exponentially with sequence length
   - Caused by repeated multiplication by W_hh
   - Solution: LSTM/GRU with gating mechanisms

3. EXPLODING GRADIENTS:
   - Gradients grow exponentially (opposite problem)
   - Caused by |W_hh| > 1
   - Solution: Gradient clipping (clip_grad_norm)

4. HIDDEN STATE:
   - Acts as memory, accumulating information over time
   - Each timestep modifies and passes state forward
   - Critical for capturing long-range dependencies

5. COMPUTATIONAL COST:
   - BPTT is O(T) where T is sequence length
   - Must store all intermediate states for backprop
   - Truncated BPTT trades accuracy for efficiency
""")

    print("\nAll visualizations saved!")


if __name__ == "__main__":
    main()
