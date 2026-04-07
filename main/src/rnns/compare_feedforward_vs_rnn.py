"""
Side-by-side comparison of Feedforward vs RNN for text generation.
This demonstrates why recurrent connections matter for sequential data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from simple_rnn_text_gen import TextDataset


class FeedforwardTextModel(nn.Module):
    """
    Simple feedforward network (like your example).
    Processes all positions independently - no recurrence.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.vocab_size = vocab_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Feedforward layers (no recurrence!)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        Returns: (batch_size, seq_len, vocab_size)
        """
        # Embed: (batch_size, seq_len, embedding_dim)
        embeds = self.embedding(x)

        # Each position processed independently!
        # No information sharing between timesteps
        h1 = F.relu(self.fc1(embeds))  # (batch_size, seq_len, hidden_dim)
        h2 = F.relu(self.fc2(h1))       # (batch_size, seq_len, hidden_dim)
        logits = self.fc_out(h2)        # (batch_size, seq_len, vocab_size)

        return logits


class SimpleRNN(nn.Module):
    """
    Simplified RNN (from previous implementation).
    Processes sequentially with hidden state.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNNCell(embedding_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        batch_size, seq_len = x.shape

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        embeds = self.embedding(x)

        outputs = []
        for t in range(seq_len):
            x_t = embeds[:, t, :]
            hidden = self.rnn(x_t, hidden)  # Recurrent step!
            outputs.append(hidden)

        output_seq = torch.stack(outputs, dim=1)
        logits = self.fc_out(output_seq)

        return logits, hidden


def train_model(model, dataset, is_rnn=False, num_epochs=100, batch_size=32, lr=0.002, device='cpu'):
    """Train either model type"""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    model_type = "RNN" if is_rnn else "Feedforward"

    print(f"\n=== Training {model_type} ===")
    for epoch in range(num_epochs):
        inputs, targets = dataset.get_batch(batch_size)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if is_rnn:
            logits, _ = model(inputs)
        else:
            logits = model(inputs)

        logits_flat = logits.reshape(-1, model.vocab_size)
        targets_flat = targets.reshape(-1)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()

        if is_rnn:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    return losses


def generate_text_feedforward(model, dataset, start_text, length, temperature, device):
    """Generate text with feedforward model - uses local context only"""
    model.eval()

    generated = start_text
    context = [dataset.char_to_idx[ch] for ch in start_text[-10:]]  # Limited context window

    with torch.no_grad():
        for _ in range(length):
            # Only use last N characters as context (no full history)
            x = torch.LongTensor([context[-10:]]).to(device)

            logits = model(x)
            logits = logits[0, -1, :] / temperature

            probs = F.softmax(logits, dim=0)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            generated += dataset.idx_to_char[next_idx]
            context.append(next_idx)

    return generated


def generate_text_rnn(model, dataset, start_text, length, temperature, device):
    """Generate text with RNN - uses full history via hidden state"""
    model.eval()

    input_seq = [dataset.char_to_idx[ch] for ch in start_text]
    generated = start_text

    hidden = torch.zeros(1, model.hidden_dim, device=device)

    with torch.no_grad():
        # Build up hidden state from start text
        for i in range(len(input_seq) - 1):
            x = torch.LongTensor([[input_seq[i]]]).to(device)
            _, hidden = model(x, hidden)

        current_idx = input_seq[-1]

        for _ in range(length):
            x = torch.LongTensor([[current_idx]]).to(device)
            logits, hidden = model(x, hidden)

            logits = logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=0)
            current_idx = torch.multinomial(probs, num_samples=1).item()

            generated += dataset.idx_to_char[current_idx]

    return generated


def compare_perplexity(model_ff, model_rnn, dataset, device, is_rnn_model1=False, is_rnn_model2=True):
    """
    Compare model perplexity on test data.
    Lower perplexity = better model (more confident/accurate predictions)
    """
    model_ff.eval()
    model_rnn.eval()

    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Get test batch
    inputs, targets = dataset.get_batch(32)
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.no_grad():
        # Feedforward
        logits_ff = model_ff(inputs)
        loss_ff = criterion(logits_ff.reshape(-1, model_ff.vocab_size), targets.reshape(-1))
        perplexity_ff = torch.exp(loss_ff).item()

        # RNN
        logits_rnn, _ = model_rnn(inputs)
        loss_rnn = criterion(logits_rnn.reshape(-1, model_rnn.vocab_size), targets.reshape(-1))
        perplexity_rnn = torch.exp(loss_rnn).item()

    return perplexity_ff, perplexity_rnn


def main():
    # Training data
    text = """
    Recurrent neural networks process sequences one element at a time.
    Each timestep updates a hidden state that carries information forward.
    This hidden state acts as memory, encoding what the network has seen so far.
    Feedforward networks process all positions independently without memory.
    The recurrent connection allows information to persist across timesteps.
    This is crucial for understanding context in sequences like text and speech.
    The backpropagation through time algorithm enables learning these connections.
    Vanilla RNNs suffer from vanishing gradients over long sequences.
    LSTM and GRU architectures use gates to better preserve long-term dependencies.
    Character-level models predict one character at a time using previous context.
    """ * 100

    # Hyperparameters
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    SEQ_LENGTH = 50
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {DEVICE}")

    # Prepare dataset
    dataset = TextDataset(text, seq_length=SEQ_LENGTH)

    # Create models
    model_ff = FeedforwardTextModel(dataset.vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    model_rnn = SimpleRNN(dataset.vocab_size, EMBEDDING_DIM, HIDDEN_DIM)

    print(f"\nFeedforward parameters: {sum(p.numel() for p in model_ff.parameters()):,}")
    print(f"RNN parameters: {sum(p.numel() for p in model_rnn.parameters()):,}")

    # Train both models
    losses_ff = train_model(model_ff, dataset, is_rnn=False, num_epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE, device=DEVICE)
    losses_rnn = train_model(model_rnn, dataset, is_rnn=True, num_epochs=NUM_EPOCHS,
                             batch_size=BATCH_SIZE, device=DEVICE)

    # Compare perplexity
    perp_ff, perp_rnn = compare_perplexity(model_ff, model_rnn, dataset, DEVICE)
    print(f"\n=== Perplexity Comparison (lower is better) ===")
    print(f"Feedforward: {perp_ff:.2f}")
    print(f"RNN: {perp_rnn:.2f}")
    print(f"RNN improvement: {((perp_ff - perp_rnn) / perp_ff * 100):.1f}%")

    # Generate text samples
    print("\n" + "="*80)
    print("=== TEXT GENERATION COMPARISON ===")
    print("="*80)

    seed = "Recurrent "
    gen_length = 250

    print(f"\nSeed text: '{seed}'")
    print(f"Generating {gen_length} characters...\n")

    print("-" * 80)
    print("FEEDFORWARD MODEL (No memory beyond fixed window):")
    print("-" * 80)
    text_ff = generate_text_feedforward(model_ff, dataset, seed, gen_length, 0.8, DEVICE)
    print(text_ff)

    print("\n" + "-" * 80)
    print("RNN MODEL (Full sequence memory via hidden state):")
    print("-" * 80)
    text_rnn = generate_text_rnn(model_rnn, dataset, seed, gen_length, 0.8, DEVICE)
    print(text_rnn)

    print("\n" + "="*80)
    print("=== ANALYSIS ===")
    print("="*80)
    print("""
Key Observations:

1. **Coherence**: The RNN typically produces more coherent long-range text
   because it maintains context through the hidden state.

2. **Consistency**: Feedforward may repeat patterns or lose thread because
   it only sees a fixed window (last 10 chars).

3. **Perplexity**: RNN should have lower perplexity (better predictions)
   because it uses full context history.

4. **Training**: RNN may take longer to train (BPTT is expensive) but
   learns better sequence dependencies.

5. **Generation Speed**: Feedforward could be faster since no hidden state
   to maintain, but RNN generates more natural text.
""")

    # Visualize loss curves
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses_ff, alpha=0.6, label='Raw')
    plt.plot(np.convolve(losses_ff, np.ones(10)/10, mode='valid'), linewidth=2, label='Smoothed')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Feedforward Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(losses_rnn, alpha=0.6, label='Raw')
    plt.plot(np.convolve(losses_rnn, np.ones(10)/10, mode='valid'), linewidth=2, label='Smoothed')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/ashritkuma.samudrala/lnex/ex_llm_rag/main/src/rnn_tutorial/comparison_losses.png')
    plt.close()
    print("\n Loss comparison plot saved to comparison_losses.png")


if __name__ == "__main__":
    main()
