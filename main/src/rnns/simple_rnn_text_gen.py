"""
Simple RNN for Character-Level Text Generation
This implements a vanilla RNN from scratch to understand the recurrent mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class VanillaRNN(nn.Module):
    """
    A simple RNN cell that maintains hidden state across time steps.

    Key difference from feedforward:
    - Hidden state h_t depends on both input x_t and previous hidden state h_{t-1}
    - h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_h)
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Embedding layer: maps token indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN layers: each layer transforms (input + prev_hidden) -> new_hidden
        self.rnn_layers = nn.ModuleList()
        for i in range(num_layers):
            input_size = embedding_dim if i == 0 else hidden_dim
            self.rnn_layers.append(
                nn.RNNCell(input_size, hidden_dim)
            )

        # Output projection: hidden -> logits over vocabulary
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights for better training
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot initialization for better gradient flow"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN.

        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices
            hidden: Optional initial hidden state of shape (num_layers, batch_size, hidden_dim)

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden: Final hidden state of shape (num_layers, batch_size, hidden_dim)
        """
        batch_size, seq_len = x.shape

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # Embed input tokens: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embeds = self.embedding(x)

        # Process sequence step by step (this is the key RNN characteristic)
        outputs = []
        for t in range(seq_len):
            # Get input at time step t: (batch_size, embedding_dim)
            x_t = embeds[:, t, :]

            # Pass through each RNN layer
            new_hidden = []
            for layer_idx, rnn_cell in enumerate(self.rnn_layers):
                # Get previous hidden state for this layer
                h_prev = hidden[layer_idx]

                # RNN cell: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
                h_t = rnn_cell(x_t, h_prev)
                new_hidden.append(h_t)

                # Output of this layer becomes input to next layer
                x_t = h_t

            # Update hidden states
            hidden = torch.stack(new_hidden)

            # Final layer output goes to vocabulary projection
            outputs.append(x_t)

        # Stack outputs: (seq_len, batch_size, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        output_seq = torch.stack(outputs, dim=1)

        # Project to vocabulary: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, vocab_size)
        logits = self.fc_out(output_seq)

        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state with zeros"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)


class TextDataset:
    """Handles text tokenization and batch creation"""

    def __init__(self, text: str, seq_length: int = 50):
        self.seq_length = seq_length

        # Create character-level vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Encode entire text
        self.encoded = np.array([self.char_to_idx[ch] for ch in text])

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Text length: {len(text)} characters")
        print(f"Encoded length: {len(self.encoded)} tokens")

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a random batch of input-target pairs.

        For each sequence, target is input shifted by 1 position.
        Input:  "hello worl"
        Target: "ello world"
        """
        # Randomly sample starting positions
        max_start = len(self.encoded) - self.seq_length - 1
        starts = np.random.randint(0, max_start, size=batch_size)

        # Extract sequences
        input_seqs = np.array([self.encoded[i:i+self.seq_length] for i in starts])
        target_seqs = np.array([self.encoded[i+1:i+self.seq_length+1] for i in starts])

        return torch.LongTensor(input_seqs), torch.LongTensor(target_seqs)


def train_rnn(model: VanillaRNN, dataset: TextDataset,
              num_epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.002, device: str = 'cpu'):
    """
    Train the RNN with backpropagation through time (BPTT).

    BPTT: Unfolds the RNN across time steps and backpropagates through the entire sequence.
    This is computationally expensive but necessary for learning temporal dependencies.
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses = []

    print("\n=== Training RNN ===")
    for epoch in range(num_epochs):
        # Get batch: inputs (batch_size, seq_len), targets (batch_size, seq_len)
        inputs, targets = dataset.get_batch(batch_size)
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass: process entire sequence
        # logits: (batch_size, seq_len, vocab_size)
        # hidden: (num_layers, batch_size, hidden_dim)
        logits, hidden = model(inputs)

        # Reshape for loss calculation
        # logits: (batch_size * seq_len, vocab_size)
        # targets: (batch_size * seq_len,)
        logits_flat = logits.reshape(-1, model.vocab_size)
        targets_flat = targets.reshape(-1)

        # Calculate cross-entropy loss
        loss = criterion(logits_flat, targets_flat)

        # Backward pass (BPTT): gradients flow backward through time
        loss.backward()

        # Gradient clipping to prevent exploding gradients (common issue in RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Update weights
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    return losses


def generate_text(model: VanillaRNN, dataset: TextDataset,
                  start_text: str = "The ", length: int = 200,
                  temperature: float = 0.8, device: str = 'cpu') -> str:
    """
    Generate text using the trained RNN.

    Temperature controls randomness:
    - Low (0.5): More deterministic, picks high-probability tokens
    - High (1.5): More random, explores diverse options
    """
    model = model.to(device)
    model.eval()

    # Encode starting text
    input_seq = [dataset.char_to_idx[ch] for ch in start_text]
    generated = start_text

    # Initialize hidden state
    hidden = model.init_hidden(1, device)

    with torch.no_grad():
        # First, process the starting text to build up hidden state
        for i in range(len(input_seq) - 1):
            x = torch.LongTensor([[input_seq[i]]]).to(device)
            _, hidden = model(x, hidden)

        # Start generating from last character of start_text
        current_char_idx = input_seq[-1]

        # Generate new characters one at a time
        for _ in range(length):
            # Forward pass with single character
            x = torch.LongTensor([[current_char_idx]]).to(device)
            logits, hidden = model(x, hidden)

            # Get logits for the last (and only) time step
            logits = logits[0, -1, :]  # (vocab_size,)

            # Apply temperature scaling
            logits = logits / temperature

            # Convert to probabilities
            probs = F.softmax(logits, dim=0)

            # Sample next character (stochastic sampling for diversity)
            current_char_idx = torch.multinomial(probs, num_samples=1).item()

            # Decode and append
            generated += dataset.idx_to_char[current_char_idx]

    return generated


def visualize_training(losses: list):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, alpha=0.7)
    plt.plot(np.convolve(losses, np.ones(10)/10, mode='valid'), linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (with 10-epoch moving average)')
    plt.grid(True, alpha=0.3)
    plt.savefig('/Users/ashritkuma.samudrala/lnex/ex_llm_rag/main/src/rnn_tutorial/training_loss.png')
    plt.close()
    print("Training loss plot saved to training_loss.png")


def main():
    # Sample text (you can replace with any text corpus)
    sample_text = """
    Deep learning is a subset of machine learning that uses neural networks with multiple layers.
    Recurrent neural networks are particularly good at processing sequential data like text and time series.
    Unlike feedforward networks, RNNs maintain a hidden state that carries information across time steps.
    This hidden state acts as a form of memory, allowing the network to learn temporal dependencies.
    The backpropagation through time algorithm unfolds the RNN and computes gradients across all time steps.
    Gradient clipping is important to prevent exploding gradients, a common problem in RNN training.
    Modern variants like LSTM and GRU address the vanishing gradient problem with gating mechanisms.
    Character-level models learn to predict the next character given previous context.
    Temperature sampling controls the randomness in text generation, balancing creativity and coherence.
    RNNs have been widely used for language modeling, machine translation, and speech recognition.
    """ * 50  # Repeat for more training data

    # Hyperparameters
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    SEQ_LENGTH = 50
    BATCH_SIZE = 64
    NUM_EPOCHS = 150
    LEARNING_RATE = 0.002
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")

    # Prepare dataset
    dataset = TextDataset(sample_text, seq_length=SEQ_LENGTH)

    # Initialize model
    model = VanillaRNN(
        vocab_size=dataset.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    )

    print(f"\nModel architecture:")
    print(f"  Embedding: {dataset.vocab_size} x {EMBEDDING_DIM}")
    print(f"  RNN layers: {NUM_LAYERS} x {HIDDEN_DIM}")
    print(f"  Output: {HIDDEN_DIM} x {dataset.vocab_size}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train the model
    losses = train_rnn(
        model, dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )

    # Visualize training
    visualize_training(losses)

    # Generate text samples with different temperatures
    print("\n=== Generated Text Samples ===\n")

    for temp in [0.5, 0.8, 1.2]:
        print(f"Temperature = {temp}")
        print("-" * 80)
        generated = generate_text(
            model, dataset,
            start_text="Deep learning ",
            length=300,
            temperature=temp,
            device=DEVICE
        )
        print(generated)
        print("\n")

    # Save model
    torch.save(model.state_dict(), '/Users/ashritkuma.samudrala/lnex/ex_llm_rag/main/src/rnn_tutorial/rnn_model.pt')
    print("Model saved to rnn_model.pt")


if __name__ == "__main__":
    main()
