from production_bpe_tokenizer import ProductionBPETokenizer

print("=" * 70)
print("SPECIAL TOKENS DEMONSTRATION")
print("=" * 70)
print()

# Train tokenizer on sample text
tokenizer = ProductionBPETokenizer()
training_text = """
The quick brown fox jumps over the lazy dog.
The dog was really lazy, and the fox was very quick.
Quick foxes and lazy dogs are common in these stories.
"""

print("Training tokenizer...")
tokenizer.train(training_text, vocab_size=300)

# Add special tokens AFTER training
print("\n" + "=" * 70)
print("STEP 1: Adding Special Tokens")
print("=" * 70)

special_tokens = [
    "<|endoftext|>",
    "<|startoftext|>",
    "<PAD>",
    "<UNK>",
    "<MASK>"
]

tokenizer.add_special_tokens(special_tokens)

# Test encoding with special tokens
print("\n" + "=" * 70)
print("STEP 2: Encoding Text with Special Tokens")
print("=" * 70)
print()

# Test 1: Text with endoftext marker
test1 = "The quick fox<|endoftext|>The lazy dog"
print(f"Test 1: Document boundary marker")
print(f"Input: '{test1}'")
print()

token_ids1 = tokenizer.encode(test1)
print(f"Token IDs: {token_ids1}")
print(f"Number of tokens: {len(token_ids1)}")

decoded1 = tokenizer.decode(token_ids1)
print(f"Decoded: '{decoded1}'")
print(f"Perfect round-trip: {test1 == decoded1} ✓")
print()

# Test 2: Text with start marker
test2 = "<|startoftext|>Once upon a time<|endoftext|>"
print(f"Test 2: Document with start and end markers")
print(f"Input: '{test2}'")
print()

token_ids2 = tokenizer.encode(test2)
print(f"Token IDs: {token_ids2}")
print(f"Number of tokens: {len(token_ids2)}")

decoded2 = tokenizer.decode(token_ids2)
print(f"Decoded: '{decoded2}'")
print(f"Perfect round-trip: {test2 == decoded2} ✓")
print()

# Test 3: Multiple special tokens
test3 = "Hello<PAD><PAD><MASK>World<UNK>"
print(f"Test 3: Multiple special tokens")
print(f"Input: '{test3}'")
print()

token_ids3 = tokenizer.encode(test3)
print(f"Token IDs: {token_ids3}")
print(f"Number of tokens: {len(token_ids3)}")

decoded3 = tokenizer.decode(token_ids3)
print(f"Decoded: '{decoded3}'")
print(f"Perfect round-trip: {test3 == decoded3} ✓")
print()

# Show token breakdown for Test 1
print("=" * 70)
print("STEP 3: Token Breakdown Analysis")
print("=" * 70)
print()

print(f"Analyzing: '{test1}'")
print()

# Manually show what happens
parts = tokenizer._split_on_special_tokens(test1)
print("Step 1: Split on special tokens:")
for i, (part, is_special) in enumerate(parts):
    print(f"  Part {i+1}: '{part}' (special={is_special})")
print()

print("Step 2: Encode each part:")
for part, is_special in parts:
    if is_special:
        special_id = tokenizer.special_tokens[part]
        print(f"  '{part}' → [ID: {special_id}] (special token, no BPE)")
    else:
        # Show BPE processing
        chunks = tokenizer.pretokenize(part)
        tokens = []
        for chunk in chunks:
            byte_chunk = tokenizer.text_to_bytes(chunk)
            bpe_tokens = tokenizer._apply_bpe(byte_chunk)
            tokens.extend(bpe_tokens)
        token_ids = [tokenizer.vocab.get(t, -1) for t in tokens]
        print(f"  '{part}' → {tokens} → IDs: {token_ids[:5]}{'...' if len(token_ids) > 5 else ''}")
print()

# Verify special token IDs
print("=" * 70)
print("STEP 4: Special Token Vocabulary")
print("=" * 70)
print()

print("Special tokens in vocabulary:")
for token, token_id in tokenizer.special_tokens.items():
    print(f"  '{token}' → ID {token_id}")
print()

print(f"Total vocabulary size: {len(tokenizer.vocab)}")
print(f"  - Base byte tokens: 256")
print(f"  - BPE merged tokens: {len(tokenizer.merges)}")
print(f"  - Special tokens: {len(tokenizer.special_tokens)}")
print()

print("=" * 70)
print("KEY INSIGHTS: Special Tokens")
print("=" * 70)
print()
print("1. ✓ Special tokens are NEVER split by BPE")
print("2. ✓ Special tokens are split BEFORE pre-tokenization")
print("3. ✓ Special tokens bypass byte-encoding completely")
print("4. ✓ Special tokens have unique IDs in vocabulary")
print("5. ✓ Perfect round-trip encoding/decoding")
print()
print("COMMON USE CASES:")
print("  • <|endoftext|>  : Document boundaries (GPT-2/3/4)")
print("  • <PAD>          : Padding for batching")
print("  • <UNK>          : Unknown tokens (rare in BPE)")
print("  • <MASK>         : Masked language modeling (BERT)")
print("  • <BOS>, <EOS>   : Sequence boundaries")
print()
print("=" * 70)
