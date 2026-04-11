from sb_tokenizer import SBTokenizer

t = SBTokenizer()
tx1 = "he is a good goon hence here referencing a fencing here"
t.train(tx1, vocab_size=300)

# Add special tokens with NON-ASCII characters
special_tokens = [
    "<START→>",      # Has → (Unicode arrow)
    "<패드>",         # Korean
    "<END🔚>",       # Has emoji
]

t.add_special_tokens(spl_tokens=special_tokens)

print("Special tokens added:")
for token, tid in t.special_tokens.items():
    print(f"  '{token}' → ID {tid}")
print()

# Try to encode text with these special tokens
text = "hello<START→>world"
print(f"Encoding: '{text}'")

try:
    c = t.encode(text)
    print(f"Token IDs: {c}")
    print()
    
    # Try simple decode
    print("Attempting simple decode...")
    decoded = t.decode(c)
    print(f"Decoded: '{decoded}'")
    print(f"Match: {text == decoded}")
    
except KeyError as e:
    print(f"❌ ERROR: {e}")
    print("Simple decode failed because special token has non-ASCII characters!")
    print("byte_decoder doesn't have mappings for these Unicode characters")

