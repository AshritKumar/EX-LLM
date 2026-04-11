import regex as re
from typing import List, Dict, Tuple
from collections import Counter

class ProductionBPETokenizer:
    """
    Production-grade BPE tokenizer with efficient implementation.
    Built step-by-step with explanations for each component.
    """

    def __init__(self):
        # GPT-2 style regex pattern for pre-tokenization
        # We'll explain this pattern in detail below
        self.pattern = re.compile(
            r"""'(?:s|t|re|ve|m|ll|d)|"""  # Contractions: 's, 't, 're, 've, 'm, 'll, 'd
            r""" ?\p{L}+|"""                # Optional space + letters (words)
            r""" ?\p{N}+|"""                # Optional space + numbers
            r""" ?[^\s\p{L}\p{N}]+|"""     # Optional space + punctuation/symbols
            r"""\s+(?!\S)|"""               # Whitespace (not followed by non-whitespace)
            r"""\s+"""                      # Remaining whitespace
        )

        # self.pattern = re.compile(
        #     r"""
        #     '(?:s|t|re|ve|m|ll|d) |   # English contractions
        #     \X+(?=\s|$) |              # One word (grapheme-safe)
        #     \p{N}+ |                  # Numbers
        #     [^\s\p{N}]+ |             # Punctuation
        #     \s+
        #     """,
        #     re.VERBOSE
        # )

        # Step 2: Byte-level encoding
        self.byte_encoder = self._bytes_to_unicode()
        # print("BE ", self.byte_encoder)
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Step 3: BPE vocabulary and merges
        self.vocab = {}  # token → token_id mapping
        self.reverse_vocab = {}  # token_id → token mapping (for decode)
        self.merges = {}  # (pair_a, pair_b) → merged_token mapping
        self.merge_ranks = {}  # merged_token → rank (order of merge)

        # PRODUCTION OPTIMIZATION: Cache for BPE results
        # This is THE most important optimization in real tokenizers!
        # Typical cache hit rate: 80-95% (thanks to Zipf's law)
        self.bpe_cache = {}  # token → List[tokens] mapping

        # Special tokens (will be set via add_special_tokens method)
        self.special_tokens = {}  # special_token_str → token_id mapping
        self.special_token_pattern = None  # Regex pattern for splitting on special tokens

    def pretokenize(self, text: str) -> List[str]:
        # """
        # Step 1: Pre-tokenization - Split text into chunks BEFORE applying BPE.

        # WHY DO WE NEED THIS?
        # ====================
        # Without pre-tokenization, BPE would merge characters across linguistic
        # boundaries, creating weird tokens like:

        # Bad (no pre-tokenization):
        #     "cat dog" → ["cat", " d", "og"]  # space merged with 'd'!
        #     "Hello!" → ["Hell", "o!"]         # word merged with punctuation!

        # Good (with pre-tokenization):
        #     "cat dog" → ["cat", " dog"]       # clean word boundaries
        #     "Hello!" → ["Hello", "!"]         # word and punctuation separate

        # WHAT DOES THE REGEX PATTERN DO?
        # ================================
        # Let's break down each part:

        # 1. r"'(?:s|t|re|ve|m|ll|d)"
        #    Matches: 's, 't, 're, 've, 'm, 'll, 'd
        #    Example: "don't" → ["don", "'t"]
        #    Why? Keeps common contractions as separate tokens

        # 2. r" ?\p{L}+"
        #    Matches: Optional space + one or more letters
        #    Example: " hello" → [" hello"], "hello" → ["hello"]
        #    Why? \p{L} matches ANY Unicode letter (English, Chinese, Arabic, etc.)
        #         The optional space keeps words with leading space together

        # 3. r" ?\p{N}+"
        #    Matches: Optional space + one or more numbers
        #    Example: " 123" → [" 123"], "456" → ["456"]
        #    Why? Treats numbers as separate units

        # 4. r" ?[^\s\p{L}\p{N}]+"
        #    Matches: Optional space + one or more non-letter/non-number/non-space chars
        #    Example: " !!!" → [" !!!"], "..." → ["..."]
        #    Why? Groups punctuation and symbols together

        # 5. r"\s+(?!\S)"
        #    Matches: Whitespace not followed by non-whitespace
        #    Example: "end   " → ["end", "   "] (trailing spaces)
        #    Why? Handles trailing whitespace properly

        # 6. r"\s+"
        #    Matches: Any remaining whitespace
        #    Why? Catch-all for any whitespace we missed

        # REAL-WORLD EXAMPLES:
        # ====================
        # """
        chunks = re.findall(self.pattern, text)
        return chunks

    def _bytes_to_unicode(self):
        """
        Step 2: Byte-level Encoding - Map raw bytes to Unicode characters.

        WHY DO WE NEED BYTE-LEVEL ENCODING?
        ====================================
        Problem with character-level tokenization:
        - Unicode has 149,186 characters (as of Unicode 15.0)
        - Impossible to have all characters in vocabulary
        - Unknown characters would be <UNK> tokens (information loss!)

        Solution: Byte-level encoding
        - ANY text can be represented as UTF-8 bytes (0-255)
        - Only 256 possible byte values = manageable base vocabulary
        - NEVER have unknown tokens - can represent ANY text!

        THE PROBLEM WITH RAW BYTES:
        ===========================
        Some bytes are control characters (0-31) or whitespace (32):
        - Byte 0: NULL character
        - Byte 10: Newline \n
        - Byte 32: Space character

        These cause issues:
        - Control characters can't be printed/displayed
        - Whitespace gets stripped or causes problems in BPE
        - Makes debugging and visualization difficult

        THE SOLUTION: Remap bytes to safe Unicode characters
        =====================================================
        We create a 1-to-1 mapping: 256 bytes → 256 Unicode characters

        Strategy (GPT-2 approach):
        1. Use printable ASCII (33-126, 161-172, 174-255) = 189 characters
           These are "safe" - visible and don't cause issues

        2. For the remaining 67 bytes (control chars + some others),
           map them to unused Unicode points (256-322)

        This gives us:
        - All 256 bytes covered
        - All mapped characters are safe/printable
        - Bijective mapping (can reverse perfectly)

        VISUAL EXAMPLE:
        ===============
        Byte 65 (letter 'A') → Unicode 'A' (safe, printable)
        Byte 10 (newline)    → Unicode 'Ċ' (safe, printable replacement)
        Byte 0  (NULL)       → Unicode 'Ā' (safe, printable replacement)
        """
        # Start with printable ASCII range (excluding control characters and space)
        # 33-126 are printable: !"#$%&'()*+,-./0-9:;<=>?@A-Z[\]^_`a-z{|}~
        bs = list(range(ord("!"), ord("~") + 1))

        # Add Latin-1 supplement printable characters
        # 161-172: ¡¢£¤¥¦§¨©ª«¬
        bs.extend(list(range(ord("¡"), ord("¬") + 1)))

        # 174-255: ®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ
        bs.extend(list(range(ord("®"), ord("ÿ") + 1)))

        # cs will be the Unicode characters we map to
        cs = bs.copy()

        # Now add mappings for the "problematic" bytes (control chars, etc.)
        # These get mapped to unused Unicode codepoints starting at 256
        n = 0
        for b in range(2**8):  # All 256 possible bytes
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)  # Map to 256, 257, 258, ...
                n += 1

        # Convert codepoints to actual characters
        cs = [chr(n) for n in cs]

        # Return dict: byte → Unicode character
        return dict(zip(bs, cs))

    def text_to_bytes(self, text: str) -> str:
        """
        Convert text to bytes, then to our safe Unicode representation.

        Process:
        1. Text → UTF-8 bytes (raw bytes 0-255)
        2. Each byte → mapped Unicode character (using byte_encoder)

        Example:
            "Hello" → b'Hello' → "Hello" (bytes 72,101,108,108,111 are printable)
            "你" → b'\xe4\xbd\xa0' → "ä½Ł" (bytes 228,189,160 mapped to safe chars)

        WHY THIS WORKS:
        - Input: Any Unicode text
        - Output: String of exactly 256 unique safe characters
        - BPE operates on these safe characters
        - Can reverse perfectly (lossless)
        """
        # Convert text to UTF-8 bytes, then map each byte to safe Unicode
        return ''.join(self.byte_encoder[b] for b in text.encode('utf-8'))

    def bytes_to_text(self, byte_string: str) -> str:
        """
        Reverse the byte-level encoding.

        Process:
        1. Safe Unicode characters → raw bytes (using byte_decoder)
        2. Raw bytes → UTF-8 text

        This perfectly reverses text_to_bytes()
        """
        return bytes([self.byte_decoder[c] for c in byte_string]).decode('utf-8', errors='replace')

    def _get_pair_counts(self, words: Dict[str, int], pair_counter: Counter = None) -> Counter:
        """
        Count all adjacent pairs in the corpus efficiently.

        MEMORY OPTIMIZATION:
        ====================
        Instead of creating a new Counter object on each call (expensive!),
        we reuse an existing Counter by clearing it first.

        For 50,000 merges (GPT-2 scale), this eliminates 50,000 allocations.

        WHY HASH MAP (Counter)?
        =======================
        Your simple tokenizer scanned the entire text repeatedly for each merge.
        Time complexity: O(n * m) where n = text length, m = num merges

        Better approach: Use a hash map to count pairs
        - Single pass through the data
        - O(1) lookup and increment for each pair
        - Time complexity: O(n) per iteration

        INPUT:
        ------
        words: Dict[str, int] - Each unique word and its frequency
        Example: {'Ġhello': 100, 'Ġworld': 50}

        pair_counter: Counter - Optional Counter to reuse (for memory efficiency)

        Why dict instead of list?
        - If "hello" appears 1000 times in text, we only process it once
        - We weight its pairs by frequency (multiply by count)
        - MUCH more efficient than processing "hello" 1000 separate times

        OUTPUT:
        -------
        Counter of pairs with their frequencies
        Example: {('Ġ', 'h'): 100, ('h', 'e'): 100, ...}
        """
        if pair_counter is None:
            pairs = Counter()
        else:
            pair_counter.clear()  # Reuse existing Counter (O(1) operation)
            pairs = pair_counter

        for word_tuple, freq in words.items():
            # word_tuple is already a tuple of tokens
            # e.g., ('Ġ', 't', 'i', 'm', 'e') or after merges: ('Ġt', 'i', 'm', 'e')

            # Count all adjacent pairs in this word
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pairs[pair] += freq  # Weight by word frequency!
        return pairs

    def _merge_pair(self, pair: Tuple[str, str], words: Dict[Tuple, int],
                    new_words: Dict[Tuple, int] = None) -> Dict[Tuple, int]:
        """
        Merge a specific pair throughout the corpus.

        MEMORY OPTIMIZATION (Double-Buffering):
        ========================================
        Instead of creating a new dictionary on each call, we can reuse
        an existing dictionary by passing it as new_words parameter.

        This "double-buffering" pattern:
        - Reuses 2 dictionaries throughout training
        - Eliminates 50,000 allocations for GPT-2 scale
        - Significantly reduces memory pressure

        WHY WE USE TUPLES INSTEAD OF STRINGS?
        ======================================
        Problem with strings:
        - 'Ġt' (merged token) looks identical to 'Ġ'+'t' (two separate chars)
        - After merging, we can't distinguish merged vs unmerged pairs!

        Solution: Use tuples of tokens
        - ('Ġ', 't', 'i', 'm', 'e') → clearly 5 tokens
        - After merge: ('Ġt', 'i', 'm', 'e') → clearly 4 tokens
        - Merged token 'Ġt' is a single tuple element!

        EXAMPLE:
        --------
        pair = ('h', 'e')
        words = {('h', 'e', 'l', 'l', 'o'): 10}

        After merge:
        words = {('he', 'l', 'l', 'o'): 10}  # 'he' is now ONE token!
        """
        if new_words is None:
            new_words = {}
        else:
            new_words.clear()  # Reuse existing dict

        for word_tuple, freq in words.items():
            # word_tuple is a tuple of tokens, e.g., ('Ġ', 't', 'i', 'm', 'e')
            new_word = []
            i = 0
            while i < len(word_tuple):
                # Check if we can merge at this position
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == pair:
                    # Merge the pair into a single token
                    new_word.append(word_tuple[i] + word_tuple[i + 1])
                    i += 2  # Skip both elements of the pair
                else:
                    # Keep the token as is
                    new_word.append(word_tuple[i])
                    i += 1
            # print(f"Working with word. {word_tuple} and pair {pair}. => New word {new_word}")
            new_words[tuple(new_word)] = freq
        return new_words

    def train(self, training_text: str, vocab_size: int = 1000):
        """
        Step 3: Train BPE - Learn which pairs to merge through frequency analysis.

        MEMORY OPTIMIZATIONS (PRODUCTION-GRADE):
        =========================================

        This implementation uses THREE key optimizations to minimize memory pressure:

        1. REUSABLE COUNTER:
           - Instead of creating 50,000 Counter objects (one per merge)
           - Reuse a single Counter, clearing it incrementally
           - Eliminates 50,000 allocations for GPT-2 scale training

        2. DOUBLE-BUFFERING:
           - Swap between two word dictionaries instead of creating new ones
           - Reuses 2 dicts throughout training vs creating 50,000
           - Significantly reduces memory allocation overhead

        3. INCREMENTAL PAIR COUNTING:
           - Don't recount ALL pairs on every iteration!
           - Only update pairs affected by the merge
           - Time complexity: O(words_affected) vs O(all_tokens)
           - 10-100x faster for large corpora

        WHY THIS TRAINING APPROACH?
        ============================

        EFFICIENCY IMPROVEMENTS OVER SIMPLE TOKENIZER:

        1. WORD-LEVEL PROCESSING (not character-level):
           Simple: Stores entire text as list of chars [O(n) memory]
           Production: Stores unique words with frequencies [O(unique_words)]

           Example:
           Text: "hello hello hello world"
           Simple: ['h','e','l','l','o',' ','h','e','l','l','o',...] (23 items)
           Production: {'hello': 3, 'world': 1} (2 items)

           Why? If "hello" appears 1M times, we process it once, not 1M times!

        2. HASH MAP FOR PAIR COUNTING:
           Simple: Scans entire list for every pair [O(n)]
           Production: Counter tracks all pairs [O(1) per pair]

        3. EFFICIENT MERGING:
           Simple: List manipulation, shifting elements [O(n)]
           Production: String replacement on unique words [O(unique_words)]

        THE TRAINING ALGORITHM:
        =======================

        1. Pre-tokenize text into chunks (words)
        2. Byte-encode each chunk
        3. Create word frequency dictionary
        4. Build initial pair counts
        5. Repeat vocab_size - 256 times:
           a. Find most frequent pair
           b. Merge that pair everywhere
           c. Incrementally update affected pair counts
           d. Record the merge rule

        WHAT WE'RE LEARNING:
        ====================
        We're learning which byte sequences commonly appear together.

        Example merges:
        Merge 1: ('Ġ', 't') → 'Ġt'      (space + 't', common word start)
        Merge 2: ('h', 'e') → 'he'       (common bigram)
        Merge 3: ('Ġt', 'he') → 'Ġthe'  (merging previous merges!)
        ...
        Merge 500: Complex multi-byte tokens representing common words/subwords
        """
        print(f"\n{'='*70}")
        print(f"STEP 3: BPE TRAINING")
        print(f"{'='*70}\n")
        print(f"Target vocabulary size: {vocab_size}")
        print(f"Base vocabulary (bytes): 256")
        print(f"Merges to learn: {vocab_size - 256}\n")

        # Step 1: Pre-tokenize and byte-encode the training text
        print("Preparing training data...")
        chunks = self.pretokenize(training_text)

        # Step 2: Convert chunks to byte-level representation and count frequencies
        # Using a dictionary to store unique words with their frequencies
        # IMPORTANT: Store as TUPLES of characters, not strings!
        # This allows us to track which characters have been merged into tokens
        words = Counter()
        for chunk in chunks:
            byte_chunk = self.text_to_bytes(chunk)
            # Convert string to tuple of characters
            word_tuple = tuple(byte_chunk)
            words[word_tuple] += 1
        print("Words, ", [f"{self.bytes_to_text(w)}: {f}" for w,f in words.items()])

        print(f"Total chunks: {len(chunks)}")
        print(f"Unique words: {len(words)}")
        # Show example words (convert tuples back to strings for readability)
        example_words = [self.bytes_to_text(''.join(word_tuple)) for word_tuple in list(words.keys())[:5]]
        print(f"Example words: {example_words}\n")

        # Step 3: Initialize vocabulary with base byte tokens (256 characters)
        # IMPORTANT: Use the actual characters from byte_encoder, not chr(i)!
        # byte_encoder maps: byte → safe Unicode char
        # vocab needs: safe Unicode char → token ID
        self.vocab = {char: idx for idx, char in enumerate(self.byte_encoder.values())}
        vocab_index = len(self.vocab)

        # Step 4: Iteratively merge the most frequent pairs
        num_merges = vocab_size - 256
        print(f"Starting BPE training ({num_merges} merges)...")
        print(f"Using memory-optimized incremental pair counting...\n")

        # MEMORY OPTIMIZATION 1: Reusable Counter for pair counting
        # Instead of creating 50,000 Counter objects, reuse one
        pair_counter = Counter()

        # MEMORY OPTIMIZATION 2: Double-buffering for word dictionaries
        # Swap between two dicts instead of creating new ones each iteration
        words_a = words  # Current words
        words_b = {}     # Buffer for next iteration

        # MEMORY OPTIMIZATION 3: Incremental pair counting
        # Build initial pair counts once
        for word_tuple, freq in words_a.items():
            for j in range(len(word_tuple) - 1):
                pair = (word_tuple[j], word_tuple[j + 1])
                pair_counter[pair] += freq

        for i in range(num_merges):
            if not pair_counter:
                print(f"No more pairs to merge. Stopping at {i} merges.")
                break

            # Find most frequent pair
            best_pair = pair_counter.most_common(1)[0][0]
            best_freq = pair_counter[best_pair]

            # Merge the pair with incremental pair updates
            words_b.clear()

            for word_tuple, freq in words_a.items():
                # Check if this word contains the best pair
                has_pair = any(
                    (word_tuple[j], word_tuple[j + 1]) == best_pair
                    for j in range(len(word_tuple) - 1)
                )

                if has_pair:
                    # Remove old pairs from this word
                    for j in range(len(word_tuple) - 1):
                        pair = (word_tuple[j], word_tuple[j + 1])
                        pair_counter[pair] -= freq
                        if pair_counter[pair] <= 0:
                            del pair_counter[pair]

                    # Merge the word
                    new_word = []
                    j = 0
                    while j < len(word_tuple):
                        if j < len(word_tuple) - 1 and (word_tuple[j], word_tuple[j + 1]) == best_pair:
                            new_word.append(word_tuple[j] + word_tuple[j + 1])
                            j += 2
                        else:
                            new_word.append(word_tuple[j])
                            j += 1

                    new_word_tuple = tuple(new_word)
                    words_b[new_word_tuple] = freq

                    # Add new pairs from merged word
                    for j in range(len(new_word_tuple) - 1):
                        pair = (new_word_tuple[j], new_word_tuple[j + 1])
                        pair_counter[pair] += freq
                else:
                    # Word unchanged, just copy
                    words_b[word_tuple] = freq

            # Swap buffers: words_b becomes source for next iteration
            words_a, words_b = words_b, words_a

            # Record this merge
            merged_token = ''.join(best_pair)
            self.merges[best_pair] = merged_token
            self.merge_ranks[merged_token] = i
            self.vocab[merged_token] = vocab_index
            vocab_index += 1

            # Print progress
            if i < 10 or (i + 1) % 100 == 0 or i == num_merges - 1:
                print(f"Merge {i+1:4d}: {best_pair} → '{merged_token}' (frequency: {best_freq})")

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Total merges learned: {len(self.merges)}\n")

        # OPTIMIZATION: Build reverse vocabulary once after training
        # Instead of rebuilding it on every decode() call (expensive!)
        # This is O(V) once vs O(V) per decode call
        print("Building reverse vocabulary for efficient decoding...")
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        print(f"Reverse vocabulary built: {len(self.reverse_vocab)} entries\n")

        print("WHAT DID WE LEARN?")
        print("-" * 70)
        print("The tokenizer now knows:")
        print("1. 256 base byte tokens (handles ANY character)")
        print("2. Frequent byte combinations as merged tokens")
        print("3. The ORDER of merges (important for encoding!)")
        print(f"{'='*70}\n")

    def add_special_tokens(self, special_tokens: List[str]):
        """
        Add special tokens to the vocabulary.

        WHY SPECIAL TOKENS?
        ===================
        Special tokens are markers that have specific meanings and should NEVER
        be split by BPE. Common examples:

        - <|endoftext|>  : Marks document boundaries (GPT-2/3)
        - <|startoftext|>: Marks document start
        - <PAD>          : Padding token for batching
        - <UNK>          : Unknown token (rare in BPE, but useful)
        - <BOS>, <EOS>   : Beginning/End of sequence
        - <MASK>         : For masked language modeling

        HOW THEY WORK:
        ==============
        1. Special tokens get unique IDs in vocabulary (usually at the end)
        2. During encoding:
           - First split text on special tokens
           - Encode each part normally with BPE
           - Special tokens themselves are NOT processed by BPE
        3. During decoding:
           - Special tokens map directly back to their strings

        EXAMPLE:
        ========
        Text: "Hello<|endoftext|>World"

        Without special tokens:
        → BPE would try to merge 'o', '<', '|', etc. → wrong!

        With special tokens:
        → Split: ["Hello", "<|endoftext|>", "World"]
        → Encode: [tokens for "Hello"] + [special_token_id] + [tokens for "World"]
        → Decode: "Hello" + "<|endoftext|>" + "World" ✓

        IMPORTANT:
        ==========
        - Call this AFTER training (so vocab is already built)
        - Special tokens are added to the END of vocabulary
        - Special tokens never go through BPE processing
        """
        print(f"\n{'='*70}")
        print("ADDING SPECIAL TOKENS")
        print(f"{'='*70}\n")

        # Start assigning IDs after the current vocabulary
        current_vocab_size = len(self.vocab)

        for i, special_token in enumerate(special_tokens):
            token_id = current_vocab_size + i

            # Add to vocabulary
            self.vocab[special_token] = token_id
            self.reverse_vocab[token_id] = special_token
            self.special_tokens[special_token] = token_id

            print(f"Added: '{special_token}' → ID {token_id}")

        # Build regex pattern to split on special tokens
        # Need to escape special regex characters and match any special token
        import re as standard_re
        escaped_tokens = [standard_re.escape(token) for token in special_tokens]
        # Pattern matches any special token
        self.special_token_pattern = standard_re.compile(
            '(' + '|'.join(escaped_tokens) + ')'
        )

        print(f"\nTotal vocabulary size: {len(self.vocab)}")
        print(f"Special tokens: {len(self.special_tokens)}")
        print(f"{'='*70}\n")

    def _split_on_special_tokens(self, text: str) -> List[tuple]:
        """
        Split text on special tokens, returning (chunk, is_special) tuples.

        WHY THIS APPROACH?
        ==================
        We need to know which parts are special tokens (don't process with BPE)
        and which parts are regular text (do process with BPE).

        EXAMPLE:
        ========
        text = "Hello<|endoftext|>World<PAD>"

        Returns:
        [
            ("Hello", False),           # Regular text - process with BPE
            ("<|endoftext|>", True),    # Special token - don't touch!
            ("World", False),           # Regular text - process with BPE
            ("<PAD>", True)             # Special token - don't touch!
        ]
        """
        if not self.special_token_pattern:
            # No special tokens registered
            return [(text, False)]

        # Split on special tokens, keeping the delimiters
        parts = self.special_token_pattern.split(text)

        result = []
        for part in parts:
            if not part:  # Skip empty strings
                continue

            # Check if this part is a special token
            is_special = part in self.special_tokens
            result.append((part, is_special))

        return result

    def _apply_bpe(self, token: str) -> List[str]:
        """
        Step 4a: Apply BPE merges to a single token (word chunk).

        WHY DO WE NEED THIS?
        ====================
        During training, we learned which pairs to merge and in what order.
        Now we need to apply those SAME merges in the SAME order to encode new text.

        THE CHALLENGE:
        ==============
        We can't just apply all merges at once! The ORDER matters:

        Example with word "test":
        - Merge 1: ('t', 'e') → 'te'     → ['te', 's', 't']
        - Merge 2: ('s', 't') → 'st'     → ['te', 'st']
        - Merge 3: ('te', 'st') → 'test' → ['test']

        If we applied merge 3 first, we'd never find the pair ('te', 'st')
        because 'te' and 'st' don't exist yet!

        THE ALGORITHM:
        ==============
        1. Start with individual characters: ['t', 'e', 's', 't']
        2. Repeatedly find the EARLIEST learned pair (lowest rank)
        3. Merge that pair
        4. Repeat until no more learned pairs exist

        This is called "greedy BPE" - always merge the most prioritized pair.

        WHY USE MERGE RANKS?
        ====================
        merge_ranks tells us the order in which pairs were learned:
        - Lower rank = learned earlier = more common = higher priority
        - ('t', 'e'): rank 0 (learned first)
        - ('s', 't'): rank 1 (learned second)
        - ('te', 'st'): rank 2 (learned third)

        We always merge the pair with the LOWEST rank first.

        OPTIMIZATIONS (PRODUCTION-GRADE):
        ==================================
        1. CACHE LOOKUP (Most Important!)
           - Check if we've already processed this token
           - Cache hit rate: 80-95% in real text (Zipf's law)
           - Example: "the" appears 1000x → computed once, cached forever

        2. VOCABULARY FAST-PATH
           - Check if entire token is already in vocabulary
           - Common for frequent complete words from training

        3. SINGLE-PASS PAIR FINDING
           - Find best pair in one iteration, not two
           - Early exit when rank-0 pair found
        """
        # OPTIMIZATION 1: Cache lookup (THE most important!)
        # This is what makes production tokenizers fast!
        if token in self.bpe_cache:
            return self.bpe_cache[token]

        # Edge case: single character tokens can't be merged
        if len(token) == 1:
            return [token]

        # OPTIMIZATION 2: Fast-path for tokens already in vocabulary
        if token in self.vocab:
            self.bpe_cache[token] = [token]  # Cache the result
            return [token]

        # Start with individual characters as separate tokens
        word = list(token)

        while True:
            # OPTIMIZATION: Find best pair in a single pass
            # Instead of creating list of all pairs then finding min,
            # we iterate once and track the best pair we've seen

            best_pair = None
            best_rank = float('inf')
            best_idx = -1

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                merged_token = pair[0] + pair[1]

                # Look up the rank of this merge
                rank = self.merge_ranks.get(merged_token, float('inf'))

                # Track the best (lowest rank) pair
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_idx = i

                    # EARLY EXIT: If we found rank 0, it's the highest priority
                    # No need to check remaining pairs!
                    if rank == 0:
                        break

            # If no valid pair found (all pairs have infinite rank), we're done
            if best_rank == float('inf'):
                break

            bigram = best_pair

            # Merge this pair throughout the word
            new_word = []
            i = 0
            while i < len(word):
                # Check if we can merge at this position
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    # Merge the pair
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    # Keep the token as is
                    new_word.append(word[i])
                    i += 1

            word = new_word

            # If we can't reduce the word anymore (single token), stop
            if len(word) == 1:
                break

        # Cache the result before returning (for future lookups)
        self.bpe_cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        """
        Step 4b: Encode text into token IDs.

        THE FULL PIPELINE (WITH SPECIAL TOKENS):
        =========================================
        Input: "Hello<|endoftext|>World"
                ↓
        0. Split on special tokens (if any)
                ↓ [("Hello", False), ("<|endoftext|>", True), ("World", False)]
        1. For regular text: Pre-tokenize (split into chunks)
                ↓ ['Hello'], [special], ['World']
        2. For regular text: Byte-encode each chunk
                ↓ ['Hello'], [special], ['World']
        3. For regular text: Apply BPE merges
                ↓ ['H', 'ello'], [special], ['W', 'orld']
        4. For special tokens: Use directly (no BPE!)
                ↓ ['H', 'ello'], ['<|endoftext|>'], ['W', 'orld']
        5. Look up token IDs in vocabulary
                ↓ [1234, 5678, 50256, 9012, 3456]

        WHY THIS WORKS:
        ===============
        - Special tokens are split first and never processed by BPE
        - Pre-tokenization ensures we don't merge across word boundaries
        - Byte-encoding handles ANY character (emojis, unicode, etc.)
        - BPE merges compress common patterns into single tokens
        - Vocabulary lookup converts tokens to IDs for the model

        SPECIAL TOKEN HANDLING:
        =======================
        Special tokens bypass all BPE processing:
        - They are NOT byte-encoded
        - They are NOT pre-tokenized
        - They are NOT merged by BPE
        - They map directly to their vocabulary IDs
        """
        # Handle empty string
        if not text:
            return []

        # Step 0: Split on special tokens (if any registered)
        parts = self._split_on_special_tokens(text)

        all_tokens = []
        for part, is_special in parts:
            if is_special:
                # Special token - add directly without any processing
                all_tokens.append(part)
            else:
                # Regular text - apply full BPE pipeline
                # Step 1: Pre-tokenize the text
                chunks = self.pretokenize(part)

                # Step 2 & 3: Byte-encode and apply BPE to each chunk
                for chunk in chunks:
                    # Convert to byte representation
                    byte_chunk = self.text_to_bytes(chunk)

                    # Apply BPE merges
                    bpe_tokens = self._apply_bpe(byte_chunk)

                    all_tokens.extend(bpe_tokens)

        # Step 4: Convert tokens to IDs
        token_ids = [self.vocab.get(token, -1) for token in all_tokens]

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Step 5: Decode token IDs back to text.

        THE REVERSE PIPELINE:
        =====================
        Input: [1234, 5678, 9012]
                ↓
        1. Look up tokens from IDs
                ↓ ['The', 'Ġqu', 'ick']
        2. Concatenate all tokens
                ↓ 'TheĠquick'
        3. Decode from byte representation
                ↓ 'The quick'

        WHY THIS WORKS:
        ===============
        - Vocabulary lookup is just a reverse dictionary lookup
        - Concatenating tokens reverses the BPE merging
        - Byte decoding reverses the byte encoding

        OPTIMIZATION:
        =============
        We use pre-computed reverse_vocab (built once after training)
        instead of creating it on every decode call.

        Before: O(V) per decode (V = vocab size, e.g., 50K)
        After:  O(1) per decode ✓

        HANDLING UNKNOWN TOKEN IDS:
        ===========================
        If we encounter an unknown token ID, we skip it (return empty string).
        This shouldn't happen in normal usage, but provides safety.
        """
        # Step 1: Convert IDs back to tokens using pre-computed reverse vocab
        # OPTIMIZATION: self.reverse_vocab is built once, reused forever
        tokens = [self.reverse_vocab.get(token_id, '') for token_id in token_ids]

        # Step 2 & 3: Handle special tokens vs regular tokens
        # Special tokens are already strings, don't byte-decode them
        # Regular tokens need byte-decoding

        result_parts = []
        current_byte_string = []

        for token in tokens:
            if token in self.special_tokens:
                # Found a special token
                # First, decode any accumulated byte tokens
                if current_byte_string:
                    byte_str = ''.join(current_byte_string)
                    result_parts.append(self.bytes_to_text(byte_str))
                    current_byte_string = []

                # Add the special token as-is
                result_parts.append(token)
            else:
                # Regular token - accumulate for byte decoding
                current_byte_string.append(token)

        # Don't forget to decode any remaining byte tokens
        if current_byte_string:
            byte_str = ''.join(current_byte_string)
            result_parts.append(self.bytes_to_text(byte_str))

        return ''.join(result_parts)


# Let's test and demonstrate the tokenizer
if __name__ == "__main__":
    tokenizer = ProductionBPETokenizer()

    # print("=" * 70)
    # print("STEP 1: PRE-TOKENIZATION DEMONSTRATION")
    # print("=" * 70)
    # print()

    # # Example 1: Simple sentence
    # text1 = "Hello, world!"
    # chunks1 = tokenizer.pretokenize(text1)
    # print(f"Text: '{text1}'")
    # print(f"Chunks: {chunks1}")
    # print(f"Why? Word, punctuation, and word with space are kept separate")
    # print()

    # # Example 2: Contractions
    # text2 = "I don't think it's working"
    # chunks2 = tokenizer.pretokenize(text2)
    # print(f"Text: '{text2}'")
    # print(f"Chunks: {chunks2}")
    # print(f"Why? Contractions like 'don't' split into ['don', \"'t\"]")
    # print()

    # print("=" * 70)
    # print("STEP 2: BYTE-LEVEL ENCODING DEMONSTRATION")
    # print("=" * 70)
    # print()

    # # Example 1: Simple ASCII text
    # text_ascii = "Hello"
    # byte_repr = tokenizer.text_to_bytes(text_ascii)
    # decoded = tokenizer.bytes_to_text(byte_repr)
    # print(f"1. ASCII Text: '{text_ascii}'")
    # print(f"   Raw bytes: {list(text_ascii.encode('utf-8'))}")
    # print(f"   Byte-encoded: '{byte_repr}'")
    # print(f"   Decoded back: '{decoded}'")
    # print(f"   Why? ASCII chars map to themselves (they're already safe)")
    # print()

    # # Example 2: Unicode (Chinese)
    # text_chinese = "你好"
    # byte_repr_chinese = tokenizer.text_to_bytes(text_chinese)
    # decoded_chinese = tokenizer.bytes_to_text(byte_repr_chinese)
    # raw_bytes = list(text_chinese.encode('utf-8'))
    # print(f"2. Chinese Text: '{text_chinese}'")
    # print(f"   Raw UTF-8 bytes: {raw_bytes} ({len(raw_bytes)} bytes)")
    # print(f"   Byte-encoded: '{byte_repr_chinese}'")
    # print(f"   Decoded back: '{decoded_chinese}'")
    # print(f"   Why? 1 Chinese char = 3 UTF-8 bytes, each mapped to safe char")
    # print()

    # # Example 3: Emoji
    # text_emoji = "Hello 😀"
    # byte_repr_emoji = tokenizer.text_to_bytes(text_emoji)
    # decoded_emoji = tokenizer.bytes_to_text(byte_repr_emoji)
    # raw_bytes_emoji = list(text_emoji.encode('utf-8'))
    # print(f"3. Text with Emoji: '{text_emoji}'")
    # print(f"   Raw UTF-8 bytes: {raw_bytes_emoji[:5]}...{raw_bytes_emoji[-4:]} ({len(raw_bytes_emoji)} bytes)")
    # print(f"   Byte-encoded: '{byte_repr_emoji}'")
    # print(f"   Decoded back: '{decoded_emoji}'")
    # print(f"   Why? Emoji (4 bytes) becomes 4 safe characters")
    # print()

    # # Example 4: Special characters (newline, tab)
    # text_special = "Hello\nWorld\t!"
    # byte_repr_special = tokenizer.text_to_bytes(text_special)
    # decoded_special = tokenizer.bytes_to_text(byte_repr_special)
    # print(f"4. Text with special chars: 'Hello\\nWorld\\t!'")
    # print(f"   Byte-encoded: '{byte_repr_special}'")
    # print(f"   Decoded back: '{decoded_special}'")
    # print(f"   Why? \\n and \\t (control chars) mapped to visible characters")
    # print()

    # # Example 5: Combined - Pre-tokenize then byte-encode
    # text_combined = "你好world!"
    # chunks = tokenizer.pretokenize(text_combined)
    # print(f"5. Combined Pipeline: '{text_combined}'")
    # print(f"   Pre-tokenized chunks: {chunks}")
    # for chunk in chunks:
    #     byte_chunk = tokenizer.text_to_bytes(chunk)
    #     print(f"   '{chunk}' → '{byte_chunk}' (length: {len(chunk)} chars → {len(byte_chunk)} bytes)")
    # print()

    # print("=" * 70)
    # print("KEY BENEFITS OF BYTE-LEVEL ENCODING:")
    # print("=" * 70)
    # print("1. ✓ Handles ANY text - emojis, rare languages, special chars")
    # print("2. ✓ No unknown tokens - everything has a representation")
    # print("3. ✓ Fixed base vocabulary size - exactly 256 byte tokens")
    # print("4. ✓ Lossless - perfect reversibility (encode → decode)")
    # print("5. ✓ Safe for BPE - all bytes mapped to printable characters")
    # print()

    # # Now let's train the tokenizer!
    # print("\n" + "=" * 70)
    # print("Now let's train the tokenizer with a sample text...")
    # print("=" * 70)

    # Sample training text - something with repeated patterns
    # training_text = """
    # The quick brown fox jumps over the lazy dog.
    # The dog was really lazy, and the fox was very quick.
    # Quick foxes and lazy dogs are common in these stories.
    # The story of the fox and the dog is a classic tale.
    # """

    # training_text = "the time traveller (for so it will be convenient to speak of him) was expounding a recondite matter to us. his pale grey eyes shone and twinkled, and his usually pale face was flushed and animated. the fire burnt brightly"
    # training_text = "The quick brown fox jumps over the lazy dog."

    training_text = "తెలుగు భాగవతం"

    # Train with a small vocabulary for demonstration
    tokenizer.train(training_text, vocab_size=300)
    print(tokenizer.merge_ranks)
    c = tokenizer.encode("భాగ")
    print(c)
    print(tokenizer.decode([256, 255, 265, 260]))
    for mt in tokenizer.merge_ranks:
        print(mt , " => ", tokenizer.bytes_to_text(mt))

    # # Show some learned merges
    # print("\nEXAMPLE OF LEARNED MERGES:")
    # print("-" * 70)
    # print("First 10 merges learned:")
    # for i, (pair, merged) in enumerate(list(tokenizer.merges.items())[:10]):
    #     print(f"  {i+1}. {pair} → '{merged}'")

    # print("\nNotice how the tokenizer learned:")
    # print("- Common character combinations")
    # print("- Frequently occurring patterns in the training text")
    # print("- The merges build on each other (hierarchical)")
    # print("=" * 70)

    # # Now let's test encode and decode!
    # print("\n" + "=" * 70)
    # print("STEP 4 & 5: ENCODE AND DECODE DEMONSTRATION")
    # print("=" * 70)
    # print()

    # # Test 1: Encode text from training corpus
    # test_text1 = "the fire burnt"
    # print(f"Test 1: Text from training corpus")
    # print(f"Input: '{test_text1}'")

    # # Show the tokenization pipeline
    # chunks = tokenizer.pretokenize(test_text1)
    # print(f"Pre-tokenized: {chunks}")

    # token_ids = tokenizer.encode(test_text1)
    # print(f"Token IDs: {token_ids}")
    # print(f"Number of tokens: {len(token_ids)}")

    # decoded = tokenizer.decode(token_ids)
    # print(f"Decoded: '{decoded}'")
    # print(f"Perfect round-trip: {test_text1 == decoded} ✓")
    # print()

    # # Test 2: Encode new text (not in training)
    # test_text2 = "the fox is quick"
    # print(f"Test 2: New text (mix of known and unknown words)")
    # print(f"Input: '{test_text2}'")

    # token_ids2 = tokenizer.encode(test_text2)
    # print(f"Token IDs: {token_ids2}")
    # print(f"Number of tokens: {len(token_ids2)}")

    # decoded2 = tokenizer.decode(token_ids2)
    # print(f"Decoded: '{decoded2}'")
    # print(f"Perfect round-trip: {test_text2 == decoded2} ✓")
    # print()

    # # Test 3: Show token breakdown
    # print(f"Test 3: Token breakdown for 'the fire'")
    # test_text3 = "the fire"
    # chunks3 = tokenizer.pretokenize(test_text3)

    # for chunk in chunks3:
    #     byte_chunk = tokenizer.text_to_bytes(chunk)
    #     tokens = tokenizer._apply_bpe(byte_chunk)
    #     token_ids_chunk = [tokenizer.vocab.get(t, -1) for t in tokens]
    #     print(f"  '{chunk}' → byte: '{byte_chunk}' → tokens: {tokens} → IDs: {token_ids_chunk}")
    # print()

    # # Test 4: Compression ratio
    # print(f"Test 4: Tokenization efficiency")
    # test_text4 = training_text[:50]  # First 50 chars
    # token_ids4 = tokenizer.encode(test_text4)
    # print(f"Text: '{test_text4}'")
    # print(f"Characters: {len(test_text4)}")
    # print(f"Tokens: {len(token_ids4)}")
    # print(f"Compression ratio: {len(test_text4) / len(token_ids4):.2f}x")
    # print(f"(Each token represents ~{len(test_text4) / len(token_ids4):.1f} characters on average)")
    # print()

    # print("=" * 70)
    # print("KEY INSIGHTS:")
    # print("=" * 70)
    # print("1. ✓ Perfect lossless encoding/decoding (encode → decode = original)")
    # print("2. ✓ Works on both training and new text")
    # print("3. ✓ Common words become single tokens (efficiency!)")
    # print("4. ✓ Rare/unknown words split into subword tokens")
    # print("5. ✓ Compression: ~2-4x fewer tokens than characters")
    # print("=" * 70)
