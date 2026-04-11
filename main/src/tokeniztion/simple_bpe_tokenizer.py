import re
class SimpleBPETokenizer:
    def __init__(self, raw_text):
        self.processed_txt = self.clean_text(raw_text)
        all_chars = list(set(self.processed_txt))
        all_chars.sort()
        self.vocab = {c:i for i,c in enumerate(all_chars)}
        self.chars_in_txt = list(self.processed_txt)
        self.curr_max_vocab_tok_idx = SimpleBPETokenizer.get_max_vocab_token_idx(self.vocab)
        self.merges = []  # Store the order of merges

    def clean_text(self, text):
        strings2replace = [
                 '\r\n\r\nâ\x80\x9c', # new paragraph
                 'â\x80\x9c',         # open quote
                 'â\x80\x9d',         # close quote
                 '\r\n',              # new line
                 'â\x80\x94',         # hyphen
                 'â\x80\x99',         # single apostrophe
                 'â\x80\x98',         # single quote
                 '_',                 # underscore, used for stressing
                 ]

        for rpl in strings2replace:
            rexp = re.compile(r'%s'%rpl)
            text = rexp.sub(' ', text)


        # remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # remove numbers
        text = re.sub(r'\d+','',text)

        # and make everything lower-case
        text = text.lower()

        return text

    
    def update_vocab(self, token, token_idx):
        self.vocab[token] = token_idx

    def get_max_vocab_token_idx(vocab_dict: dict):
        return max(vocab_dict.values())

    def get_most_freq_pair(chars_in_txt):
        pairs = {}
        max_pair = ''
        max_freq = 1
        for c in range(len(chars_in_txt)-1):
            pair = chars_in_txt[c] + chars_in_txt[c+1]
            freq = pairs.get(pair, 0) + 1
            if freq > max_freq:
                max_freq = freq
                max_pair = pair
            pairs[pair] = freq
        return (max_pair, max_freq)

    def update_text_with_new_pair(chars_in_txt, pair_2_update):
        new_txt_chars = []
        i=0
        while (i < len(chars_in_txt)-1):
            pair = chars_in_txt[i] + chars_in_txt[i+1]
            if pair == pair_2_update:
                new_txt_chars.append(pair)
                i += 2
            else:    
                new_txt_chars.append(chars_in_txt[i])
                i += 1
        if i < len(chars_in_txt):
            new_txt_chars.append(chars_in_txt[i])
    
        return new_txt_chars

    def train(self, num_itr = 1000):
        print(f"Initial txt length size = {len(self.chars_in_txt)}")

        tmp_txt = self.chars_in_txt
        itr = num_itr
        while(itr > 0):
            pair, freq = SimpleBPETokenizer.get_most_freq_pair(tmp_txt)
            # print(f"Max pair = {pair} freq = {freq}")
            self.curr_max_vocab_tok_idx+=1
            self.update_vocab(pair, self.curr_max_vocab_tok_idx)
            self.merges.append(pair)  # Store the merge order
            tmp_txt = SimpleBPETokenizer.update_text_with_new_pair(tmp_txt, pair)
            itr -= 1
        print(f"Max pair = {pair} freq = {freq}")
        print(f"length of txt after replacement {len(tmp_txt)}")
        print(f"Vocab size = {len(self.vocab)}")

    
    def encode(self, text):
        # Clean the text using the same preprocessing
        cleaned_text = self.clean_text(text)

        # Convert to list of characters
        chars = list(cleaned_text)

        # Apply all merges in the order they were learned
        for pair in self.merges:
            chars = SimpleBPETokenizer.update_text_with_new_pair(chars, pair)

        # Convert tokens to their vocabulary indices
        token_ids = [self.vocab.get(token, -1) for token in chars]

        return token_ids
    
    def decode(self, tokens):
        # Create reverse vocabulary mapping (index -> token)
        reverse_vocab = {idx: token for token, idx in self.vocab.items()}

        # Convert token IDs back to tokens
        chars = [reverse_vocab.get(token_id, '') for token_id in tokens]

        # Join them to form the decoded text
        return ''.join(chars)

