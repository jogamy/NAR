class MyTokenizer:
    def __init__(
        self,
        pad="<pad>",
        unk="<unk>",
        mask = "<mask>",
        extra_special_symbols=None,
    ):
        self.unk_word, self.pad_word, self.mask_word = unk, pad, mask
        self.symbols = []
        self.indices = {}
        self.pad_token_id = self.add_symbol(pad)
        self.unk_token_id = self.add_symbol(unk)
        self.mask_token_id = self.add_symbol(mask)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)
        self.vocab_size = len(self.symbols)
    
    # def vocab_size(self):
    #     return len(self.symbols)
    # def pad(self):
    #     """Helper to get index of pad symbol"""
    #     return self.pad_token_id

    # def unk(self):
    #     """Helper to get index of unk symbol"""
    #     return self.unk_index
        
    # def mask(self):
    #     return self.mask_index

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def word(self, id):
        if id < len(self.symbols):
            return self.symbols[id]
        return self.unk_word

    def add_symbol(self, word):
        """Adds a word to the dictionary"""
        if word not in self.indices:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.vocab_size = len(self.symbols)
            return idx

    def encode(self, line : list) -> list:
        ids = []

        for i, word in enumerate(line):
            idx = self.index(word)
            ids.append(idx)

        return ids

    def decode(self, ids, remove_special_token=True):
        strings = []

        for i, id in enumerate(ids):
            if remove_special_token:
                if id >= self.nspecial:
                    word = self.word(id)
                    strings.append(word)
            else:
                word = self.word(id)
                strings.append(word)

        return strings

    def read_vocab(self, path):
        f = open(path, 'r', encoding='utf-8-sig')
        for word in f:
            if word == " \n":   # to add space 
                word = " "
            else :
                word = word.strip()
            self.add_symbol(word)
        self.vocab_size = len(self.symbols)