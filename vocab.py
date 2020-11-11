import pickle


class Vocab(object):

    def __init__(self,
                 tokenizer=lambda w: w.lower().split(),
                 is_lower=True,
                 num_words=None):
        self.freq = {}
        self.num_words = num_words
        self.is_lower = is_lower

        self.itos = {}
        self.stoi = {}
        self.reserved = ['_pad_', '_unk_', '_begin_', '_end_']

        for r_idx, r in enumerate(self.reserved):
            self.itos[r_idx] = r
            self.stoi[r] = r_idx

        self.tokenizer = tokenizer

    def _preprocess (self, texts):
        if self.is_lower:
            texts = map(lambda l: l.lower(), texts)
        sentence_token = map(self.tokenizer, texts)
        return sentence_token

    def fit(self, texts):
        sentence_token = self._preprocess (texts)
        self.fit_token(sentence_token)

    def transform(self, texts):
        sentence_token = self._preprocess (texts)

        index = []
        for sent in sentence_token:
            index.append([])
            for token in sent:
                index[-1].append(self.stoi.get(token, self.stoi['_unk_']))
        return index

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def fit_token(self, sentence_token):
        for sent in sentence_token:
            for token in sent:
                if token not in self.reserved:
                    self.freq[token] = self.freq.get(token, 0) + 1

        if not self.num_words:
            self.num_words = len(self.freq)
        # truncate vocab, get most common word
        top_words = sorted(self.freq.keys(), key=lambda k: self.freq[k], reverse=True)[:self.num_words]

        for word_idx, word in enumerate(top_words):
            self.itos[word_idx+4] = word  # 4 being the length of reserved token
            self.stoi[word] = word_idx+4

    def clear(self):
        self.itos = {}
        self.stoi = {}
        self.freq = {}
        for r_idx, r in enumerate(self.reserved):
            self.itos[r_idx] = r
            self.stoi[r] = r_idx

    def save(self, path):
        with open(path, 'wb') as f_:
            # only save itos, since everything can be
            # derived from either itos, or stoi.
            data = {
                "itos": self.itos,
                "num_words" : self.num_words,
                "is_lower": self.is_lower
            }

            pickle.dump(data, f_)

    @staticmethod
    # tokenizer is a function and thus can't be saved
    # re-provided as argument
    def load (path, tokenizer=lambda l: l.lower().split ()):
        with open (path, 'rb') as f_:
            data = pickle.load (f_)
            self = Vocab (
                tokenizer=tokenizer, 
                is_lower = data['is_lower'],
                num_words = data['num_words']
            )

            self.itos = data['itos']
            self.stoi = {w: idx for (idx, w) in self.itos.items ()}

            return self
    

if __name__ == "__main__":
    text = [
        "From my grandfather Verus I learned good morals and the government of my temper.",
        "From the reputation and remembrance of my father, modesty and a manly character.",
        "From my mother, piety and beneficence, and abstinence, not only from evil deeds, but even from evil thoughts",
        "From my great-grandfather, not to have frequented public schools, and to have had good teachers at home"
    ]
    vocab = Vocab(is_lower=True)
    sentence_token = vocab.fit_transform(text)
    for sent in sentence_token:
        print(sent)

    # test saving
    path_vocab = './bin/tokenizer.pkl'
    vocab.save (path_vocab)

    # test loading
    vocab = Vocab.load (path_vocab)

    # test transform preloaded
    sentence_token = vocab.transform(text)
    for sent in sentence_token:
        print(sent)

