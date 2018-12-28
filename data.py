import os
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, reverse=False):
        self.dictionary = Dictionary()
        self.files = [f for f in os.listdir(path) if f.endswith('.raw')]
        train_files = self.files[:int(len(self.files)*0.8)]
        test_files = self.files[int(len(self.files)*0.8):int(len(self.files)*0.9)]
        valid_files = self.files[int(len(self.files)*0.9):]
        self.train = self.tokenize([os.path.join(path, f) for f in train_files], reverse)
        self.valid = self.tokenize([os.path.join(path, f) for f in valid_files], reverse)
        self.test = self.tokenize([os.path.join(path, f) for f in test_files], reverse)

    def tokenize(self, paths, reverse=False):
        """Tokenizes a text file."""
        tokens = []
        for path in paths:
            print(F'handle {path}')
            with open(path, 'r', encoding='utf8') as fi:
                for line in fi:
                    tokens += list(line.strip()) + ['<eos>']
        ids = torch.LongTensor(len(tokens))
        if reverse:
            tokens.reverse()
        for i, token in enumerate(tokens):
            ids[i] = self.dictionary.add_word(token)
        return ids
