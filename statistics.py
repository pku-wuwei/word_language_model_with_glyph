# encoding: utf-8
"""
@author: wuwei 
@contact: wu.wei@pku.edu.cn

@version: 1.0
@license: Apache Licence
@file: statistics.py
@time: 18-12-27 上午11:21

统计语料的属性：unigram, bigram, trigram的数目，条件概率和困惑度
"""
import math
import os
import pickle
from collections import Counter


class DatasetStatistics(object):
    def __init__(self, data_dir):
        self.sentences = []
        for file_name in os.listdir(data_dir):
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf8') as fi:
                for i, line in enumerate(fi):
                    if not line.startswith('<'):
                        self.sentences.append(line.strip())
        token_list = [list(sent) + ['<eos>'] for sent in self.sentences]
        train_tokens = [i for j in token_list[: int(0.8 * len(token_list))] for i in j]
        eval_tokens = [i for j in token_list[int(0.8 * len(token_list)):] for i in j]

        self.list_of_tokens = [i for j in token_list for i in j]
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.bigram_prob = {}
        self.trigram_prob = {}
        self.vocab = pickle.load(open('/data/nfsdata/nlp/projects/glyph_lm/corpus.pkl', 'rb'))
        cleaned_train_tokens = []
        for t in train_tokens:
            if t in self.vocab:
                cleaned_train_tokens.append(t)
            else:
                cleaned_train_tokens.append('<unk>')

        cleaned_eval_tokens = []
        for t in eval_tokens:
            if t in self.vocab:
                cleaned_eval_tokens.append(t)
            else:
                cleaned_eval_tokens.append('<unk>')
        self.get_counts(cleaned_train_tokens)
        self.get_ppl(cleaned_eval_tokens)

    def get_counts(self, train_token_list):
        self.unigram_counts.update(train_token_list)
        for i, token in enumerate(train_token_list):
            if i > 0:
                self.bigram_counts[' '.join(train_token_list[i - 1: i + 1])] += 1
                if i > 1:
                    self.trigram_counts[' '.join(train_token_list[i - 2: i + 1])] += 1

        for bigram in self.bigram_counts:
            self.bigram_prob[bigram] = (self.bigram_counts[bigram] + 1) / (
                        self.unigram_counts[bigram.split(' ')[0]] + len(self.unigram_counts))
        for trigram in self.trigram_counts:
            const_bigram = ' '.join(trigram.split(' ')[:-1])
            self.trigram_prob[trigram] = (self.trigram_counts[trigram] + 1) / (
                        self.bigram_counts[const_bigram] + len(self.unigram_counts))
        print(self.unigram_counts.most_common()[: -5: -1])
        print(self.bigram_counts.most_common(10))
        print(self.trigram_counts.most_common(10))

    def get_ppl(self, eval_token_list):
        unigram_sum = 0
        bigram_sum = 0
        trigram_sum = 0
        for i, token in enumerate(eval_token_list):
            unigram_sum += math.log(self.unigram_counts[token] / len(eval_token_list))
            if i > 0:
                bigram = ' '.join(eval_token_list[i - 1: i + 1])
                prob = 1 / len(self.unigram_counts)
                if bigram in self.bigram_prob:
                    prob = self.bigram_prob[bigram]
                    bigram_sum += math.log(prob)
                if i > 1:
                    trigram = ' '.join(eval_token_list[i - 2: i + 1])
                    prob = 1 / len(self.unigram_counts)
                    if trigram in self.trigram_prob:
                        prob = self.trigram_prob[trigram]
                        trigram_sum += math.log(prob)
        print(F'perplexity for unigram: {math.exp(-1.0 * unigram_sum / len(eval_token_list))} ')
        print(F'perplexity for bigram: {math.exp(-1.0 * bigram_sum / len(eval_token_list))} ')
        print(F'perplexity for trigram: {math.exp(-1.0 * trigram_sum / len(eval_token_list))} ')


if __name__ == '__main__':
    s = DatasetStatistics('/data/nfsdata/nlp/datasets/language_modeling/ctb_v6/data/utf8/raw')
