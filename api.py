# encoding: utf-8
"""
@author: wuwei 
@contact: wu.wei@pku.edu.cn

@version: 1.0
@license: Apache Licence
@file: api.py.py
@time: 18-9-14 下午7:35

输入前n个字，输出第n+1个字在整个词表里的概率分布
"""
import json
import os

import torch


class LanguageModel(object):
    """用真实文本数据训练的语言模型，用于预测OCR识别出的中文字符的概率分布"""

    def __init__(self, model_path='./data', vocab_path='./data/dictionary.json'):
        self.device = torch.device("cuda")
        with open(os.path.join(model_path, 'forward.pt'), 'rb') as ff, \
                open(os.path.join(model_path, 'reverse.pt'), 'rb') as fr, \
                open(vocab_path, 'rb') as fv:
            self.forward_model = torch.load(ff).to(self.device)
            self.backward_model = torch.load(fr).to(self.device)
            self.vocab = json.load(fv)
        self.forward_model.eval()
        self.backward_model.eval()

    def get_probability_from_words(self, context: str, model):
        """输入前n个字，输出第n+1个字在整个词表里的概率分布"""
        if context:
            inputs = [self.vocab['word2idx'].get(c, 0) for c in context]
            hidden = model.init_hidden(1)
            inputs = torch.tensor(inputs).view(-1, 1).to(self.device)
            with torch.no_grad():
                output, hidden = model(inputs, hidden)
                prob = output[-1].squeeze().cpu()
                prob = torch.nn.Softmax(dim=-1)(prob)
            return prob.tolist()
        else:
            return [1 / len(self.vocab['idx2word'])] * len(self.vocab['idx2word'])

    def get_bidirectional_prob(self, sentence: str):
        out_probs = []
        for i, c in enumerate(sentence):
            left_context = sentence[:i]
            right_context = sentence[i + 1:][::-1]
            print(left_context, right_context)
            out_probs.append({'forward': self.get_probability_from_words(left_context, self.forward_model),
                              'backward': self.get_probability_from_words(right_context, self.backward_model)})
        return out_probs


if __name__ == '__main__':
    lm = LanguageModel()
    probs = lm.get_bidirectional_prob('株洲电机')
    print(probs)
    for p in probs:
        print(len(p['forward']))
        print(len(p['backward']))
