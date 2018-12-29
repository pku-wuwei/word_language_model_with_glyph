#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import ImageFont
import json

default_fontsize = 12
default_font = ImageFont.truetype('/data/nfsdata/nlp/fonts/Noto-hinted/NotoSansCJKsc-Regular.otf', default_fontsize)

with open(os.path.join('/data/nfsdata/nlp/projects/glyph_lm/', 'dictionary.json')) as fo:
    idx2word = json.load(fo)['idx2word']


def vocab_glyph_embedding(font=default_font):
    r = np.array([np.reshape(render_text_with_token_id(i, font), -1) for i in range(len(idx2word))])
    print(r.shape)
    return r


def render_text_with_token_id(token_id, font):
    word = idx2word[token_id]
    if len(word) > 1:
        return np.zeros((font.size + 1, font.size + 1))
    else:
        return pad_mask(render_text(word, font), font.size)


def render_text(text, font):
    mask = font.getmask(text)
    size = mask.size[::-1]
    a = np.asarray(mask).reshape(size)
    return a


def ascii_print(glyph_array):
    print('='*100)
    for l in glyph_array:
        char_line = ''
        for c in l:
            if c != 0:
                char_line += str(c % 2)
            else:
                char_line += ' '
        print(char_line)


def pad_mask(mask, fontsize):
    padded_mask = []
    for l in mask:
        padded_mask.append(l.tolist() + [0] * (fontsize + 1 - len(l)))
    for i in range(fontsize + 1 - len(padded_mask)):
        padded_mask.append([0]*(fontsize + 1))
    return np.array(padded_mask)


if __name__ == '__main__':
    vocab_glyph_embedding()
    # for i in range(1000):
    #     feat = render_text_with_token_id(i, default_font)
    #     print(i, idx2word[i], feat.shape)
    #     ascii_print(feat)
