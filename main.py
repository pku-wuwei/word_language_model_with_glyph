# coding: utf-8
import argparse
import json
import math
import os
import time

import torch

from data import Corpus
from model import RNNModel

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='/data/nfsdata/nlp/datasets/language_modeling/ctb_v6/data/utf8/raw/', help='location of the data corpus')
parser.add_argument('--save', type=str, default='/data/nfsdata/nlp/projects/glyph_lm/', help='path to save the final model')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=1500, help='size of word embeddings')
parser.add_argument('--nfeat', type=int, default=300, help='size of feat embeddings')
parser.add_argument('--nhid', type=int, default=1800, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=500, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=500, metavar='N', help='batch size')
parser.add_argument('--use_glyph', action='store_true', help='use glyph in word embedding')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights(default)')
parser.add_argument('--font_size', type=int, default=12, help='fontsize for glyph')
parser.add_argument('--font_path', type=str, default='/data/nfsdata/nlp/fonts/Noto-hinted/NotoSansCJKsc-Regular.otf', help='the path of font for glyph')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--no_cuda', action='store_true', help='use CUDA (default) or not')
parser.add_argument('--reload', action='store_true', help='reload data from files or load from cache(default)')
parser.add_argument('--reverse', action='store_true', help='train the language model from forward(default) or backward')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='report interval')
parser.add_argument('--gpu_id', type=str, default='0', help='the gpu id to train language model')
parser.add_argument('--bptt', type=int, default=35, help='sequence length')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
torch.manual_seed(args.seed)
if torch.cuda.is_available() and args.no_cuda:
    print("WARNING: You have a CUDA device, so you should probably run without --no_cuda")
device = torch.device("cpu" if args.no_cuda else "cuda")

if args.reload:
    print('Producing dataset...')
    corpus = Corpus(args.data, reverse=args.reverse)
    torch.save(corpus, os.path.join(args.save, 'corpus.pkl'))
    with open(os.path.join(args.save, 'dictionary.json'), 'w') as fo:
        json.dump({'idx2word': corpus.dictionary.idx2word, 'word2idx': corpus.dictionary.word2idx}, fo)
else:
    print('Loading cached dataset...')
    corpus = torch.load(os.path.join(args.save, 'corpus.pkl'))
print(F'train:{len(corpus.train)}\nvalid:{len(corpus.valid)}\ntest:{len(corpus.test)}')


def batchify(data, bsz):
    """
    取训练集中所有的token，整理成batch_size个长Tensor
    :param data: 整个数据集的所有token
    :param bsz: batch_size
    :return: (long_seq_len, batch_size)
    """
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):  # 普通RNN
        return h.detach()
    else:  # LSTM
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, idx):
    """
    取source中第i个滑动窗口的数据作为data, 第i+1个滑动窗口的数据作为target, (seq_len, batch_size)
    :param source: 整理成batch的数据
    :param idx: 在长tensor上的index
    :return: 在长tensor上以index开头，最长为bptt，总共batch_size个样例组成的输入，以及下一个time_step作为gold
    """
    seq_len = min(args.bptt, len(source) - 1 - idx)  # 长度一般是bptt，在结尾处委屈求全一下
    data = source[idx: idx + seq_len]
    target = source[idx + 1: idx + 1 + seq_len]
    return data, target


def train():
    best_val_loss = 100

    ntokens = len(corpus.dictionary)
    train_data = batchify(corpus.train, args.batch_size)  # num_batches, batch_size
    val_data = batchify(corpus.valid, args.batch_size)
    model = RNNModel(rnn_type=args.model,
                     ntoken=ntokens,
                     ninp=args.emsize,
                     nfeat=args.nfeat,
                     nhid=args.nhid,
                     nlayers=args.nlayers,
                     font_path=args.font_path,
                     font_size=args.font_size,
                     dropout=args.dropout,
                     tie_weights=args.tied,
                     ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    print('start training...')
    hidden = model.init_hidden(args.batch_size)
    epoch_start_time = time.time()

    for epoch in range(args.epochs):

        model.eval()  # 在validation上测试
        total_loss = 0.
        with torch.no_grad():
            for idx in range(0, val_data.size(0) - 1, args.bptt):
                data, targets = get_batch(val_data, idx)
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, ntokens)  # (seq_len, batch, ntokens) -> (seq_len*batch, ntokens)
                total_loss += len(data) * criterion(output_flat, targets.view(-1)).item()
                hidden = repackage_hidden(hidden)
        val_loss = total_loss / len(val_data)
        best_val_loss = min(best_val_loss, val_loss)
        print('-' * 100)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f} | best valid ppl {:8.2f}'
              .format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), math.exp(best_val_loss)))
        print('-' * 100)
        epoch_start_time = time.time()
        if val_loss == best_val_loss:  # Save the model if the validation loss is best so far.
            torch.save(model, os.path.join(args.save, 'model.pkl'))
        else:
            args.lr /= 4.0

        model.train()  # 在training set上训练
        total_loss = 0.
        start_time = time.time()
        for i, idx in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            data, targets = get_batch(train_data, idx)
            hidden = repackage_hidden(hidden)
            model.zero_grad()  # 求loss和梯度
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets.view(-1))
            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # 用梯度更新参数
            for p in model.parameters():
                p.data.add_(-args.lr, p.grad.data)

            if i % args.log_interval == 0 and i > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} |loss {:5.2f} | ppl {:8.2f}'
                      .format(epoch + 1, i, len(train_data) // args.bptt, args.lr, elapsed * 1000 / args.log_interval,
                              cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()


if __name__ == '__main__':
    train()
