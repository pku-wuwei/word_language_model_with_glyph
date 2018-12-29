import torch
import torch.nn as nn
import torch.nn.functional as F
from render import vocab_glyph_embedding, default_fontsize


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nfeat, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.glyph_embedding = torch.from_numpy(vocab_glyph_embedding()).float().to(torch.device("cuda"))
        self.glyph_embedder = CNNforFonts(cnn_type='simple', kernel_sizes=[3, 4, 5], output_channels=120, num_features=nfeat)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp + nfeat, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp + nfeat, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)  # 单层LSTM，从hidden_size映射到ntoken维度上的概率分布

        if tie_weights:
            if nhid != ninp + nfeat:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()  # 模型初始化时权值也要初始化

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1  # (-0.1, 0.1)的均匀分布，只对embedding和最后的线性层做初始化
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.glyph_embedder.fc1.weight.data.uniform_(-initrange, initrange)
        for m in self.glyph_embedder.convs1:
            m.weight.data.uniform_(-initrange, initrange)
            m.bias.data.zero_()
        self.glyph_embedder.fc1.bias.data.zero_()
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):  # 前向传播，输入输出加dropout，input:  (seq_len, batch)
        glyph_emb = self.glyph_embedding.index_select(0, input.flatten())  # (seq_len, batchsize, fontsize*fontsize)
        reshaped_glyph_emb = glyph_emb.reshape(input.shape[0], input.shape[1], -1)
        glyph_feat = self.glyph_embedder(reshaped_glyph_emb)  # (seq_len, batch, featsize)
        emb = self.drop(self.encoder(input))  # output: (seq_len, batch, num_directions * hidden_size)
        emb = torch.cat((emb, glyph_feat), 2)
        output, hidden = self.rnn(emb, hidden)  # hidden: (num_layers * num_directions, batch, hidden_size)
        output = self.drop(output)  # 过线性层之前reshape一下
        reshaped = output.view(output.size(0) * output.size(1), output.size(2))
        decoded = self.decoder(reshaped)
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):  # 用全零向量初始化隐层并返回
        weight = next(self.parameters())    # weight只是想获得别的参数的数据类型和存储位置
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class CNNforFonts(nn.Module):
    """用CNN将字体的灰度图卷积成特征向量"""
    def __init__(self, cnn_type, kernel_sizes, output_channels, num_features, dropout=0.5):
        super(CNNforFonts, self).__init__()
        self.cnn_type = cnn_type
        self.convs1 = nn.ModuleList([nn.Conv2d(1, output_channels, (K, default_fontsize)) for K in kernel_sizes])
        self.num_features = num_features
        self.fc1 = nn.Linear(len(kernel_sizes) * output_channels, num_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xinp):
        x = xinp.reshape(-1, 1, default_fontsize + 1, default_fontsize + 1)  # (seq_len*batchsize, 1,fontsize, fontsize)
        x = [F.relu(conv(x)) for conv in self.convs1]  # [(seq_len*batchsize, Co, h, w), ...]*len(Ks)
        m = nn.AdaptiveMaxPool2d((1, 1))
        x = [m(i).squeeze() for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(self.fc1(x))  # (N, len(Ks)*Co)
        x = x.reshape(-1, xinp.shape[1], self.num_features)
        return x  # (N, C)
