import torch
import torch.nn as nn
import torch.nn.functional as F

import cuda_functional as MF


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class FastKNNCell(nn.Module):
    def __init__(self, n_in, n_out, activation=lambda x:x, highway=1):
        super(FastKNNCell, self).__init__()
        #if n_in != n_out: highway = 0
        self.n_in = n_in
        self.n_out = n_out
        self.highway = highway
        self.activation = activation

        self.input_op = nn.Linear(n_in, n_out, bias=False)
        self.lambda_op = nn.Linear(n_in, n_out)

        if highway:
            self.input_op2 = nn.Linear(n_in, n_out, bias=False) if n_in != n_out else (lambda x:x)
            self.highway_op = nn.Linear(n_in, n_out)
        else:
            self.highway_op = None

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        wx = self.input_op(input)
        decay = self.lambda_op(input)
        decay = F.sigmoid(decay)
        #c_1 = c_0*decay + wx*(1-decay)
        c_1 = (c_0-wx)*decay + wx
        h_1 = self.activation(c_1)
        if self.highway:
            transform = F.sigmoid(self.highway_op(input))
            input2 = self.input_op2(input)
            # h_1 = h_1*transform + input2*(1-transform)
            h_1 = (h_1-input2)*transform + input2
        return (h_1, c_1)

class StackedFastKNN(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedFastKNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(FastKNNCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        assert h_0.size(0) == self.num_layers
        assert c_0.size(0) == self.num_layers
        h_0 = [ x.squeeze(0) for x in h_0.chunk(self.num_layers, 0) ]
        c_0 = [ x.squeeze(0) for x in c_0.chunk(self.num_layers, 0) ]

        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class FastKNN(nn.Module):
    def __init__(self, n_in, n_out, num_layers=1, dropout=0.0,
            bidirectional=False, activation=(lambda x:x), rnn_dropout=0.0, highway=1):
        super(FastKNN, self).__init__()
        assert bidirectional==False, "Bidirectional RNN not supported yet."
        self.n_in = n_in
        self.n_out = n_out
        self.num_layers = num_layers
        self.activation = activation
        self.rnn_dropout = rnn_dropout
        self.highway = highway
        self.drop_o = nn.Dropout(dropout)
        self.rnn_lst = nn.ModuleList()

        for i in range(num_layers):
            self.rnn_lst.append(
                FastKNNCell(
                    n_in = n_in if i==0 else n_out,
                    n_out = n_out,
                    activation = activation,
                    highway = highway
                )
            )

    def forward(self, input, hidden):
        assert input.dim() == 3 # (len, batch, n_in)
        if hidden is None:
            zeros = torch.autograd.Variable(
                input.data.new(self.num_layers, input.size(1), self.n_out).zero_()
            )
            hidden = (zeros, zeros)
        assert isinstance(hidden, tuple) or isinstance(hidden, list)
        assert len(hidden) == 2
        prevh, prevc = hidden  # (num_layers, batch, n_out)
        assert prevc.dim() == 3 and prevh.dim() == 3
        prevc = prevc.chunk(self.num_layers, 0)
        prevh = prevh.chunk(self.num_layers, 0)

        prevx = input
        lstc, lsth = [], []
        for i, cell in enumerate(self.rnn_lst):
            h, c = self.fast_forward_(cell, prevx, (prevh[i], prevc[i]))
            if i < self.num_layers-1:
                prevx = self.drop_o(h)
            else:
                prevx = h
            lstc.append(c[-1])
            lsth.append(h[-1])

        return prevx, (torch.stack(lsth), torch.stack(lstc))

    def fast_forward_(self, cell, input, hidden):
        assert input.dim() == 3 # (len, batch, n_in)
        assert isinstance(hidden, tuple) or isinstance(hidden, list)
        assert len(hidden) == 2
        h0, c0 = hidden # (batch, n_out)
        assert c0.size() == h0.size()

        length, bs = input.size(0), input.size(1)
        n_in, n_out = input.size(2), self.n_out
        input_op, lambda_op, highway_op, input_op2 = cell.input_op, cell.lambda_op, cell.highway_op, cell.input_op2

        if self.training and (self.rnn_dropout>0):
            mask_x = self.get_dropout_mask_((bs,n_in), self.rnn_dropout)
            mask_x = mask_x.expand_as(input)
            x = input*mask_x
        else:
            x = input

        x_2d = x.view(-1, n_in)
        wx = input_op(x_2d).view(length, bs, n_out)
        decay = lambda_op(x_2d).view(length, bs, n_out)
        decay = F.sigmoid(decay)
        wx = wx*(1-decay)
        c = MF.weighted_cumsum(wx, decay, c0)
        h = self.activation(c)
        if self.highway:
            transform = highway_op(x_2d).view(length, bs, n_out)
            transform = F.sigmoid(transform)
            if n_in != n_out:
                input2 = input_op2(input.view(-1, n_in)).view(length, bs, n_out)
            else:
                input2 = input
            h = h*transform + input2*(1-transform)

        return h, c

    def get_dropout_mask_(self, size, p):
        w = self.input_op.weight.data
        return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))


