import torch
import torch.nn as nn
import torch.nn.functional as F

import cuda_functional2 as MF


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


class StackedFastKNN(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout, use_tanh=1):
        super(StackedFastKNN, self).__init__()
        #self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(MF.FastKNNCell(
                input_size, rnn_size,
                dropout = dropout if i+1 < num_layers else 0.0,
                use_tanh = use_tanh
            ))
            input_size = rnn_size

    def forward(self, input, c_0):
        if isinstance(c_0, tuple):
            c_0 = c_0[0]
        c_1 = []
        assert c_0.size(0) == self.num_layers
        c_0 = [ x.squeeze(0) for x in c_0.chunk(self.num_layers, 0) ]

        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, c_0[i])
            input = h_1_i
            #if i + 1 != self.num_layers:
            #    input = self.dropout(input)
            c_1 += [c_1_i]

        c_1 = torch.stack(c_1)

        return input, c_1


class FastKNN_(MF.FastKNN):
    def __init__(self, n_in, n_out, num_layers=1, dropout=0.0, rnn_dropout=0.0,
            bidirectional=False, use_tanh=1):
        super(FastKNN_, self).__init__(
            n_in, n_out,
            depth = num_layers,
            dropout = dropout,
            rnn_dropout = rnn_dropout,
            use_tanh = use_tanh
        )
        assert bidirectional==False, "Bidirectional RNN not supported yet."


