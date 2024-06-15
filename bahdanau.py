import torch
from torch import nn


class BahdanauEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim,
                 encoder_hidden_dim, decoder_hidden_dim, dropout_p):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, encoder_hidden_dim,
                          bidirectional=True)
        # Times 2 because of bidirectional (I guess)
        self.linear = nn.Linear(encoder_hidden_dim*2, decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)
        hidden = torch.tanh(self.linear(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]),
                      dim=1)
        ))
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, query_size=None,
                 key_size=None, dropout_p=0.15):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_size = hidden_size if query_size is None else query_size
        # Assume that encoder is bidirectional (times 2)
        self.key_size = 2*hidden_size if key_size is None else key_size

        self.query_layer = nn.Linear(self.query_size, hidden_size)
        self.key_layer = nn.Linear(self.key_size, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, hidden, encoder_outputs, src_mask=None):
        query_out = self.query_layer(hidden)
        key_out = self.key_layer(encoder_outputs)
        energy_input = torch.tanh(query_out + key_out)
        energies = self.energy_layer(energy_input).squeeze(2)

        if src_mask is not None:
            energies.data.masked_fill_(src_mask == 0, float("-inf"))

        weights = torch.softmax(energies, dim=0)
        return weights.transpose(0, 1)
