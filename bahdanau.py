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


class BahdanauDecoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, encoder_hidden_dim,
                 decoder_hidden_dim, attention, dropout_p):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.attention = attention
        self.gru = nn.GRU((encoder_hidden_dim*2) +
                          embedding_dim, decoder_hidden_dim)
        self.out = nn.Linear((encoder_hidden_dim*2) +
                             embedding_dim+decoder_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden, encoder_outputs, src_mask=None):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        attentions = self.attention(hidden, encoder_outputs, src_mask)
        a = attentions.unsqueeze(1)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        '''Multiplying two matrices that are batched. If 1st batched matrix size is (10, 3, 5) 
        and 2nd batched matrix size is (10, 5, 6) then output matrix size will be (10, 3, 6)'''
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.transpose(0, 1)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        linear_input = torch.cat((output, weighted, embedded), dim=1)

        output = self.out(linear_input)
        return output, hidden.squeeze(0), attentions
