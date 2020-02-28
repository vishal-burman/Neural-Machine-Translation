import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_query, hidden, encoder_outputs):
        # input=[batch_size]
        # hidden=[batch_size, dec_hid_dim]
        # encoder_outputs=[src_len, batch_size, enc_hid_dim*2]

        input_query = input_query.unsqueeze(0)
        # input_query=[1, batch_size]

        embedded = self.dropout(self.embedding(input_query))
