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
        # embedded=[1, batch_size, emb_dim]

        a = self.attention(hidden, encoder_outputs)
        # a=[batch_size, src_len]

        a = a.unsqueeze(1)
        # a=[batch_size, 1, src_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs=[batch_size, src_len, enc_hid_dim*2]

        weighted = torch.bmm(a, encoder_outputs)
        # weighted=[batch_size, 1, enc_hid_dim*2]

        weighted = weighted.permute(1, 0, 2)
        # weighted=[1, batch_size, enc_hid_dim*2]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input=[1, batch_size, enc_hid_dim*2+emb_dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output=[seq_len, batch_size, dec_hid_dim*num_directions]
        # hidden=[n_layers*n_directions, batch_size, dec_hid_dim]

        # seq_len, n_layers and n_directions will always be 1 in this decoder
        # output=[1, batch_size, dec_hid_dim]
        # hidden=[1, batch_size, dec_hid_dim]
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        # embedded=[batch_size, emb_dim]
        # output=[batch_size, dec_hid_dim]
        # weighted=[batch_size, enc_hid_dim*2]

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction=[batch_size, output_dim]

        return prediction, hidden.squeeze(0)
