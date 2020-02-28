import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding=nn.Embedding(input_dim, emb_dim)
        self.rnn=nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc=nn.Linear(enc_hid_dim*2, dec_hid_dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self, src):
        
        # src=[src_len, batch_size]
        embedded=self.dropout(self.embedding(src))
        #src=[src_len, batch_size, emb_dim]
        
        outputs, hidden=self.rnn(embedded)
        #outputs=[src_len, batch_size, hid_dim*num_directions]
        #hidden=[n_layers*num_directions, batch_size, hid_dim]

        #hidden is stacked [forward1, backward1, forward2, backward2, ...]
        #outputs are always from the last layer

        # hidden[-2, :, :] is the last of the forward RNN
        # hidden[-1, :, :] is the last of the backward RNN

        hidden=torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden
