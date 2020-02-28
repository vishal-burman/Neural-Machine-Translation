class Attention:
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn=nn.Linear((enc_hid_dim*2)+dec_hid_dim, dec_hid_dim)
        self.v=nn.Linear(dec_hid_dim, 1, bias=False)

        def forward(self, hidden, encoder_outputs):

            # hidden=[batch_size, dec_hid_dim]
            # encoder_outputs=[src_len, batch_size, enc_hid_dim*2]

            batch_size=encoder_outputs[1]
            src_len=encoder_outputs.shape[0]

            # repeat decoder hidden state src_len times
            hidden=hidden.unsqueeze(1).repeat(1, src_len, 1)