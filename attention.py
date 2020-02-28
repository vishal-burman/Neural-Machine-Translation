class Attention:
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn=nn.Linear((enc_hid_dim*2)+dec_hid_dim, dec_hid_dim)
        self.v=nn.Linear(dec_hid_dim, 1, bias=False)
