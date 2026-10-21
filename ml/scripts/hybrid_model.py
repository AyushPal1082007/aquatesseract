"""
Hybrid LSTM + Transformer for OceanVerse dissolved-oxygen (DO) reconstruction.

One sample = one ocean column in one year:
    x_factor  [B, 33, 8]    8 factors per depth
    do_series [B, 33, 11]   11-year window of DO per depth (centre index = target year)
    x_geo     [B, 5]        geography features
    valid     [B, 33]       1 where the depth cell is ocean (graph.mask)

Pipeline
    1) LSTM over the 11-year window at every depth      -> temporal feature per depth
    2) Transformer encoder over the 33 depths            -> vertical context
    3) head                                              -> mean and log-variance of DO per depth

The centre of the DO window is the target year itself, so it is replaced by a learned token
(same trick as the authors' baselines, "avoid label leakage").
"""
import torch
import torch.nn as nn


class LSTMTransformerHybrid(nn.Module):
    def __init__(self, factor_dim=8, geo_dim=5, time_len=11, profile_len=33,
                 lstm_hidden=32, d_model=64, n_heads=4, n_layers=2, d_ff=128,
                 dropout=0.1, do_scale=5.0):
        super().__init__()
        self.time_len = time_len
        self.profile_len = profile_len
        self.center = time_len // 2
        self.do_scale = do_scale  # DO values are ~0-0.4, scale them up for the network

        self.center_token = nn.Parameter(torch.zeros(profile_len))
        # input per time step: [DO value, "is there a value" flag]
        self.lstm = nn.LSTM(input_size=2, hidden_size=lstm_hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.in_proj = nn.Linear(2 * lstm_hidden + factor_dim, d_model)
        self.geo_proj = nn.Linear(geo_dim, d_model)
        self.depth_emb = nn.Parameter(torch.zeros(profile_len, d_model))
        nn.init.normal_(self.depth_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x_factor, do_series, x_geo, valid):
        """Returns (mu, logvar) in the *scaled* space (DO * do_scale), shape [B, 33] each."""
        B = x_factor.shape[0]
        assert do_series.shape[-1] == self.time_len, "time window length mismatch"

        x_factor = torch.nan_to_num(x_factor)
        x_geo = torch.nan_to_num(x_geo)
        do = torch.nan_to_num(do_series).clone()
        flag = (do != 0).float()
        do[:, :, self.center] = self.center_token      # hide target-year DO
        flag[:, :, self.center] = 0.0

        seq = torch.stack((do * self.do_scale, flag), dim=-1)          # [B, 33, T, 2]
        seq = seq.reshape(B * self.profile_len, self.time_len, 2)
        out, _ = self.lstm(seq)                                         # [B*33, T, 2h]
        feat = out[:, self.center, :].reshape(B, self.profile_len, -1)  # [B, 33, 2h]

        tok = self.in_proj(torch.cat((feat, x_factor), dim=-1))
        tok = tok + self.geo_proj(x_geo).unsqueeze(1) + self.depth_emb

        pad = valid < 0.5            # True = ignore this depth (below sea floor)
        pad[:, 0] = False            # never mask a whole row
        h = self.norm(self.encoder(tok, src_key_padding_mask=pad))
        o = self.head(h)
        return o[..., 0], o[..., 1].clamp(-10.0, 3.0)

    @torch.no_grad()
    def predict(self, x_factor, do_series, x_geo, valid):
        """Returns (mean, std) in original DO units."""
        mu, lv = self.forward(x_factor, do_series, x_geo, valid)
        return mu / self.do_scale, torch.exp(0.5 * lv) / self.do_scale
