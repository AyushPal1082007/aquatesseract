"""
The authors' point-prediction baselines (MLP / BiLSTM / Transformer), copied unchanged from their
OceanVerse repo (models.py) so they can run without torch_geometric / torch-sparse.
Source: https://github.com/jingwei-sjtu/OceanVerse  (check their licence before redistributing).
BaselineAdapter at the bottom only wraps them so they plug into hybrid_train.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        if d_model % 2 == 1:
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, key_padding_mask=src_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class Baseline_MLP(nn.Module):
    def __init__(self, factor_dim, geo_dim, hidden_dim, time_len, layer_num, edge_dim, meta_dim, profile_len=33):
        super(Baseline_MLP, self).__init__()
        self.profile_len = profile_len
        self.input_dim = geo_dim + factor_dim * profile_len + time_len * profile_len
        self.hidden_dim = hidden_dim
        self.output_dim = profile_len
        self.layer_num = layer_num
        self.token_embedding = nn.Parameter(torch.zeros(self.profile_len, 1))
        self.build()

    def build(self):
        mlp_layer = [
            nn.Linear(self.input_dim, self.hidden_dim)
        ]
        for _ in range(self.layer_num-2):
            mlp_layer.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        mlp_layer.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.mlp_model = nn.ModuleList(mlp_layer)
    
    def forward(self, x_geo, x_factor, do_series, edge_index, edge_attr, x_meta):
        do_series[:, :, int(do_series.shape[-1]/2)] = self.token_embedding.squeeze() # avoid label leakage
        x_factor = x_factor.reshape(x_factor.shape[0], -1)
        do_series = do_series.reshape(do_series.shape[0], -1)
        input_feature = torch.cat((x_geo, x_factor, do_series), dim=-1)
        for i in range(self.layer_num):
            input_feature = torch.relu(self.mlp_model[i](input_feature))
        output = torch.clamp(input_feature, min=-0.1, max=1)
        return output
    
class Baseline_BiLSTM(nn.Module):
    def __init__(self, factor_dim, geo_dim, hidden_dim, time_len, layer_num, edge_dim, meta_dim, profile_len=33):
        super(Baseline_BiLSTM, self).__init__()
        self.time_len = time_len
        self.factor_dim = factor_dim
        self.profile_len = profile_len
        self.geo_dim = geo_dim
        self.geo_encoder = nn.Linear(geo_dim, self.profile_len)
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.token_embedding = nn.Parameter(torch.zeros(self.profile_len, 1))
        d_model = self.factor_dim + self.time_len + 1
        self.temporal_model = nn.LSTM(input_size=d_model, hidden_size=self.hidden_dim, num_layers = self.layer_num, bidirectional = True, batch_first=True)
        self.decoder = nn.Linear(int(hidden_dim * 2), 1)
    def forward(self, x_geo, x_factor, do_series, edge_index, edge_attr, x_meta):
        do_series[:, :, int(do_series.shape[-1]/2)] = self.token_embedding.squeeze() # avoid label leakage
        x_geo = self.geo_encoder(x_geo).unsqueeze(-1)
        series_input = torch.cat((x_factor, do_series, x_geo), dim=-1)
        output, _ = self.temporal_model(series_input)
        output = self.decoder(torch.relu(output)).squeeze(-1)
        return output


class Baseline_Transformer(nn.Module):
    def __init__(self, factor_dim, geo_dim, hidden_dim, time_len, layer_num, edge_dim, meta_dim, profile_len=33):
        super(Baseline_Transformer, self).__init__()
        self.time_len = time_len
        self.factor_dim = factor_dim
        self.profile_len = profile_len
        self.geo_dim = geo_dim
        self.geo_encoder = nn.Linear(geo_dim, self.profile_len)
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.token_embedding = nn.Parameter(torch.zeros(self.profile_len, 1))
        d_model = self.factor_dim + self.time_len + 1
        self.transformer_encoder = TransformerEncoder(self.layer_num, d_model = d_model, num_heads=2, d_ff=self.hidden_dim)
        self.positional_encoding = PositionalEncoding(d_model, max_len=33)
        self.decoder_layer_1 = nn.Linear(d_model, int(d_model * 2))
        self.decoder_layer_2 = nn.Linear(int(d_model * 2), 1)
    def forward(self, x_geo, x_factor, do_series, edge_index, edge_attr, x_meta):
        do_series[:, :, int(do_series.shape[-1]/2)] = self.token_embedding.squeeze() # 避免标签泄漏
        x_geo = self.geo_encoder(x_geo).unsqueeze(-1)
        series_input = torch.cat((x_factor, do_series, x_geo), dim=-1)
        series_input = self.positional_encoding(series_input)
        output = self.transformer_encoder(series_input)
        output = self.decoder_layer_1(torch.relu(output))
        output = self.decoder_layer_2(torch.relu(output)).squeeze(-1)
        return output


class BaselineAdapter(nn.Module):
    """Gives the authors' baselines the same interface as LSTMTransformerHybrid (they only predict a mean)."""
    has_uncertainty = False

    def __init__(self, kind, hidden_dim, num_layers, factor_dim=8, geo_dim=5, time_len=11, edge_dim=3):
        super().__init__()
        cls = {"mlp": Baseline_MLP, "lstm": Baseline_BiLSTM, "transformer": Baseline_Transformer}[kind]
        self.net = cls(factor_dim=factor_dim, geo_dim=geo_dim, hidden_dim=hidden_dim, time_len=time_len,
                       layer_num=num_layers, edge_dim=edge_dim, meta_dim=geo_dim)
        self.do_scale = 1.0

    def forward(self, x_factor, do_series, x_geo, valid):
        x_factor = torch.nan_to_num(x_factor)
        do = torch.nan_to_num(do_series).clone()   # the baselines overwrite the centre time step in place
        mu = self.net(x_geo, x_factor, do, None, None, x_geo)
        return mu, torch.zeros_like(mu)

    @torch.no_grad()
    def predict(self, x_factor, do_series, x_geo, valid):
        mu, _ = self.forward(x_factor, do_series, x_geo, valid)
        return mu, torch.full_like(mu, float("nan"))
