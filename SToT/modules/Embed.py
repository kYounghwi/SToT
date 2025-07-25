import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from sklearn.preprocessing import StandardScaler
import math
import numpy as np
import os


class GeoPositionalEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(GeoPositionalEmbedding, self).__init__()
        # Compute the national position
        self.location_embedding = nn.Linear(c_in, d_model)

    def forward(self, x):
        return self.location_embedding(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # cross autocorrelation / channel-wise self-attention + conv1d
        self.tokenConv = nn.Linear(c_in, d_model)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        x = self.tokenConv(x)
        # print(f'Token: {x.size()}')
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, root_path, node_num, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        # spatial (24, 3)
        national_pos = np.load(os.path.join(root_path), allow_pickle=True)[:, :]
        scaler = StandardScaler()
        scaler.fit(national_pos)
        national_pos = scaler.transform(national_pos)   # scaler 풀어보기 @@
        self.national_pos = torch.from_numpy(national_pos).float()
        self.national_embedding = GeoPositionalEmbedding(3 * (self.national_pos.shape[0] // node_num),  # (node, 3*f) -> (node, d_model) projection
                                                         d_model)
        # value
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # temporal
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

        # others
        self.node_num = node_num

    def forward(self, x, x_mark):
        B, L, D, _ = x.shape

        x = self.value_embedding(x)     # (B, L, S, 1) -> (B, L, S, D)
        x2 = self.temporal_embedding(x_mark).unsqueeze(2).repeat(1, 1, self.node_num, 1)
        national_position = self.national_pos.to(x.device) 
        x3 = self.national_embedding(national_position).unsqueeze(1).unsqueeze(0).repeat(B, 1, L, 1).permute(0, 2, 1, 3).contiguous()       

        return self.dropout(x+x2+x3)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)


class STDataEmbedding_inverted_geo_embed(nn.Module):
    def __init__(self, c_in, d_model, root_path, seq_len, embed_type='fixed', freq='h', dropout=0.1):
        super(STDataEmbedding_inverted_geo_embed, self).__init__()
        # spatial (24, 3)
        national_pos = np.load(os.path.join(root_path), allow_pickle=True)[:, :]
        # self.national_pos = torch.from_numpy(national_pos).int()
        self.national_pos = torch.from_numpy(national_pos)
        print(f'national position size: {self.national_pos.size()}')

        # value
        self.value_embedding = nn.Linear(c_in, d_model)
        # temporal
        # self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
        #                                             freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
        #     d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

        self.lon_embed = nn.Linear(43, seq_len)
        self.lat_embed = nn.Linear(43, seq_len)
        self.num_embed = nn.Embedding(43, seq_len)


    def forward(self, x, x_mark):

        B, S, N, = x.shape
        x = x.permute(0, 2, 1).contiguous()     # B N S
        x_mark = x_mark.permute(0, 2, 1).contiguous()   
        national_position = self.national_pos.to(x.device).unsqueeze(0).repeat(B, 1, 1)

        lon = self.lon_embed(national_position[:, :, 0:1].permute(0, 2, 1).float())
        lat = self.lat_embed(national_position[:, :, 1:2].permute(0, 2, 1).float())
        num = self.num_embed(national_position[:, 0:1, 2].int())

        national_position = torch.cat([lon, lat, num], axis=1)

        x = self.value_embedding(torch.cat([x, national_position, x_mark], axis=1))     # (B, N, S) -> (B, N, D)

        return self.dropout(x)
