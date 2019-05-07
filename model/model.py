import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class DeepGRU(BaseModel):
    def __init__(self, in_features=63, num_gestures=3):
        super(DeepGRU, self).__init__()
        self.encoder_0 = nn.GRU(input_size=in_features, hidden_size=512,
                                num_layers=2)
        self.encoder_1 = nn.GRU(input_size=512, hidden_size=256, num_layers=2)
        self.encoder_2 = nn.GRU(input_size=256, hidden_size=128)
        self.attention_context = nn.Linear(128, 128)
        self.attention_gru = nn.GRU(input_size=128, hidden_size=128)
        self.classification_0 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.Linear(256, 256))
        self.classification_1 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.Linear(256, num_gestures))


    def forward(self, x):
        # Encoder
        x, h0 = self.encoder_0(x)
        x, h1 = self.encoder_1(x)
        x, h2 = self.encoder_2(x)

        # Attention Module
        h = x.permute((1, 0, 2))
        h_l = h[:, -1]
        c = self.attention_context(h)
        c = torch.bmm(c, h_l.unsqueeze(-1)).squeeze(-1)
        c = F.softmax(c, dim=0)
        c = torch.bmm(c.unsqueeze(1), h).squeeze(1)
        c_prime, _ = self.attention_gru(c.unsqueeze(0), h_l.unsqueeze(0))
        o = torch.cat((c, c_prime.squeeze()), dim=-1)

        # Classification
        y = self.classification_1(F.relu(self.classification_0(o)))

        return F.softmax(F.relu(y), dim=1)
