import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, size=5):
        super().__init__()
        self._size = size
        self._input_dim = input_dim
        layers = []
        last_size = input_dim
        base_size = 128
        sizes = [input_dim] + [base_size*2**(min(i,2*size-2-i)) for i in range(2*size-1)] + [1]
        print("layer sizes", sizes)
        for last_size, curr_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(last_size, curr_size))
            if curr_size > 1:
                layers.append(nn.BatchNorm1d(curr_size)) 
                layers.append(nn.LeakyReLU())

        layers.append(nn.ReLU())
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        """Evaluate the network
        """
        return self._model(x)


class SimpleTransformerBlock(nn.Module):
    """
    four main parts of this model:
      1. multi head self attention for feature correlation
      2. Skip connection + Batchnorm
      3. ANN (2 layers for now... probably more later)
      4. Skip connection + Batchnorm again
    """
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        #Multi head self attention
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        #Ann
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model) where seq_len = n_features
        """
        attn_output, _ = self.mha(x, x, x) 
        x = x + self.dropout1(attn_output)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        ff_output = self.linear2(F.gelu(self.linear1(x)))  
        x = x + self.dropout2(ff_output)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)

        return x


class SimpleFTTransformer(nn.Module):

    def __init__(
        self,
        n_features: int = 80,
        d_token: int = 32,
        n_blocks: int = 2,
        n_heads: int = 4,
        ff_factor: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        #scale + bias for each feature
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))

        #N blocks
        dim_feedforward = int(d_token * ff_factor)
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_token, n_heads, dim_feedforward, dropout=dropout)
            for _ in range(n_blocks)
        ])

        #Project to output (dimension is 1)
        self.head = nn.Linear(n_features * d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the network
        """
        x = x.unsqueeze(-1) * self.weight + self.bias
        for block in self.blocks:
            x = block(x)
        x = x.reshape(x.size(0), -1)
        out = self.head(x)
        return out
