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

class VisionNet(nn.Module):
    
    def __init__(self,
                 input_channels,
                 hidden_layers,
                 hidden_channels=32,
                 kernel_size=3
                 ):
    
        super().__init__()
       
        self.layers = []

        for i in range(hidden_layers):
            layer_in_channels = input_channels if not i else hidden_channels
            self.layers.append(nn.Conv2d(layer_in_channels, hidden_channels, kernel_size=kernel_size, padding=int(kernel_size/2)))
            self.layers.append(nn.BatchNorm2d(hidden_channels))
            self.layers.append(nn.LeakyReLU())

        self._encoder = nn.Sequential(*self.layers)
        #self.classifier = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.regressor = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        # filter out land from sea
        sea_mask = (x[:,0,...] > -10).unsqueeze(1)
        x = self._encoder(x)
        pred = F.relu(self.regressor(x))
        return pred * sea_mask
        #return self.classifier(x), F.relu(self.regressor(x))

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet4(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Output layer
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        # filter out land from sea
        sea_mask = (x[:,0,...] > -10).unsqueeze(1)
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        x4 = self.enc4(F.max_pool2d(x3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(x4, 2))

        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, x4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        return F.relu(self.out_conv(d1)) * sea_mask

class SegmentationUNet2(nn.Module):
    """Handles segmentation and regression."""

    
    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Output layer
        self.classifier = nn.Conv2d(base_channels, 2, kernel_size=1)
        self.regressor = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(x2, 2))

        # Decoder

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        return {"zeta_mask": self.classifier(d1), "zeta": F.relu(self.regressor(d1))}

def SegmentedLoss(positive_weight, regression_weight=.5):
    """Segmented loss criterion."""
    mse_loss = nn.MSELoss()
    def loss_fn(preds, target):
        mask = target["zeta_mask"].bool()
        seg_loss = nn.CrossEntropyLoss(weight=torch.Tensor([1-positive_weight, positive_weight]).cuda(mask.device))
        err_seg = seg_loss(preds["zeta_mask"], mask.squeeze(1).long())
        batch_size = mask.shape[0]
        image_reg_losses = []
        for i in range(batch_size):
            masked_pred = preds["zeta"][i][mask[i]]
            masked_target = target["zeta"][i][mask[i]]
            if masked_pred.numel() > 0:
                image_reg_losses.append(mse_loss(masked_pred, masked_target))

        if len(image_reg_losses):
            reg_loss = torch.stack(image_reg_losses).mean()
            #print("cross entropy loss", err_seg, "regression loss", reg_loss)
            return (1-regression_weight) * err_seg + regression_weight * reg_loss
        else:
            return err_seg

    return loss_fn
    