import torch
import torch.nn as nn
# import torch.nn.functional as F
# from linformer import Linformer

class FeedForwardNet(nn.Module):
    """Simple shallow feed-forward network with Dropout and BatchNorm
    """
    def __init__(self, input_dim, size=3):
        """Initialize the network
        """
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
                layers.append(nn.BatchNorm1d(curr_size)) # normalization
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(p=0.5)) # dropout

        # use relu for output activation
        layers.append(nn.ReLU()) 
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        """Evaluate the network
        """
        return self._model(x)

# # DenseNet attempt
# class DenseBlock(nn.Module):
#     def __init__(self, input_dim, growth_rate, num_layers, dropout_rate=0.0):

#         super(DenseBlock, self).__init__()
#         self.layers = nn.ModuleList()
#         self.dropout_rate = dropout_rate
#         for i in range(num_layers):
#             self.layers.append(nn.Sequential(
#                 nn.Linear(input_dim + i * growth_rate, growth_rate),
#                 nn.BatchNorm1d(growth_rate),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(dropout_rate)
#             ))
#     def forward(self, x):
#         outputs = [x]
#         for layer in self.layers:
#             new_input = torch.cat(outputs, dim=1)
#             out = layer(new_input)
#             outputs.append(out)
#         return torch.cat(outputs, dim=1)

# class DenseNet(nn.Module):
#     def __init__(self, input_dim, growth_rate, block_layers, num_blocks, output_dim=1, dropout_rate=0.0):
#         super(DenseNet, self).__init__()
#         self.blocks = nn.ModuleList()
#         num_features = input_dim
#         for _ in range(num_blocks):
#             block = DenseBlock(num_features, growth_rate, block_layers, dropout_rate)
#             self.blocks.append(block)
#             num_features += block_layers * growth_rate
#         self.fc = nn.Linear(num_features, output_dim)
#         self._initialize_weights()

#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         x = self.fc(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
    
# # Linear transformer attempt
# class LinformerAttention(nn.Module):
#     def __init__(self, input_dim, num_heads, k):

#         super(LinformerAttention, self).__init__()
#         # print('Currently on LINFORMER')
#         self.linformer = Linformer(
#             dim=input_dim,
#             seq_len=input_dim,  
#             depth=6, # num of layers
#             heads=num_heads,
#             k=k,               
#             dropout=0.1
#         )
#         self.fc = nn.Linear(input_dim, 1)

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.linformer(x)
#         x = self.fc(x[:, 0, :])  
#         return x
