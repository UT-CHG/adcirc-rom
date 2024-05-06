import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
    """Simple shallow feed-forward network
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
              layers.append(nn.LeakyReLU())
        
        # use relu for output activation
        layers.append(nn.ReLU)
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        """Evaluate the network
        """

        return self._model(x)
