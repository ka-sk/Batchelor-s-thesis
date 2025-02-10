from torch import nn
import torch

'''
https://www.learnpytorch.io/04_pytorch_custom_datasets/
'''

# Create a convolutional neural network
class CnnModel(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,  # how big is the square that's going over the image?
                      stride=1,  # default
                      padding=1,
                      dtype=torch.double),
            # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      dtype=torch.double),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(hidden_units, hidden_units, 3, padding=1, dtype=torch.double),
            nn.ReLU(),
            nn.Conv1d(hidden_units, hidden_units, 3, padding=1, dtype=torch.double),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.output = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units * 2790,
                      out_features=output_shape,
                      dtype=torch.double)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.output(x)
        # print(x.shape)
        return x

