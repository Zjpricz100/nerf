# --- Neural Networks Required for the NERF in Pytorch --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from nerf import volrend


class PositionalEncoder(nn.Module):
    def __init__(self, L, input_dim=2):
        super().__init__()
        self.L = L
        self.input_dim = input_dim

        # Arrange the frequencies for embedding
        # NOTE: We save the buffer of frequencies so we do not tune these values as weights in backprop.
        frequencies = torch.tensor(math.pi * 2.0 ** (np.arange(self.L)), dtype=torch.float32)
        self.register_buffer('frequencies', frequencies)

    def get_output_dim(self):
        return (2 * self.L * self.input_dim) + self.input_dim


    def forward(self, x):
        """
        Applies sinuosoidal embeddings to increase the dimensionality of input x

        x: Input tensor of shape (batch_size, in_dim)

        Returns: Encoded tensor of shape (batch_size, out_dim)
        """

        
        # Get a scaled version of X (batch_size, in_dim, L)
        # * torch.reshape(x, (1, 1, self.L))
        scaled_x = x.unsqueeze(-1) * self.frequencies
        
        B, C = scaled_x.shape[0:2]
        sines = torch.sin(scaled_x)
        cosines = torch.cos(scaled_x)



        interleaved = torch.empty((B, C, 2 * self.L), device=x.device, dtype=x.dtype)
        interleaved[..., 0::2] = sines
        interleaved[..., 1::2] = cosines

        interleaved_flattened = torch.flatten(interleaved, start_dim=1)


        final_encoding = torch.cat((x, interleaved_flattened), dim=-1)
        
        return final_encoding
        
        

class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()

        L = params["L"]
        n_layers = params["n_layers"]
        width = params["width"]
        
        self.encoder = PositionalEncoder(L=L)
        self.encoding_dim = self.encoder.get_output_dim()
        print(f"Encoder Dim: {self.encoding_dim}")

        self.layers = nn.ModuleList()
        current_dim = self.encoding_dim

        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(current_dim, width))
            current_dim = width


        # Append the last layer
        self.layers.append(nn.Linear(current_dim, 3))
        
    def forward(self, x):
        # Positional encoding
        x = self.encoder(x)

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        x = F.sigmoid(self.layers[-1](x))
        return x

# 3D Nerf
class MLP_3D(nn.Module):
    def __init__(self, params):
        super().__init__()
        L_coords = params["L_coords"]
        L_direction = params["L_direction"]
        n_layers = params["n_layers"]
        width = params["width"]
        
        near = params["near"]
        far = params["far"]
        self.n_samples = params["n_samples"]
        self.step_size = (far - near) / self.n_samples

        self.coordinate_encoder = PositionalEncoder(L=L_coords, input_dim=3)
        self.direction_encoder = PositionalEncoder(L=L_direction, input_dim=3)
        self.coordinate_encoding_dim = self.coordinate_encoder.get_output_dim()
        self.direction_encoding_dim = self.direction_encoder.get_output_dim()


        print("Coordinate Encoding Dim: ", self.coordinate_encoder.get_output_dim())
        print("Direction Encoding Dim: ", self.direction_encoder.get_output_dim())

        self.first_half_layers = nn.ModuleList()
        self.second_half_layers = nn.ModuleList()
        self.middle_idx = n_layers // 2

        current_dim = self.coordinate_encoding_dim

        for i in range(self.middle_idx):
            self.first_half_layers.append(nn.Linear(current_dim, width))
            current_dim = width

        # Handle the concatenation
        current_dim += self.coordinate_encoding_dim

        for i in range(n_layers - self.middle_idx):
            self.second_half_layers.append(nn.Linear(current_dim, width))
            current_dim = width



        # Additional layers for predicting density and rgb values seperately
        self.density_layer = nn.Linear(current_dim, 1)
        self.rgb_layer_1 = nn.Linear(width, width)
        self.rgb_layer_2 = nn.Linear(width + self.direction_encoding_dim, width // 2)
        self.rgb_layer_3 = nn.Linear(width // 2, 3)

        
    # Will return the predicted density and predicted rgb value
    def forward(self, x_coords, x_dirs):
        """
        x_coords : (batch_size, 3)
        x_dirs : (batch_size, 3)

        Returns : density (batch_size, 1) and rgb values (batch_size, 3)
        """
        batch_size = x_coords.shape[0]


        x_coords_encoded = self.coordinate_encoder(x_coords)
        x_dirs_encoded = self.direction_encoder(x_dirs)


        # Feeding in the coordinates through all the linear layers
        x_coords_output = x_coords_encoded
        for idx, layer in enumerate(self.first_half_layers):
            x_coords_output = F.relu(layer(x_coords_output))
        x_coords_output = torch.concat([x_coords_output, x_coords_encoded], dim=1)
        for idx, layer in enumerate(self.second_half_layers):
            if idx == len(self.second_half_layers) - 1:
                x_coords_output = layer(x_coords_output) # we dont want to apply relu to the last layer
            else:
                x_coords_output = F.relu(layer(x_coords_output))



        # Feeding for density
        density = F.relu(self.density_layer(x_coords_output))


        # Feeding for rgb
        rgb = self.rgb_layer_1(x_coords_output)
        rgb = torch.concat([rgb, x_dirs_encoded], dim=1)

        rgb = F.relu(self.rgb_layer_2(rgb))
        rgb = F.sigmoid(self.rgb_layer_3(rgb))

        return density, rgb



        





