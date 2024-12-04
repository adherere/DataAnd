import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Dataset
class SpectralDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# CustomEmbedding
class CustomEmbedding(nn.Module):
    def __init__(self, input_dim, model_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, model_dim)

    def forward(self, x):
        return self.linear(x)



# --------------Transformermodel---------------------------------
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.embedding = CustomEmbedding(input_dim, model_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.regressor = nn.Linear(model_dim, num_classes)


    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        output = self.regressor(x)
        return output
# --------------Transformermodel---------------------------------


