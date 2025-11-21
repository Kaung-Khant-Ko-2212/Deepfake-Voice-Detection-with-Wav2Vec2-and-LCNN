# model.py
import torch.nn as nn
import torch.nn.functional as F
import torch

class LCNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(LCNN, self).__init__()
        # Simple feed-forward network to classify Wave2Vec2 embeddings
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.3) # Increased dropout for regularization
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    # Simple test for LCNN model
    input_dim = 768 # Example Wave2Vec2 hidden size
    model = LCNN(input_dim=input_dim)
    print(model)

    # Dummy input
    dummy_input = torch.randn(4, input_dim) # Batch size 4, input_dim features
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be [4, 2] for 2 classes