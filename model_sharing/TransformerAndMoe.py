"""
Defines the Transformer model architecture and training process for time series anomaly detection. 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, max_len=500, node_count=5):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.node_encoding = nn.Embedding(node_count, d_model)
        self.node_count = node_count
        self.max_len = max_len


    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device
        node_indices = torch.arange(self.node_count, device=device).repeat_interleave(self.max_len)[:seq_len]
        node_positional_encoding = self.node_encoding(node_indices).unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + node_positional_encoding
        return x


# Mixture of Experts (MoE) Layer
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts=4, k=2):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        gate_scores = torch.softmax(self.gate(x), dim=-1)
        topk_indices = torch.topk(gate_scores, self.k, dim=-1).indices

        # Initialize output tensor
        out = torch.zeros_like(x)
        for i in range(self.k):
            expert_weights = torch.gather(gate_scores, -1, topk_indices[:, :, i].unsqueeze(-1))
            expert_output = sum(expert_weights[:, :, :] * F.relu(self.experts[j](x)) for j in range(self.num_experts))

            out += expert_output

        return out
    

# Custom Transformer Encoder Layer
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_experts=4, k=2):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.moe_layer = MoELayer(d_model, num_experts=num_experts, k=k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Self-attention layer
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # MoE layer
        src2 = self.moe_layer(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# Transformer Model Definition
class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, dim_feedforward, device, num_experts=4, k=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        self.pos_encoder = PositionalEncoding(dim_feedforward, device, node_count=5)
        encoder_layers = CustomTransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward, num_experts=num_experts, k=k)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(dim_feedforward, input_dim)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(output)
        return output


def model_fit(data, epoch_size, nhead=3, num_layers=3, num_experts=3, k=1):
    """
    Trains a Transformer model with weighted loss.

    Creates a PyTorch DataLoader from the input data, defines a Transformer model with specified parameters,
    and trains the model using Adam optimizer and MSE loss. The loss is weighted by the values in the last column
    of the input data.

    Args:
        data (np.ndarray): Input data for training. Should be a 2D array where the last column contains weights.
        epoch_size (int): Number of epochs to train the model.
        nhead (int, optional): Number of attention heads in the Transformer model. Defaults to 3.
        num_layers (int, optional): Number of encoder layers in the Transformer model. Defaults to 3.
        num_experts (int, optional): Number of experts in the MoE layer. Defaults to 3.
        k (int, optional): Number of top experts to select in the MoE layer. Defaults to 1.

    Returns:
        tuple: A tuple containing the trained Transformer model and a list of training losses.
    """
    train_dataset = CustomDataset(data)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=False)
    input_dim = data.shape[1] - 1
    dim_feedforward = (input_dim // nhead +1) * nhead if input_dim % nhead ==1 else  input_dim  # Must be a multiple of nhead
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(input_dim=input_dim, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, device=device, num_experts=num_experts, k=k).to(device)

    # train
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses = []

    for _ in tqdm(range(epoch_size)):
        for inputs in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            for start in range(0, inputs.size(0), model.pos_encoder.max_len):
                end = min(start + model.pos_encoder.max_len, inputs.size(0))
                batch_inputs = inputs[start:end, :-1].unsqueeze(1)
                weights = inputs[start:end, -1]

                outputs = model(batch_inputs)
                loss = criterion(outputs.squeeze(1), batch_inputs.squeeze(1))
                weighted_loss = (loss * weights).mean()
                weighted_loss.backward() 
                optimizer.step()
                train_losses.append(weighted_loss.item())
                
    return model, train_losses


def model_predict(model, predict_data):
    """
    Uses a trained Transformer model to reconstruct input data and returns the reconstructed and original data.

    Creates a PyTorch DataLoader from the input `predict_data`, feeds the data to the trained `model`
    to generate reconstructed data, and returns both the reconstructed and original data as lists.

    Args:
        model (TransformerModel): The trained Transformer model.
        predict_data (np.ndarray): The data to be reconstructed.

    Returns:
        tuple: A tuple containing two lists:
            - reconstructed: The reconstructed data from the model.
            - original: The original input data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = CustomDataset(predict_data)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    reconstructed = []
    original = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            for start in range(0, inputs.size(0), model.pos_encoder.max_len):
                end = min(start + model.pos_encoder.max_len, inputs.size(0))
                batch_inputs = inputs[start:end].unsqueeze(1)
                outputs = model(batch_inputs)
                reconstructed.extend(outputs.squeeze(1).cpu().numpy())
                original.extend(batch_inputs.squeeze(1).cpu().numpy())

    return reconstructed, original
