import torch
import torch.nn as nn

###########################################################################

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

###########################################################################

# LSTM model for training
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


###########################################################################

# Transformer model for training
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.input_size = input_size
        

        self.input_projection = nn.Linear(input_size, hidden_size)
        

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_stacked_layers)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.input_projection(x)
        
        out = self.transformer(x)
        
        out = self.fc(out[:, -1, :])
        return out



###########################################################################


def model_switch(model):
    if model == "lstm":
        return LSTM(1, 4, 1)
    else:
        return TransformerModel(1, 4, 1)