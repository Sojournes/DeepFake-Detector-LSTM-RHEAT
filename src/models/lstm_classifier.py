import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)       # (B, seq, hid*2)
        out     = out[:, -1]        # take last step
        return self.fc(out).squeeze(1)
