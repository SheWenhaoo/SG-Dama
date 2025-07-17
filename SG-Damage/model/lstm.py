import torch
import torch.nn as nn

class LSTM200(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(LSTM200, self).__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x=x.unsqueeze(-1)
        batch_size = x.size(0)  
        _, (hidden, cell) = self.encoder_lstm(x)
        hidden_repeated = hidden.permute(1, 0, 2)  
        hidden_repeated = hidden_repeated.repeat(1, 200, 1)  
        decoder_output, _ = self.decoder_lstm(hidden_repeated)
        output = self.fc(decoder_output)
        return output.squeeze(-1)
    

    


