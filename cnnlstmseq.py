import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMseq(nn.Module):
    def __init__(self, ebd, args):
        super(CNNLSTMseq, self).__init__()
        self.args = args
        self.ebd = ebd
        self.input_dim = self.ebd.embedding_dim

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.input_dim, out_channels=args.cnn_num_filters, kernel_size=K)
            for K in args.cnn_filter_sizes
        ])

        # LSTM layer
        self.hidden_dim = self.ebd.embedding_dim
        self.lstm = nn.LSTM(self.ebd.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True)

    def forward(self, data, weights=None):
        # Apply word embedding
        ebd = self.ebd(data["Utterance"], weights)

        # Apply convolutional layers
        conv_outputs = [F.relu(conv(ebd.permute(0, 2, 1))) for conv in self.convs]
        
        # Max pooling over time
        pooled_outputs = [F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in conv_outputs]

        # Concatenate pooled outputs
        combined = torch.cat(pooled_outputs, 1)

        # LSTM layer
        lstm_out, _ = self.lstm(combined.unsqueeze(1))

        # Squeeze and return
        return lstm_out.squeeze(1)
