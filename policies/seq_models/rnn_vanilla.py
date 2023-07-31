import torch.nn as nn
import torchkit.pytorch_utils as ptu


class RNN(nn.Module):
    name = "rnn"
    rnn_class = nn.RNN

    def __init__(self, input_size, hidden_size, n_layer, **kwargs):
        super().__init__()
        self.model = self.rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layer,
            batch_first=False,
            bias=True,
        )
        self.hidden_size = hidden_size
        self.num_layers = n_layer

        self._initialize()

    def _initialize(self):
        # default RNN initialization is uniform, not recommended
        # https://smerity.com/articles/2016/orthogonal_init.html orthogonal has eigenvalue = 1
        # to prevent grad explosion or vanishing
        for name, param in self.model.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def forward(self, inputs, h_0):
        """
        inputs: (T, B, input_dim)
        h_0: (num_layers=1, B, hidden_size)
        return
        output: (T, B, hidden_size)
        h_n: (num_layers=1, B, hidden_size), only used in inference
        """
        output, h_n = self.model(inputs, h_0)
        return output, h_n

    def get_zero_internal_state(self, batch_size=1):
        return ptu.zeros((self.num_layers, batch_size, self.hidden_size)).float()


class GRU(RNN):
    name = "gru"
    rnn_class = nn.GRU


class LSTM(RNN):
    name = "lstm"
    rnn_class = nn.LSTM

    def get_zero_internal_state(self, batch_size=1):
        # for LSTM, current_internal_state also includes cell state
        hidden_state = ptu.zeros(
            (self.num_layers, batch_size, self.hidden_size)
        ).float()
        cell_state = ptu.zeros((self.num_layers, batch_size, self.hidden_size)).float()
        return hidden_state, cell_state
