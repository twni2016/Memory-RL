from .rnn_vanilla import RNN, LSTM, GRU
from .gpt2_vanilla import GPT2


SEQ_MODELS = {RNN.name: RNN, LSTM.name: LSTM, GRU.name: GRU, GPT2.name: GPT2}
