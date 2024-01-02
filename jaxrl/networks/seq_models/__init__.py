from .recurrence import VanillaLSTM
from .gpt2_transformer import TransformerGPT


def get_seq_model(name):
    if name == "lstm":
        return VanillaLSTM
    elif name == "gpt":
        return TransformerGPT
    else:
        raise ValueError
