import dataclasses
from typing import Any, Dict, List, Optional

from clairvoyance2.data import DEFAULT_PADDING_INDICATOR


@dataclasses.dataclass
class Seq2seqParams:
    # Encoder:
    encoder_rnn_type: str = "LSTM"
    """Encoder RNN type. Available options: ``"LSTM"``, ``"GRU"``, ``"RNN"``."""
    encoder_hidden_size: int = 100
    """Encoder hidden size."""
    encoder_num_layers: int = 1
    """Encoder number of layers."""
    encoder_bias: bool = True
    """Encoder bias enabled/disabled."""
    encoder_dropout: float = 0.0
    """Encoder dropout values (0.0 to 1.0)"""
    encoder_bidirectional: bool = False
    """Encoder bidirectional enabled/disabled."""
    encoder_nonlinearity: Optional[str] = None
    """Encoder nonlinearity."""
    encoder_proj_size: Optional[int] = None
    """Encoder projection size."""
    # Decoder:
    decoder_rnn_type: str = "LSTM"
    """Decoder RNN type. Available options: ``"LSTM"``, ``"GRU"``, ``"RNN"``."""
    decoder_hidden_size: int = 100
    """Decoder hidden size."""
    decoder_num_layers: int = 1
    """Decoder number of layers."""
    decoder_bias: bool = True
    """Decoder bias enabled/disabled."""
    decoder_dropout: float = 0.0
    """Decoder dropout values (0.0 to 1.0)"""
    decoder_bidirectional: bool = False
    """Decoder bidirectional enabled/disabled."""
    decoder_nonlinearity: Optional[str] = None
    """Decoder nonlinearity."""
    decoder_proj_size: Optional[int] = None
    """Decoder projection size."""
    # Adapter FF NN:
    adapter_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [50])
    """Adapter hidden dimensions, as a list of integers corresponding to different layers."""
    adapter_out_activation: Optional[str] = "Tanh"
    """Adapter output activation function."""
    # Predictor FF NN:
    predictor_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [])
    """Predictor hidden dimensions, as a list of integers corresponding to different layers."""
    predictor_out_activation: Optional[str] = None
    """Predictor output activation function."""
    # Misc:
    max_len: Optional[int] = None
    """Maximum length of the input sequence."""
    optimizer_str: str = "Adam"
    """Optimizer type."""
    optimizer_kwargs: Dict[str, Any] = dataclasses.field(default_factory=lambda: dict(lr=0.01, weight_decay=1e-5))
    """Optimizer kwargs."""
    batch_size: int = 32
    """Batch size."""
    epochs: int = 100
    """Number of epochs."""
    padding_indicator: float = DEFAULT_PADDING_INDICATOR
    """Padding indicator value."""
