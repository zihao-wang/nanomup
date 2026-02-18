"""TP4 nano: Word2Vec (CBOW) and MAML experiments."""

from .tp4_cbow import train_word2vec_nano
from .tp4_maml import train_maml_nano, evaluate_maml_nano

__all__ = [
    "train_word2vec_nano",
    "train_maml_nano",
    "evaluate_maml_nano",
]
