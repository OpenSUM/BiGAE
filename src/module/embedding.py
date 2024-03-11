import os
import math

import torch
import torchtext


def get_pretrained_word_embedding(vocab, word_dim, pretrained_size="840B", pretrained_dim=300):
    pretrained_vec = torchtext.vocab.GloVe(name=pretrained_size, dim=300, cache=os.getenv("GLOVE"))
    vectors = torch.zeros((vocab.size(), pretrained_dim))
    oov_mask = torch.zeros(vocab.size())

    words = vocab.word_list()

    for i, word in enumerate(words):
        wid = pretrained_vec.stoi.get(word, None)
        if wid:
            vectors[i] = pretrained_vec.vectors[wid, :word_dim]
        else:
            oov_mask[i] = 1

    avg_vec = vectors.sum(0) / (1 - oov_mask).sum()

    # add unknown word vector using average embedding
    vectors[oov_mask.bool()] = avg_vec

    vectors = vectors.float()

    return vectors


def get_sinusoid_position_embedding(n_position, dim):
    """Sinusoid position encoding"""
    pe = torch.zeros(n_position, dim)

    positions = torch.arange(n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000) / dim))

    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)

    return pe
