# _*_coding:utf-8_*_

from word2id import Word2Id

from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import math
import numpy as np
import argparse

VOCAB_SIZE = 300
HIDDEN_SIZE = 50

def getArgs():
    """
    コマンド引数をパースします
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-f", "--input",
        dest="input_file",
        type=argparse.FileType("r"),
        help="input filename"
    )

    return parser.parse_args()


class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__(
            embed=L.EmbedID(VOCAB_SIZE, HIDDEN_SIZE),  # word embedding
            mid=L.LSTM(HIDDEN_SIZE, HIDDEN_SIZE),  # the first LSTM layer
            out=L.Linear(HIDDEN_SIZE, VOCAB_SIZE),  # the feed-forward output layer
        )

    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, cur_word):
        # Given the current word ID, predict the next word.
        x = self.embed(cur_word)
        h = self.mid(x)
        y = self.out(h)
        return y


if __name__ == "__main__":
    args = getArgs()

    tokens_list = args.input_file.readlines()

    rnn = RNN()
    model = L.Classifier(rnn)
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    rnn.reset_state()

    def compute_loss(x_list):
        # 損失関数
        loss = 0
        for cur_word, next_word in zip(x_list, x_list[1:]):
            # one-hotベクトルの生成
            cur_word_vec = np.zeros(VOCAB_SIZE, dtype=np.int32)
            cur_word_vec[cur_word] = 1
            next_word_vec = np.zeros(VOCAB_SIZE, dtype=np.int32)
            next_word_vec[cur_word] = 1
            loss += model(cur_word_vec, next_word_vec)
        return loss

    w2i = Word2Id()

    id_lists = []

    for tokens in tokens_list:
        id_lists.append(np.array(w2i.convert_id_list(tokens), dtype=np.int32))

    for id_list in id_lists:
        optimizer.update(compute_loss, id_list)

