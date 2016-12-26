#_*_coding:utf-8_*_

from ngram import NGram

import pickle
import argparse

NGRAM_FILE = "database/ngram.dat"

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

    parser.add_argument(
        "-n",
        dest="N",
        required=True,
        type=int,
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = getArgs()

    tokens_list = args.input_file.readlines()

    G = NGram(tokens_list, N=args.N)

    for l in list(G):
        print(l)

