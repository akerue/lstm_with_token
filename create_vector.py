# _*_coding:utf-8_*_

from gensim.models import word2vec
import argparse


def getArgs():
    """
    コマンド引数をパースします
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-l", "--letterbook",
        dest="letterbook_path",
        type=str,
        help="letterbook filepath"
    )

    parser.add_argument(
        "-m", "--model_path",
        dest="model_path",
        type=str,
        help="model filepath"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = getArgs()

    tokens = word2vec.Text8Corpus(args.letterbook_path)

    model = word2vec.Word2Vec(tokens, size=200, min_count=10, window=10)

    model.save(args.model_path)
