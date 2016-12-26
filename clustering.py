# _*_coding:utf-8_*_

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import argparse


def getArgs():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-l", "--label",
        dest="label_num",
        required=True,
        type=int,
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = getArgs()

    with open("model/model.vec", "r") as model:
        vector = model.readlines()

        col_num = None
        row_num = None

        data = []
        index_of_token = {}
        index = 0

        for line in vector:
            elements = line.split(" ")

            if col_num is None and row_num is None:
                row_num = elements[0]
                col_num = elements[1]
                continue

            elements.pop()
            token = elements.pop(0)

            index_of_token[token] = index
            index += 1

            data.append(elements)

        vector_array = np.array(data)

        pred = KMeans(n_clusters=args.label_num).fit_predict(vector_array)

        labels = pred[:31]

        for token, index in sorted(index_of_token.items(), key=lambda x: x[1]):
            print("{}: {} -> {}".format(index, token, labels[index]))

            if index == 30:
                break
