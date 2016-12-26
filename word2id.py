# _*_coding:utf-8_*_

import collections


class Word2Id:
    def __init__(self):
        self.word2id_dict = collections.defaultdict(lambda: len(self.word2id_dict))

    def __getitem__(self, key):
        return self.word2id_dict[key]

    def search_word_by(self, word_id):
        word2id_dict = dict(self.word2id_dict)

        try:
            return word2id_dict.keys()[word2id_dict.values().index(word_id)]
        except ValueError:
            return None

    def convert_id_list(self, tokens):
        id_list = []

        for token in tokens:
            id_list.append(self.word2id_dict[token])

        return id_list
