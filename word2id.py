# _*_coding:utf-8_*_

import collections
import pprint


class Word2Id:
    def __init__(self):
        """
        始端文字ID: 0
        終端文字ID: 1
        として追加する
        """
        self.word2id_dict = collections.defaultdict(lambda: len(self.word2id_dict))
        self.word2id_dict["BOF"] = 0
        self.word2id_dict["EOF"] = 1

    def __getitem__(self, key):
        return self.word2id_dict[key]

    def search_word_by(self, word_id):
        word2id_dict = dict(self.word2id_dict)

        try:
            return word2id_dict.keys()[word2id_dict.values().index(word_id)]
        except ValueError:
            return None

    def convert_id_list(self, tokens):
        # 最初と最後にBOFとEOFを追加しておく
        id_list = [0, ] # BOF追加

        for token in tokens.split(" "):
            id_list.append(self.word2id_dict[token])

        id_list.append(1) # EOF追加

        return id_list

    def show_dict(self):
        pprint.pprint(self.word2id_dict)
