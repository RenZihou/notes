# -*- coding: utf-8 -*-
# @Author: RZH

from re import findall
from collections import defaultdict
from distance import edit_d


def words(text):
    return findall('[a-z]+', text.lower())


def train(features):
    model = defaultdict(lambda: 1)  # make sure that each word appears at least one times
    for feature in features:
        model[feature] += 1
    return model


WORDS = train(words(open('text.txt').read()))  # import the corpus and train the model


def known(words_: [list, set]) -> set:
    """
    return the correct words in `words_`
    :param words_: a set of words
    :return: correct words
    """
    return set(w for w in words_ if w in WORDS)


def correct(word):
    word = word.lower()
    candidates = known([word]) or known(edit_d(word, d=1)) or known(edit_d(word, d=2)) or [word]
    return max(candidates, key=lambda w: WORDS[w])


if __name__ == '__main__':
    print(correct('worls'))
