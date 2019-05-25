# -*- coding: utf-8 -*-
# @Author: RZH

from math import sqrt

locations = {
    'q': (0, 0), 'w': (1, 0), 'e': (2, 0), 'r': (3, 0), 't': (4, 0),
    'y': (5, 0), 'u': (6, 0), 'i': (7, 0), 'o': (8, 0), 'p': (9, 0),
    'a': (.4, 1), 's': (1.4, 1), 'd': (2.4, 1), 'f': (3.4, 1), 'g': (4.4, 1),
    'h': (5.4, 1), 'j': (6.4, 1), 'k': (7.4, 1), 'l': (8.4, 1),
    'z': (.8, 2), 'x': (1.8, 2), 'c': (2.8, 2), 'v': (3.8, 2), 'b': (4.8, 2),
    'n': (5.8, 2), 'm': (6.8, 2)
}


def keyboard_d(letter_1: str, letter_2: str) -> float:
    """
    calculate the distance of two letters on the standard keyboard
    :param letter_1: a letter, both lowercase and uppercase is supported
    :param letter_2: a letter, both lowercase and uppercase is supported
    :return: the distance
    """
    letter_1 = letter_1.lower()
    letter_2 = letter_2.lower()
    return sqrt(
        (locations[letter_1][0] - locations[letter_2][0]) ** 2 +
        (locations[letter_1][1] - locations[letter_2][1]) ** 2)


def edit_d(word: str, d: int = 1) -> set:
    """
    find all the words whose edit distance to `word` is `d`
    :param word: typo word
    :param d: edit distance
    :return: a set of filtered words
    """
    n = len(word)
    if d == 1:  # return all the 'words' whose edit distance to `word` is 1
        return set(
            [word[0:i] + word[i+1:] for i in range(n)] +   # deletion
            [word[0:i] + word[i+1] + word[i] + word[i+2:] for i in range(n-1)] +   # transposition
            [word[0:i] + c + word[i+1:] for i in range(n) for c in locations.keys()] +   # alteration
            [word[0:i] + c + word[i:] for i in range(n+1) for c in locations.keys()]  # insertion
        )
    if d == 2:  # return all the 'words' whose edit distance to `word` is 2
        return set(e2 for e1 in edit_d(word, d=1) for e2 in edit_d(e1, d=1))
    else:
        raise Exception('The edit distance `d` can only be 1 or 2.')


if __name__ == '__main__':
    print(keyboard_d('a', 'B'))
