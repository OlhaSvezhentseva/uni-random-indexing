# Olha Svezhentseva
# 10.03.2023

import string

import spacy
en = spacy.load('en_core_web_sm')
STOPWORDS = en.Defaults.stop_words


def is_punctuation(token: str) -> bool:
    if token[0] in string.punctuation:
        return True
    return False


def is_stopword(token: str) -> bool:
    if token in STOPWORDS:
        return True
    return False


def is_number(word: str) -> bool:
    """The function checks that a word doesn't solely consist of numbers."""
    try:
        float(word)
    except ValueError:
        return False
    return True


def is_valid_token(token: str) -> bool:
    """The function checks that a token is a valid token."""
    if is_punctuation(token):
        return False
    if is_stopword(token):
        return False
    if is_number(token):
        return False
    return True
