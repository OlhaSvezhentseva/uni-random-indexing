# Olha Svezhentseva
# 10.03.2023

from typing import List

from nltk.corpus import brown, wordnet
from nltk.stem import WordNetLemmatizer
from .check_token import is_valid_token

LEMMATIZER = WordNetLemmatizer()

WORDNET_MAP = {
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "J": wordnet.ADJ,
    "R": wordnet.ADV
}


class CleanDocument:
    """A class to represent a clean document."""
    def __init__(self, file_id: int):
        self.id = file_id
        self.lemmas = self._extract_lemmas()

    def _extract_lemmas(self) -> List:
        """This method collects lemmas of the words."""
        lemmas = []
        for token, tag in brown.tagged_words(self.id):
            token = token.lower()
            if is_valid_token(token):
                lemma = self._lemmatize(token, tag)
                lemmas.append(lemma)
        return lemmas

    @staticmethod
    def _lemmatize(token: str, tag: str) -> str:
        """This method determines lemma of a word."""
        tag = tag[0]
        if tag in WORDNET_MAP:
            tag = WORDNET_MAP[tag]
            lemma = LEMMATIZER.lemmatize(token, tag)
        else:
            lemma = LEMMATIZER.lemmatize(token)
        return lemma
