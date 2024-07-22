
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

RANGE = np.random.default_rng(seed=42)


class Vectors(ABC):

    @abstractmethod
    def _build_vectors(self, *kwargs):
        pass


class IndexVectors(Vectors):

    def __init__(self, words, variants, probs, size):
        self.vectors = self._build_vectors(words, variants, probs, size)

    def _build_vectors(self, words, variants, probs, size):
        """The function creates index vectors for words. """
        # "words" passed over as a set
        index_vectors = {}
        for word in words:
            index_vector = RANGE.choice(variants, size, p=probs)
            index_vectors[word] = index_vector
        return index_vectors


class ContextVectors(Vectors):
    def __init__(self, index_vectors, all_words, left, right):
        self.vectors = self._build_vectors(index_vectors, all_words, left, right)

    @staticmethod
    def create_context_for_word_0(index_vectors, words, right, size):
        """The function creates context for the 1st word. """
        context = np.zeros(size)
        for i in range(0, right):
            context += index_vectors[words[i]]
        return context

    def _build_vectors(self, index_vectors, all_words, left, right):
        # now on document level, so there is border between documents!
        values = list(index_vectors.values())
        size = len(values[0])
        context_vectors = {word: np.zeros(size) for word in index_vectors}
        for words in all_words:
            context = self.create_context_for_word_0(index_vectors, words, right, size)
            for current_index, token in enumerate(words):
                if current_index + right < len(words):
                    context += index_vectors[words[current_index + right]]
                if current_index - left > 0:
                    context -= index_vectors[words[current_index - left - 1]]
                # subtract index_vector of the word itself
                context_vectors[token] += (context - index_vectors[token])
        return context_vectors


class DocumentVectors(Vectors):
    def __init__(self,  context_vectors, file_ids, clean_documents):
        self.vectors = self._build_vectors(context_vectors, file_ids, clean_documents)

    @staticmethod
    def compute_document_vector(context_vectors, clean_document, size):
        document_vector = np.zeros(size)
        for token in clean_document:
            if token in context_vectors:
                document_vector += context_vectors[token]
            else:
                document_vector += np.zeros(size)
        return document_vector

    def _build_vectors(self, context_vectors, file_ids, clean_documents):
        # or dict comprehension and iter over keys?
        # normalizing?
        values = list(context_vectors.values())
        size = len(values[0])
        documents_vectors = []
        for file_id in file_ids:
            document_vector = self.compute_document_vector(context_vectors, clean_documents[file_id], size)
            documents_vectors.append(document_vector)
        return documents_vectors


def save_vectors(vectors, file_name, base_dir: Optional[str] = "random_indexing"):
    if base_dir is not None:
        file_name = os.path.join(base_dir, file_name)
    try:
        np.savetxt(f"{file_name}", vectors, delimiter=",")
    except TypeError:
        np.savetxt(f"{file_name}", vectors, delimiter=",", fmt='%s')
