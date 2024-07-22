# Olha Svezhentseva
# 10.03.2023

import os
import pickle
from typing import List, Optional

from .vectors import IndexVectors, ContextVectors


class Model:
    """A Class to represent a model."""
    def __init__(self):
        self.vectors = []
        self.parameters = {}

    def train(
            self,
            train_words: List,
            probs: List,
            vector_size: int,
            left: int,
            right: int) -> None:

        """This method trains a model by creating context vectors."""
        clean_words_flat = [token for document in train_words for token in document]
        # random indexing
        index_vectors = IndexVectors(set(clean_words_flat), [-1, 0, 1], probs, vector_size)
        context_vectors = ContextVectors(index_vectors.vectors, train_words, left, right)
        self.vectors = context_vectors.vectors
        self.parameters = {
            "probs": probs,
            "vector_size": vector_size,
            "left_window": left,
            "right_window": right
        }
        return

    def serialize(self, filename: str, base_dir: Optional[str] = "random_indexing") -> None:
        """This method serializes vectors."""
        if base_dir is not None:
            filename = os.path.join(base_dir, filename)
        dir_path = os.path.dirname(filename)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if len(self.vectors) > 0:
            with open(filename, mode="wb") as file:
                pickle.dump((self.vectors, self.parameters), file)
                print(f'Model saved in {filename}')
        else:
            print("Context vector were not created yet")
        return

    @classmethod
    def deserialize(cls, filename: str, base_dir: Optional[str] = "random_indexing") -> "Model":
        """This method loads saved vectors."""
        if base_dir is not None:
            filename = os.path.join(base_dir, filename)
        with open(filename, "rb") as file:
            vectors, params = pickle.load(file)
        print(f'Loading  model  from {filename}...')
        model = Model()
        model.vectors = vectors
        model.parameters = params
        return model
