# Olha Svezhentseva
# 10.03.2023

import os
import argparse
from typing import List


from .model import Model
from .. import utils



# variants = [-1, 0, 1]
# clean_documents_file = "clean_documents_new.pkl"
# PROBS = [0.1, 0.8, 0.1]
# VECTOR_SIZE = 10
# LEFT = 1
# RIGHT = 2


def build_model(
        clean_data_file: str,
        probs: List,
        vector_size: int,
        left: int,
        right: int,
        directory: str) -> None:
    """The method loads train data and trains the model on it."""
    clean_data = utils.load_data(clean_data_file)
    file_ids_train = utils.load_data("train_data.pkl")[0]
    train_words_clean = [clean_data[file_id] for file_id in file_ids_train]

    # create and train model
    model = Model()
    model.train(train_words_clean, probs, vector_size, left, right)
    # save model
    filename = directory + ".pkl"
    parent_dir = "models"
    path = os.path.join(parent_dir, directory)

    new_file = os.path.join(path, filename)
    model.serialize(f"{new_file}")


# clean_documents_file = "clean_documents_new.pkl"
# build_model(clean_documents_file, PROBS, VECTOR_SIZE, LEFT, RIGHT, "model_6")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--probs", nargs='?', default=[0.1, 0.8, 0.1])
    parser.add_argument("--size", type=int)
    parser.add_argument("--l", type=int)
    parser.add_argument("--r", type=int)
    parser.add_argument("--dir", type=str)

    args = parser.parse_args()
    clean_documents = build_model(args.file, args.probs, args.size, args.l, args.r, args.dir)