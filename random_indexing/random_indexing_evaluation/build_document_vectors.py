# Olha Svezhentseva
# 10.03.2023

import argparse

from .model import Model
from .vectors import DocumentVectors, save_vectors
from ..utils import load_data


def build_document_vectors(
        clean_data_file: str,
        model: "Model",
        train_set: str,
        dev_set: str,
        test_set: str) -> str:
    """
    This function builds document vectors
    for every set and saves them.
    """
    clean_data = load_data(clean_data_file)
    train_document_vectors = \
        DocumentVectors(model.vectors, train_set[0], clean_data)
    save_vectors(train_document_vectors.vectors, "vectors_train")
    train_document_genres = train_set[1]
    save_vectors(train_document_genres, "genres_train")

    dev_document_vectors = \
        DocumentVectors(model.vectors, dev_set[0], clean_data,)
    save_vectors(dev_document_vectors.vectors, "vectors_dev")
    dev_document_genres = dev_set[1]
    save_vectors(dev_document_genres, "genres_dev")

    test_document_vectors = \
        DocumentVectors(model.vectors, test_set[0], clean_data)
    save_vectors(test_document_vectors.vectors, "vectors_test")
    test_document_genres = test_set[1]
    save_vectors(test_document_genres, "genres_test")

    return "Train, test and dev sets were saved."



# train_data_file = load_data("train_data.pkl")
# dev_data_file = load_data("dev_data.pkl")
# test_data_file = load_data("test_data.pkl")
#
# clean_documents_file = "clean_documents_new.pkl"
# model_path = "models/model_6/model_6.pkl"
# model = Model.deserialize(model_path)
# print(build_document_vectors(clean_documents_file, model, train_data_file, dev_data_file, test_data_file))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--train", nargs='?', default="train_data.pkl")
    parser.add_argument("--dev", nargs='?', default="dev_data.pkl")
    parser.add_argument("--test", nargs='?', default="test_data.pkl")
    args = parser.parse_args()

    train_data = load_data(args.train)
    dev_data = load_data(args.dev)
    test_data = load_data(args.test)
    model = Model.deserialize(args.model)

    print(build_document_vectors(args.data,
                            model,
                           train_data,
                           dev_data,
                           test_data))