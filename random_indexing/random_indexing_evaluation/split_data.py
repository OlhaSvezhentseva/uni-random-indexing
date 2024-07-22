# Olha Svezhentseva
# 10.03.2023

from sklearn.model_selection import train_test_split
from nltk.corpus import brown
import argparse
from typing import Dict, Tuple, List
from .. import utils


SEED = 42


def prepare_preprocessed_data(clean_documents: Dict) -> Tuple[List, List]:
    """The function extracts file ids and corresponding labels."""
    file_ids = []
    labels = []
    for key in clean_documents.keys():
        file_ids.append(key)
        labels.append(brown.categories(fileids=key)[0])
    return file_ids, labels


def split_data(
        documents: List, labels: List) \
        -> Tuple[List, List, List, List, List, List]:

    doc_train, documents_test, doc_genre_train, genres_test = \
        train_test_split(
            documents,
            labels,
            # indexes,
            test_size=.2,
            stratify=labels,
            random_state=SEED
        )

    # get train and dev data
    documents_train, documents_dev, genres_train, genres_dev = \
        train_test_split(
            doc_train,
            doc_genre_train,
            # indexes,
            test_size=.125,
            # with stratify -> 14, without 15 classes present
            # stratify=doc_genre_train,
            random_state=SEED
        )
    # print(len(set(genres_train)), len(set(genres_dev)), len(set(genres_dev)))
    return documents_train, documents_dev, \
           genres_train, genres_dev, \
           documents_test, genres_test


def create_train_dev_test_sets(clean_data_file):
    """The method splits the data into train, dev and test sets."""
    clean_data = utils.load_data(clean_data_file)
    file_ids, labels = prepare_preprocessed_data(clean_data)

    # split data
    file_ids_train, file_ids_dev, \
    genres_train, genres_dev, \
    file_ids_test, genres_test = split_data(file_ids, labels)
    # save data
    utils.save_data((file_ids_train, genres_train), "train_data.pkl")
    utils.save_data((file_ids_dev, genres_dev,), "dev_data.pkl")
    utils.save_data((file_ids_test, genres_test,), "test_data.pkl")
    print(file_ids_train, file_ids_dev, file_ids_test)
    return

# if __name__ == "__main__":
#     clean_documents_file = "clean_documents_new.pkl"
#     print(create_train_dev_test_sets(clean_documents_file))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    clean_documents = create_train_dev_test_sets(args.file)
