# Olha Svezhentseva
# 10.03.2023

from nltk.corpus import brown
import argparse
from typing import Dict

from .. import utils
from .clean_document import CleanDocument


def get_clean_documents(source: str = "brown") -> Dict:
    """The function gets clean documents and saves them in the dict."""
    if source == "brown":
        documents = {}
        for file_id in brown.fileids():
            document = CleanDocument(file_id)
            documents[file_id] = document.lemmas
        return documents


# clean_documents = get_clean_documents(source="brown")
# utils.save_data(clean_documents, "clean_documents_new5.pkl")
# clean_documents = utils.load_data("clean_documents_new5.pkl")
# print(clean_documents["ca10"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--corpus", nargs='?', default="brown")
    args = parser.parse_args()

    clean_documents = get_clean_documents(args.corpus)
    utils.save_data(clean_documents, args.file)
