# Olha Svezhentseva
# 10.03.2023

import argparse
from typing import List, Dict

from .model import Model
from .similarity import SimilarityCalculator

SOURCE_INDICES = {
    "wordsim":{"start": 0, "first_word": 0, "second_word": 1, "score": 2},
    "simlex": {"start": 1, "first_word": 0, "second_word": 1, "score": 3}
                  }


def load_annotation_data(file: str, source: str) -> List:
    """The method loads annotated data from the file."""
    annotation = []
    with open(file) as file_in:
        for line in file_in.readlines()[SOURCE_INDICES[source]["start"]:]:
            line = line.strip()
            line = line.split()
            word_pair_score = [line[SOURCE_INDICES[source]["first_word"]],
                               line[SOURCE_INDICES[source]["second_word"]],
                               line[SOURCE_INDICES[source]["score"]]]
            annotation.append(word_pair_score)
    return annotation


def evaluate_ratings(
        context_vectors: Dict,
        model_info: Dict,
        annotated_data_file: str,
        source: str = 'wordsim'
                    ) -> str:
    """
    The method gives the spearman coefficient,
    based on the model and human rankings.
    """
    print(f"Model parameters: {model_info}")
    ratings = load_annotation_data(annotated_data_file, source)
    calculator = SimilarityCalculator()
    return f"Spearman coefficient for model predictions compared to {source}: " \
           f"{round(calculator.compute_spearman_coefficient(ratings, context_vectors), 2)}"


#
# annotation_path = "/Users/olhasvezhentseva/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt"
# # # annotation_path = "/Users/olhasvezhentseva/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt"
# # annotation_path = "/Users/olhasvezhentseva/SimLex-999/SimLex-999.txt"
#

# model_path = "models/model_6/model_6.pkl"
# model = Model.deserialize(model_path)
#
# # "wordsim" by deafualt vs "simlex"
# print(evaluate_ratings(model.vectors, model.parameters, annotation_path, source="wordsim"))
# # print(evaluate_ratings(model.vectors, annotation_path, source="simlex"))

# ratings global!
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ratings", type=str)
    parser.add_argument("--source", nargs='?', default="wordsim")

    args = parser.parse_args()
    model = Model.deserialize(args.model)
    print(evaluate_ratings(model.vectors,
                           model.parameters,
                           args.ratings,
                           args.source))