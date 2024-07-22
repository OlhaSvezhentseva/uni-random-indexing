# Olha Svezhentseva
# 10.03.2023

from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from typing import List, Dict, Tuple


class SimilarityCalculator:
    """A class to calculate similarity between 2 data sets."""

    def _compute_cosine_similarity\
            (
            self,
            annotation_data: List,
            context_vectors: Dict
            ) -> List:
        """The method computes cosine similarity between 2 vectors."""
        similarity_scores = []
        for word_pair in annotation_data:
            if word_pair[0] in context_vectors.keys() \
                    and word_pair[1] in context_vectors.keys():

                cosine_sim = cosine_similarity(
                    [context_vectors[word_pair[0]]],
                    [context_vectors[word_pair[1]]]
                )
                similarity_scores.append(cosine_sim[0][0])
            else:
                similarity_scores.append("None")
        return similarity_scores

    def _filter_pairs(
            self,
            gold_standard: List,
            cosine_sim: List)\
            -> Tuple[List, List]:
        """
        This method filters annotated word pairs.
        """
        gold_standard_found = []
        cosine_sim_found = []
        for index in range(0, len(gold_standard)):
            if cosine_sim[index] != "None":
                # words are also found in context vectors
                gold_standard_found.append(gold_standard[index][2])
                cosine_sim_found.append(cosine_sim[index])
        return gold_standard_found, cosine_sim_found

    def compute_spearman_coefficient(self,
                                     ratings: List,
                                     context_vectors: Dict
                                     ) -> float:
        """This method computes spearman coefficient between 2 rankings."""
        cosine_similarity_scores = \
            self._compute_cosine_similarity(ratings, context_vectors)
        human_ratings, model_predictions = \
            self._filter_pairs(ratings, cosine_similarity_scores)
        print(f"Found {len(human_ratings)} pairs"
              f" from {len(cosine_similarity_scores)}.")
        # why expected type different now??
        result = stats.spearmanr(human_ratings, model_predictions)
        return result[0]


# print(np.mean([abs(score) for score in model]))
# print(np.mean([abs(float(score)) for score in human_ratings]))

# print(human_ratings[:20], model[:20])

# print(compute_spearman_coeff(human_ratings, model))