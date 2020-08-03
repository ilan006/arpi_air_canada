"""
This script evaluates a candidate clustering given a reference.
"""
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
import sys

NO_CLUSTER_LABEL = -1


def evaluate_recurrent_defects(ref_df: pd.DataFrame, predictions):
    """
    Uses sklearn's adjusted Rand Index
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    to evaluate the clustering predictions.

    This is another para.

    :param ref_df: The reference dataframe.
    :param predictions: The predictions. Their format is an iterable collection of sets of defect labels belonging to
                        the same cluster, i.e.
                        [{'C-6414274-1', 'L-5245081-1'}, {'C-6414294-1', 'C-6414295-1', 'C-6414296-1'}, ...]
    :return: A tuple:
        score from 0 (worst) to 1 (best)
        additional debug information dictionary
    """

    # extract cluster assignments from the reference
    ref_clusters = list(ref_df.recurrent.fillna(NO_CLUSTER_LABEL))

    # extract cluster assignments from the predictions in the same order as those from the ref
    pred_clusters = convert_cluster_labels_to_seq(ref_df, predictions)

    # evaluate
    score = adjusted_rand_score(ref_clusters, pred_clusters)
    return score, {'score': score, 'pred_cluster_list': pred_clusters}


def convert_cluster_labels_to_seq(ref_df: pd.DataFrame, predictions):
    """Convert the predictions in a format usable by adjusted_rand_score"""

    label_to_cluster_name = {}
    for i, cluster in enumerate(predictions):
        for label in cluster:
            label_to_cluster_name[label] = i

    result = [NO_CLUSTER_LABEL] * len(ref_df)

    for i, label in enumerate(ref_df.index):
        result[i] = label_to_cluster_name.get(label, NO_CLUSTER_LABEL)

    return result
