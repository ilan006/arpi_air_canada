"""
This script evaluates a candidate clustering given a reference.
"""
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd

NO_CLUSTER_LABEL = -1


def evaluate_recurrent_defects(ref_df: pd.DataFrame, predictions):
    """
    Uses sklearn's adjusted Rand Index
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    to evaluate the clustering predictions.

    :param ref_df: The reference dataframe.
    :param predictions: The predictions. Their format is an iterable collection of sets of defect labels belonging to
                        the same cluster, i.e.
                        [{'C-6414274-1', 'L-5245081-1'}, {'C-6414294-1', 'C-6414295-1', 'C-6414296-1'}, ...]
                        Clusters containing a single element are ignored during evaluation.
    :return: A tuple:
        ARI (adjusted rand index) score. Similarity score between -1.0 and 1.0. Random labelings have an ARI close to 0.
                                         1.0 stands for perfect match.
        additional debug information dictionary
    """

    # extract cluster assignments from the reference, and remove clusters with a single member, which are not clusters
    filled_df = ref_df.recurrent.fillna(NO_CLUSTER_LABEL)
    duplicate_df = filled_df.duplicated(keep=False)
    filled_df.where(duplicate_df, NO_CLUSTER_LABEL, inplace=True)
    ref_clusters = filled_df

    # extract cluster assignments from the predictions in the same order as those from the ref
    pred_clusters = convert_cluster_labels_to_seq(ref_df, predictions)

    # evaluate
    score = adjusted_rand_score(ref_clusters, pred_clusters)
    return score, {'ari_score': score, 'pred_clusters': pred_clusters, 'ref_clusters': ref_clusters}


def convert_cluster_labels_to_seq(ref_df: pd.DataFrame, predictions):
    """Convert the predictions in a format usable by adjusted_rand_score"""

    label_to_cluster_name = {}
    for i, cluster in enumerate(predictions):
        if len(cluster) > 1:  # we only keep clusters whose size is > 1
            for label in cluster:
                label_to_cluster_name[label] = i

    result = [NO_CLUSTER_LABEL] * len(ref_df)

    for i, label in enumerate(ref_df.index):
        result[i] = label_to_cluster_name.get(label, NO_CLUSTER_LABEL)

    return result


def dump_debug_info(defect_df: pd.DataFrame, debug_info, fout):
    """
    Dumps debug info in fout for analyzing results. Lines will have format:
    defect_label    predicted_cluster   reference_cluster

    :param defect_df: The defect dataframe used for prediction/evaluation.
    :param debug_info: The debug info returned by function evaluate_recurrent_defects
    :param fout: The stream onto we write the debug info.
    :return: Nothing.
    """
    for id, pred, ref in zip(defect_df.index, debug_info['pred_clusters'], debug_info['ref_clusters']):
        print(f"{id}\t{str(pred)}\t{str(ref)}", file=fout)
