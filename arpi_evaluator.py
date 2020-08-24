"""
This script evaluates a candidate clustering given a reference.
"""
import os

from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd

NO_CLUSTER_LABEL = -1


def evaluate_recurrent_defects(ref_df: pd.DataFrame, predictions, remove_ata_zero_section=True,
                               remove_invalid_clusters=True):
    """
    Uses sklearn's Adjusted Rand Index, homogeneity, completeness and v-measure
    to evaluate the clustering predictions.

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html

    :param ref_df: The reference dataframe.
    :param predictions: The predictions. Their format is an iterable collection of sets of defect labels belonging to
                        the same cluster, i.e.
                        [{'C-6414274-1', 'L-5245081-1'}, {'C-6414294-1', 'C-6414295-1', 'C-6414296-1'}, ...]
                        Clusters containing a single element are ignored during evaluation.
    :param remove_ata_zero_section: Remove from the reference all clusters for which the ATA section is 0 (recommended)
    :param remove_invalid_clusters: Remove clusters which were invalidated by humans (recommended).
    :return: A dict with the following keys
        ari_score - Adjusted Rand Index, similarity score between -1.0 and 1.0. Random labelings have an ARI close to 0.
                                         1.0 stands for perfect match.
        homogeneity - A clustering result satisfies homogeneity if all of its predicted clusters contain only data
                      points that are clustered in the reference.
        completeness - A clustering result satisfies completeness if all the data points that are members of the
                       same reference cluster are found in the same predicted cluster.
        v_measure - harmonic mean of homogeneity and completeness
        pred_clusters - a list of predicted cluster labels, useful for debug
        ref_clusters - a list of reference cluster labels, useful for debug
        remove_ata_zero_section - copy of argument remove_ata_zero_section for this function
        nb_ref_clusters: nb of clusters in the reference
        n_pred_clusters: nb of cluster predicted
    """

    filled_df = ref_df.recurrent.fillna(NO_CLUSTER_LABEL)  # when there is no recurrent id, define as not clustered

    if remove_ata_zero_section:
        filled_df.where(ref_df.section == 0, NO_CLUSTER_LABEL, inplace=True)

    if remove_invalid_clusters:
        valid_cluster_ids = get_valid_cluster_ids()
        filled_df = filled_df.apply(lambda clus_id: clus_id if clus_id in valid_cluster_ids else NO_CLUSTER_LABEL)

    # remove clusters with a single member, which are not clusters at all
    duplicate_df = filled_df.duplicated(keep=False)
    filled_df.where(duplicate_df, NO_CLUSTER_LABEL, inplace=True)
    ref_clusters = filled_df

    # convert cluster assignments from the predictions in the same order as those from the ref
    pred_clusters = convert_cluster_labels_to_seq(ref_df, predictions)

    # evaluate
    homogeneity, completeness, v_measure_score = homogeneity_completeness_v_measure(ref_clusters, pred_clusters)
    ari_score = adjusted_rand_score(ref_clusters, pred_clusters)

    return {'ari_score': ari_score, 'homogeneity': homogeneity,
            'completeness': completeness, 'v_measure': v_measure_score,
            'pred_clusters': pred_clusters, 'ref_clusters': ref_clusters,
            'remove_ata_zero_section': remove_ata_zero_section,
            'nb_ref_clusters': ref_clusters.nunique() - 1, 'nb_pred_clusters': len(set(pred_clusters)) }


def convert_cluster_labels_to_seq(ref_df: pd.DataFrame, predictions):
    """Convert the predictions in a format usable by adjusted_rand_score. Returns list."""

    label_to_cluster_name = {}
    for i, cluster in enumerate(predictions):
        if len(cluster) > 1:  # we only keep clusters whose size is > 1
            for label in cluster:
                assert label in ref_df.index, f"Invalid cluster label {label}, not found in reference dataframe."
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
    print("id\tpred_label\tref_label", file=fout)
    for id, pred, ref in zip(defect_df.index, debug_info['pred_clusters'], debug_info['ref_clusters']):
        print(f"{id}\t{str(pred)}\t{str(ref)}", file=fout)


def get_valid_cluster_ids():
    result = None
    valid_id_file = os.path.join(os.path.dirname(__file__), 'small_resources', 'valid_cluster_ids.txt')
    with open(valid_id_file, 'rt', encoding='utf-8') as fin:
        result = set([int(x.strip()) for x in fin.readlines()])
    return result


def relabel_ata(df):
    return None