"""
My clustering efforts.
"""
import argparse
import re

import numpy as np
import os
import pandas as pd
import pickle
from pprint import pprint
import sys
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import arpi_evaluator

__NUM_TOKEN = 'numba'
__SPELLING_FULL = None


def main():
    # parse args
    parser = argparse.ArgumentParser("Aggron - a clustering program.")
    parser.add_argument("input_file", help="A pickle input file, e.g. aircan-data-split-clean.pkl.")
    parser.add_argument("working_dir", help="A directory to write intermediate results if necessary/possible.")
    parser.add_argument("output_file", help="An output file where evaluation details will be written.")
    parser.add_argument('--text_fields', help='Text field(s) to use, comma-sep.', default='defect_description')
    parser.add_argument('--norm_steps', help='Normalization steps, e.g. tokenize,spelling,num')
    parser.add_argument('--clustering_method', help='Clustering method', default='kmeans')
    args = parser.parse_args()

    lsa = True

    # read stuff
    with open(args.input_file, 'rb') as fin:
        [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)
        print(f"Read # samples: {len(defect_df_train)} train, {len(defect_df_test)} dev, {len(defect_df_test)} test.")

    # intermediate result directory check
    if not os.path.exists(args.working_dir):
        print("Creating " + args.working_dir)
        os.mkdir(args.working_dir)

    # create text content
    text_fields = ['defect_description'] if args.text_fields is None else args.text_fields.split(',')
    for df in [defect_df_train, defect_df_dev, defect_df_test]:
        df['text_content'] = df[text_fields].apply(lambda x: ' '.join(['' if pd.isnull(y) else y for y in x]).lower(), axis=1)

    # apply normalizations
    print("Normalizing.", file=sys.stderr)
    normalization_functions = {'identity': lambda x: x, 'spelling': norm_spelling, 'tokenize': norm_tokenize,
                               'num': norm_num}
    if args.norm_steps is not None:
        steps = [x.strip().lower() for x in args.norm_steps.split(',')]
        for df in [defect_df_train, defect_df_dev, defect_df_test]:
            for step in steps:
                if step == 'spelling':
                    load_full_spelling()
                df['text_content'] = df.text_content.apply(normalization_functions.get(step))

    # train vectorizer
    print("Tfidf.", file=sys.stderr)
    train_and_dev = pd.concat([defect_df_train, defect_df_dev], sort=True)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2),
                            stop_words='english', max_features=10000)
    vectorizer = tfidf.fit(train_and_dev.text_content.tolist())

    print("Clustering...", end=' ', flush=True)
    df_cluster_predictions = pd.DataFrame(index=defect_df_test.index)
    df_cluster_predictions['cluster'] = ''  # new column for cluster label

    svd = TruncatedSVD(100)  # 100 is better!
    normalizer = Normalizer(copy=False)  # svd does not yield normalized vectors
    lsa = make_pipeline(svd, normalizer)

    additional_dist_matrices = load_distance_matrices(defect_df_test, args.working_dir)

    grouped_by_ac = defect_df_test.groupby('ac')
    for name, ac_group in grouped_by_ac:
        print(f'{name} ({len(ac_group)})', end=', ', flush=True)
        X = vectorizer.transform(ac_group.text_content)

        if lsa:  # may have to be run on full corpus rather than on subset
            X = lsa.fit_transform(X)

        dist_matrix = pairwise_distances(X, None, metric='euclidean', n_jobs=-1)  # custom fun is very slow? 'euclidean' is fast and good

        clustering_model = AgglomerativeClustering(n_clusters=None, affinity='precomputed',  # None here and
                                                   distance_threshold=1.0, linkage='average')  # 1.0 for thresh
        clusters = clustering_model.fit_predict(dist_matrix)

        row_number = 0
        for index, _ in ac_group.iterrows():
            df_cluster_predictions.loc[index]['cluster'] = f'{name}-{str(clusters[row_number])}'
            row_number += 1

    # evaluation
    print("\nEvaluating.")
    predicted_cluster_list = convert_to_cluster_list(df_cluster_predictions)
    eval_debug_info = arpi_evaluator.evaluate_recurrent_defects(defect_df_test, predicted_cluster_list)
    pred_clusters, ref_clusters = eval_debug_info['pred_clusters'], eval_debug_info['ref_clusters']
    eval_debug_info['pred_clusters'] = eval_debug_info['ref_clusters'] = 'muted'
    pprint(eval_debug_info)

    print(f"\nDumping debug info in file {args.output_file}")
    eval_debug_info['pred_clusters'], eval_debug_info['ref_clusters'] = pred_clusters, ref_clusters
    with open(args.output_file, 'wt', encoding='utf-8') as fout:
        arpi_evaluator.dump_debug_info(defect_df_test, eval_debug_info, fout)


def convert_to_cluster_list(df_cluster_predictions):
    cluster_to_id = {}
    for index, row in df_cluster_predictions.iterrows():
        cur_cluster = row['cluster']
        cur_set = cluster_to_id.get(cur_cluster, set())
        if len(cur_set) == 0:
            cluster_to_id[cur_cluster] = cur_set
        cur_set.add(index)
    predicted_cluster_list = []
    for label_cluster in cluster_to_id.values():
        if len(label_cluster) > 1:
            predicted_cluster_list.append(label_cluster)
    return predicted_cluster_list


def norm_tokenize(text: str):
    return ' '.join(re.split(r'([\s\.,;/:\(\)"-])', text))


def norm_spelling(text: str):
    """Splits at whitespace"""
    toks = text.split()
    for i, tok in enumerate(toks):
        if tok in __SPELLING_FULL:
            toks[i] = __SPELLING_FULL.get(tok)

    return ' '.join(toks)


def norm_num(text: str):
    """Splits at whitespace, make sure to tokenize first then"""
    return re.sub(r'\b[0-9]+\b', __NUM_TOKEN, text)


def load_full_spelling():
    global __SPELLING_FULL

    if __SPELLING_FULL is None:
        __SPELLING_FULL = dict()

        spelling_file = os.path.join(os.path.dirname(__file__), 'small_resources', 'spelling_full.txt')
        with open(spelling_file, 'rt', encoding='utf-8') as fin:
            for line in fin.readlines():
                parts = line.lower().strip().split('\t')
                assert len(parts) == 3
                __SPELLING_FULL[parts[0]] = parts[1]  # I keep one's and two's

        print(f"Read spelling corrections with {len(__SPELLING_FULL)} entries.")


def compute_distance_matrix(df_view: pd.DataFrame, dist_matrix: str):
    if dist_matrix == 'ref':  # a reference to another defect in writing
        result = pairwise_distances(np.reshape(range(0, len(df_view)), (-1, 1)), n_jobs=-1, metric=distance_metric_ref,
                                    dataframe=df_view)
    else:
        raise ValueError("Invalid distance metric " + dist_matrix)

    return result


def load_distance_matrices(df: pd.DataFrame, working_dir: str):
    result = {}

    for dist_matrix in ['ref', 'time']:
        dist_file = os.path.join(working_dir, dist_matrix + '.pkl')
        if os.path.exists(dist_file):
            print("Loading feature " + dist_matrix)
            matrix = pickle.load(open(dist_file, 'rb'))
        else:
            # compute the distance matrix
            matrix = {}
            grouped_by_ac = df.groupby('ac')
            for name, ac_group in grouped_by_ac:
                matrix[name] = compute_distance_matrix(ac_group, dist_matrix)

        pickle.dump(matrix, open(dist_file, 'wb'))
        result[dist_matrix] = matrix

    return result


def distance_metric_ref(index1, index2, dataframe):
    """index1 is the index in the dataframe of the line"""
    print(f"{index1}-{index2}")
    return 1


if __name__ == '__main__':
    main()
