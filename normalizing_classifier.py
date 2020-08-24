"""Normalizing framework stub, used in a classification context."""
import argparse
import re
import numpy as np
import os
import pickle
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score

import arpi_evaluator


def main():
    # normalization possibilities, add your functions here
    NORMALIZATION_FUNCTIONS = {'none': lambda x: x, 'acro_replacement': replace_acros}

    # parse args
    parser = argparse.ArgumentParser("A sample program to test text normalization.")
    parser.add_argument("input_file", help="A pickle input file, e.g. aircan-data-split-clean.pkl.")
    parser.add_argument("normalization_method", help="Normalization method.", choices=NORMALIZATION_FUNCTIONS.keys())
    parser.add_argument('--reliable', '-r', action='store_true', help='Use relabeled reliable ATA chapter/sections only.', default=True)

    args = parser.parse_args()

    print("Loading...")
    with open(args.input_file, 'rb') as fin:
        [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)

    # normalize text
    if args.normalization_method in NORMALIZATION_FUNCTIONS:
        normalization_function = NORMALIZATION_FUNCTIONS[args.normalization_method]
    else:
        raise ValueError("Please add your normalization function in the dictionary NORMALIZATION_FUNCTIONS.")

    if args.reliable:
        print("Relabeling with reliable ATA chapters and sections...")
        nb_valids = 0
        for df in [defect_df_train, defect_df_dev, defect_df_test]:
            arpi_evaluator.relabel_ata(df)
            nb_valids += df['reliable_chapter'].count()

        print(f"Reliable labeling decreased corpus size from {len(defect_df_train) + len(defect_df_test) + len(defect_df_dev)} to {nb_valids}")

        for df in [defect_df_train, defect_df_dev, defect_df_test]:
            df.dropna(subset=['defect_description', 'reliable_chapter'], inplace=True, how='any')  # removes empty text and ata info
            df['label'] = df[['reliable_chapter', 'reliable_section']].apply(lambda data: f"{str(data['reliable_chapter'])}-{str(data['reliable_section'])}", axis=1)
            df['normalized_desc'] = df.defect_description.apply(normalization_function)

        train_df, dev_df, test_df = defect_df_train, defect_df_dev, defect_df_test
    else:  # we will be working with trax dataset
        # remove empty descriptions
        trax_df_clean = trax_df.dropna(subset=['defect_description'])
        # drop recurrent defects with section 0 (it is a catch-all section that indicates a certain sloppiness when labeling
        trax_df_clean = trax_df_clean[trax_df_clean.rec_sec != 0]
        # add a label made from the concat of the chapter and section -> chap-sec, this is what we want to predict
        trax_df_clean['label'] = trax_df_clean[['rec_ch', 'rec_sec']].apply(lambda data: f"{data['rec_ch']}-{data['rec_sec']}", axis=1)
        # normalize text
        trax_df_clean['normalized_desc'] = trax_df_clean.defect_description.apply(normalization_function)

        # split corpus
        train_df, dev_df, test_df = np.split(trax_df_clean.sample(frac=1, random_state=42),
                                            [int(.6 * len(trax_df_clean)), int(.8 * len(trax_df_clean))])

    print(f"Dataset split is: {len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test.")

    # let us try a little classifier based on tf-idf
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(train_df.normalized_desc.tolist()).toarray()
    labels = train_df.label
    model = LinearSVC(random_state=42)
    model.fit(features, labels)

    predictions = model.predict(tfidf.transform(test_df.normalized_desc.tolist()).toarray())

    precision = precision_score(test_df.label, predictions, average='micro')
    print(f"Precision is {str(precision)}")


__acro_map: dict = None
__acro_keys: set = None


def load_acro_map():
    global __acro_map, __acro_keys
    acronym_file = os.path.join(os.path.dirname(__file__), 'small_resources', 'acronyms_1.tsv')
    with open(acronym_file, 'rt', encoding='utf-8') as fin:
        lines = fin.readlines()

    __acro_map = dict()
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            __acro_map[parts[0].upper()] = parts[1].upper()

    __acro_keys = set(__acro_map.keys())


def replace_acros(text: str):
    assert type(text) == str, "Invalid type " + str(type(text)) + " of value " + str(text)

    if __acro_map is None:
        load_acro_map()

    toks = re.split(r'[\s\.,;/:\(\)-]', text)  # do not do this
    for i, tok in enumerate(toks):
        if tok in __acro_keys:
            toks[i] = __acro_map.get(tok)

    return ' '.join(toks)


if __name__ == '__main__':
    main()
