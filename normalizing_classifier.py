"""Normalizing framework stub, used in a classification context."""
import argparse
import re
import numpy as np
import os
import pickle
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score


def main():
    # normalization possibilities, add your functions here
    NORMALIZATION_FUNCTIONS = {'none': lambda x: x, 'acro_replacement': replace_acros}

    # parse args
    parser = argparse.ArgumentParser("A sample program to test text normalization.")
    parser.add_argument("input_file", help="A pickle input file, e.g. aircan-data-split-clean.pkl.")
    parser.add_argument("normalization_method", help="Normalization method.", choices=NORMALIZATION_FUNCTIONS.keys())
    args = parser.parse_args()

    with open(args.input_file, 'rb') as fin:
        [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)

    # remove empty descriptions
    trax_df_clean = trax_df.dropna(subset=['defect_description'])
    # drop recurrent defects with section 0 (it is a catch-all section that indicates a certain sloppiness when labeling
    trax_df_clean = trax_df_clean[trax_df_clean.rec_sec != 0]
    # add a label made from the concat of the chapter and section -> chap-sec, this is what we want to predict
    trax_df_clean['label'] = trax_df_clean[['rec_ch', 'rec_sec']].apply(lambda data: f"{data['rec_ch']}-{data['rec_sec']}", axis=1)

    # normalize text
    if args.normalization_method in NORMALIZATION_FUNCTIONS:
        normalization_function = NORMALIZATION_FUNCTIONS[args.normalization_method]
    else:
        raise ValueError("Please add your normalization function in the dictionary NORMALIZATION_FUNCTIONS.")

    trax_df_clean['normalized_desc'] = trax_df_clean.defect_description.apply(normalization_function)

    # split corpus
    train, validate, test = np.split(trax_df_clean.sample(frac=1, random_state=42),
                                     [int(.6 * len(trax_df_clean)), int(.8 * len(trax_df_clean))])
    print(f"Trax dataset split is: {len(train)} train, {len(validate)} dev, {len(test)} test.")

    # let us try a little classifier based on tf-idf
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(train.normalized_desc.tolist()).toarray()
    labels = train.label
    model = LinearSVC(random_state=42)
    model.fit(features, labels)
    predictions = model.predict(tfidf.transform(test.normalized_desc.tolist()).toarray())

    f1 = f1_score(test.label, predictions, average='micro')
    print(f"F1 score on test is {f1 * 100:.1f}%")


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
    if __acro_map is None:
        load_acro_map()

    toks = re.split(r'[\s\.,;/:\(\)-]', text)  # do not do this
    for i, tok in enumerate(toks):
        if tok in __acro_keys:
            toks[i] = __acro_map.get(tok)

    return ' '.join(toks)


if __name__ == '__main__':
    main()
