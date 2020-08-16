"""Sample labeler (ATA taxonomy)"""
import argparse
import numpy as np
import pickle
import traceback
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
import os
import sys


def main():
    # parse args
    parser = argparse.ArgumentParser("A sample program.")
    parser.add_argument("input_file", help="A pickle input file, e.g. aircan-data-split-clean.pkl.")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Invalid input file: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # read data; this will load the data as 6 pandas DataFrames, which allow fast manipulations and (slower) iterations
    # more info on pandas here: https://pandas.pydata.org/
    try:
        with open(args.input_file, 'rb') as fin:
            [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)
    except:
        print("Loading the pickle failed.", file=sys.stderr)

        if pd.__version__ != '1.1.0':
            print("""You can upgrade your version of pandas with the command 
                  'pip install 'pandas==1.1.0' --force-reinstall'.""", file=sys.stderr)

        print("""You can also recreate the pickle by following the instructions here: 
                 https://github.com/rali-udem/arpi_air_canada#data-preparation""", file=sys.stderr)
        print()
        traceback.print_exc()

    # let us look at the labeling data from TRAX dataset!

    # remove empty descriptions
    trax_df_clean = trax_df.dropna(subset=['defect_description'])
    # drop recurrent defects with section 0 (it is a catch-all section that indicates a certain sloppiness when labeling
    trax_df_clean = trax_df_clean[trax_df_clean.rec_sec != 0]
    # add a label made from the concat of the chapter and section -> chap-sec, this is what we want to predict
    trax_df_clean['label'] = trax_df_clean[['rec_ch', 'rec_sec']].apply(lambda data: f"{data['rec_ch']}-{data['rec_sec']}", axis=1)
    train, validate, test = np.split(trax_df_clean.sample(frac=1, random_state=42),
                                     [int(.6 * len(trax_df_clean)), int(.8 * len(trax_df_clean))])
    print(f"Trax dataset split is: {len(train)} train, {len(validate)} dev, {len(test)} test.")

    # let us try a little classifier based on tf-idf, which even ignores the hierarchical nature of the ATA labels (chapter / section)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(train.defect_description.tolist()).toarray()
    labels = train.label
    model = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42)
    model.fit(features, labels)
    predictions = model.predict(tfidf.transform(test.defect_description.tolist()).toarray())

    f1 = f1_score(test.label, predictions, average='micro')
    print(f"F1 score on test is {f1 * 100:.1f}%")


if __name__ == '__main__':
    main()
