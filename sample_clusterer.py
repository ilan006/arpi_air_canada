"""
Sample program to show how to load the data and how to evaluate a (dummy) algorithm.
"""
import argparse
import arpi_evaluator
import pandas as pd
import pickle
import sys


def main():
    # parse args
    parser = argparse.ArgumentParser("A sample program.")
    parser.add_argument("input_file", help="A pickle input file, e.g. aircan-data-split-clean.pkl.")
    parser.add_argument("output_file", help="An output file.")
    args = parser.parse_args()

    # read data; this will load the data as 6 pandas DataFrames, which allow fast manipulations and (slower) iterations
    # more info on pandas here: https://pandas.pydata.org/
    with open(args.input_file, 'rb') as fin:
        [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)
        print(f"Read # samples: {len(defect_df_train)} train, {len(defect_df_test)} dev, {len(defect_df_test)} test.")

    # show a few things possible with the pandas data frames for those who are not familiar
    print(f"\nThere are {len(defect_df_train.columns)} columns in all 3 defect dataframes, whose names and types are:")
    print(defect_df_train.dtypes)
    print("\nThe first few rows for train are:")
    print(defect_df_train.head())
    print(f"\nThere are {len(defect_df_train.ac.unique())} unique aircrafts in train, "
          f"{len(defect_df_dev.ac.unique())} in dev and {len(defect_df_test.ac.unique())} in test.")
    print(f"The 3rd defect text for dev : {defect_df_dev.iloc[2, defect_df_dev.columns.get_loc('defect_description')]}")

    # show how a dummy clusterer can be evaluated and further shows how pandas can be used
    print("\nPredicting clusters.")
    test_predictions = find_recurrent_defects(defect_df_test)

    print("\nEvaluating clusters.")
    eval_results = arpi_evaluator.evaluate_recurrent_defects(defect_df_test, test_predictions)


def find_recurrent_defects(defect_df):
    """
    Finds recurrent defects (dummy algorithm to show how to proceed with the data).

    :param defect_df: The defect dataframe for which we try to find recurrent defects.
    :return: A mapping between defect id and predicted cluster. The cluster None is used to indicate that the
             defect is not recurrent.
    """

    # we prepare the result datastructure, mapping a defect id to its cluster assignment, e.g.
    # {'C-65335-1': 'blue', 'L-32159-1': 'red', 'C-12341-2': None} where None means that the defect is not recurrent
    result = {}

    # we regroup defects by aircraft ('ac') since a given defect cannot be recurrent across different aircraft
    grouped_by_ac = defect_df.groupby('ac')

    # we iterate over each aircraft group, with a tuple (name of aircraft, dataframe for all defects for this aircraft)
    for name, ac_group in grouped_by_ac:
        print(f"Aircraft {name} has {len(ac_group)} defects reported.")

        candidate_clusters = {}  # a map for our dymmy algorithm

        # we can then iterate over all rows of the data and use the fields we want.
        # index is the row number and row is the data itself
        for index, row in ac_group.iterrows():
            cur_type = row['defect_type']  # e.g. C, E or L
            cur_chapter = row['chapter']  # the ATA chapter

            my_key = cur_type + '-' + cur_chapter
            candidate_clusters[my_key] = candidate_clusters.get(my_key, set())
            candidate_clusters[my_key].add(row)




    return result


if __name__ == '__main__':
    main()
