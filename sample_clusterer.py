"""
Sample program to show how to load the data and how to evaluate an algorithm.
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

    # read data, this will load the data as 6 pandas DataFrames, which allow fast manipulations and (slower) iterations
    # more info on pandas here: https://pandas.pydata.org/
    with open(args.input_file, 'rb') as fin:
        [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)
        print(f"Read # samples: {len(defect_df_train)} train, {len(defect_df_test)} dev, {len(defect_df_test)} test.")

    # show a few things possible with the pandas data frames
    print(f"\nThere are {len(defect_df_train.columns)} columns in all 3 defect dataframes, whose names and types are:")
    print(defect_df_train.dtypes)
    print("\nThe first few rows for train are:")
    print(defect_df_train.head())
    print(f"\nThere are {len(defect_df_train.ac.unique())} unique aircrafts in train, "
          f"{len(defect_df_dev.ac.unique())} in dev and {len(defect_df_test.ac.unique())} in test.")
    print(f"The 3rd defect text for dev : {defect_df_dev.iloc[2, defect_df_dev.columns.get_loc('defect_description')]}")

    # show how a dummy clusterer can be evaluated and further shows ahow pandas can be used
    test_predictions = find_recurrent_defects(defect_df_test)
    eval_results = arpi_evaluator.evaluate_recurrent_defects(defect_df_test, test_predictions)


def find_recurrent_defects(defect_df):
    """
    Finds recurrent defects.
    :param defect_df: The defect dataframe for which we try to find recurrent defects.
    :return: A simple list of strings indicating the cluster each defect is in, e.g. ['a', 'b', 'red', 'yo', ...].
             If a given defect is not in a cluster, None is returned for that index, e.g. ['a', 'b', None, 'red', ...].
             The list of clusters is in the same order as the defects in the input dataframe. The cluster names are
             arbitrary, and can be anything.
    """
    return None


if __name__ == '__main__':
    main()
