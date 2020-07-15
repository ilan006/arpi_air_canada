"""
This script combines datasets, one for 2018 and the other for 2019. It should also remove duplicates.
"""
import pickle

import pandas as pd
import sys


def main():
    if len(sys.argv) != 4:
        print("Usage: prog dataset_2018.pkl dataset_2019.pkl output_file.pkl", file=sys.stderr)
        sys.exit(1)

    dataset_2018 = sys.argv[1]
    dataset_2019 = sys.argv[2]
    output_file = sys.argv[3]

    [defect_df_2018, ata_df, mel_df, trax_df] = pickle.load(open(dataset_2018, 'rb'))  # better dataset
    [defect_df_2019, _, _, _] = pickle.load(open(dataset_2019, 'rb'))

    print("Removing duplicates...", file=sys.stderr)
    defect_df_full = pd.concat([defect_df_2018, defect_df_2019], sort=True)
    defect_df_full.drop_duplicates(['defect_type', 'defect', 'defect_item'], inplace=True)

    print("Writing...", file=sys.stderr)
    pickle.dump([defect_df_full, ata_df, mel_df, trax_df], open(output_file, 'wb'))


if __name__ == '__main__':
    main()
