"""
This script splits the data in train, dev and test.

In the context of this workshop, we will only consider defects for the years 2018 and 2019. To split the dataset, we
will consider a partition of the aircraft mentioned in the defects. 80% of the aircraft will be to train, and 10%
of the aircraft will be held out for tuning and testing purposes.

This script carries out the split in a reproducible manner.

Fun fact: the plural of aircraft is aircraft.
"""
import hashlib
import pandas as pd
import pickle
import sys


def ac_name_to_split(ac_name):
    """
    A reproducible, portable way of partitioning dataset.
    :param ac_name: The aircraft name, i.e. AA-12028
    :return: Either 'train', 'dev', or 'test'
    """
    result = 'none'
    if not pd.isna(ac_name):
        byte_data = ac_name.strip().lower().encode('utf-8')
        cur_digest = int(hashlib.sha256(byte_data).hexdigest()[0:4], 16)
        cur_bin = cur_digest % 1000

        if cur_bin <= 800:
            result = 'train'
        elif cur_bin <= 900:
            result = 'dev'
        else:
            result = 'test'

    return result


def main():
    if len(sys.argv) != 3:
        print("Usage: prog input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    [defect_df, ata_df, mel_df, trax_df] = pickle.load(open(input_file, 'rb'))

    # find aircraft and their corresponding slice, e.g. AA-2012 -> 'train'
    aircraft_mapping = {ac: ac_name_to_split(ac) for ac in defect_df.ac.unique()}
    slices = list(aircraft_mapping.values())
    print(f"Split by ac is {100 * slices.count('train') / len(slices):.1f}% train, "
          f"{100 * slices.count('dev') / len(slices):.1f}% dev, "
          f"{100 * slices.count('test') / len(slices):.1f}% test")

    # now split the initial dataframe into three parts
    slice_df = defect_df['ac'].apply(ac_name_to_split)

    defect_df_train = defect_df[slice_df == 'train']
    defect_df_dev = defect_df[slice_df == 'dev']
    defect_df_test = defect_df[slice_df == 'test']

    print(f"Split by defects is {100 * len(defect_df_train) / len(defect_df):.1f}% train, "
          f"{100 * len(defect_df_dev) / len(defect_df):.1f}% dev, "
          f"{100 * len(defect_df_test) / len(defect_df):.1f}% test")

    # write
    print("Writing...", file=sys.stderr)
    pickle.dump([defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df], open(output_file, 'wb'))


if __name__ == '__main__':
    main()
