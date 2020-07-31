"""
This script sanitizes the workshop data by identifying problems, filtering out invalid data and formatting
fields with sensible types.
"""
import math

import pandas as pd
import pickle
import sys


def multiple_to_datetime(args):
    result = args[0]
    hour = args[1]
    minute = args[2]

    if pd.isna(minute):
        minute = 0

    if pd.isnull(result) or pd.isna(hour):
        result = pd.NaT
    else:
        result = result.replace(hour=hour, minute=minute)

    return result


def convert_datetime(df, column_list: list, new_column_name):
    """
    Converts dataframe time stamps.
    :param df: the dataframe to convert columns for.
    :param column_list: date, hour, minute columns in that order
    :param new_column_name: new column name to add to returned df
    :return: a new dataframe with column_list removed in favor of a new datetime column
    """
    df = df.astype({col: 'Int64' for col in column_list[1:3]}, copy=False)
    df[new_column_name] = df[column_list].apply(multiple_to_datetime, axis=1)
    df.drop(column_list, axis=1, inplace=True)

    return df


def main():
    if len(sys.argv) != 3:
        print("Usage: prog input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    [defect_df, ata_df, mel_df, trax_df] = pickle.load(open(input_file, 'rb'))

    # all column names in uppper case
    defect_df.columns = map(str.upper, defect_df.columns)
    trax_df.columns = map(str.upper, trax_df.columns)

    # convert all field types
    corrupted_dataset = False
    try:
        defect_df.astype({'DEFECT': 'Int64'})
    except ValueError:
        # more vigorous filtering necessary (2019 dataset)
        corrupted_dataset = True
        print("Detected corrupted dataset - will try to patch.")

    int_field_names = ['DEFECT', 'PARAGRAPH', 'MDDR', 'REPORTED_HOUR', 'REPOTED_MINUTE', 'RECURRENT', 'SCHEDULE_DAYS',
                       'DEFER_HOUR', 'DEFER_MINUTE', 'DEFER_TO_HOUR', 'DEFER_TO_MINUTE', 'RESOLVED_HOUR',
                       'RESOLVED_MINUTE']

    if corrupted_dataset:
        for int_field_name in int_field_names:
            defect_df[int_field_name] = pd.to_numeric(defect_df[int_field_name], errors='coerce')
            defect_df[int_field_name] = defect_df[int_field_name].apply(lambda x, axis: x if pd.isna(x) else math.floor(x), axis=1)
        for datetime_fieldname in ['REPORTED_DATE', 'DEFER_DATE', 'DEFER_TO_DATE', 'RESOLVED_DATE']:
            defect_df[datetime_fieldname] = pd.to_datetime(defect_df[datetime_fieldname], errors='coerce')
        defect_df.drop(defect_df.columns[55:], axis=1, inplace=True)

    defect_df = defect_df.astype({int_field_name: 'Int64' for int_field_name in int_field_names}, copy=False)

    print("Converting dates...", file=sys.stderr)
    defect_df = convert_datetime(defect_df, ['REPORTED_DATE', 'REPORTED_HOUR', 'REPOTED_MINUTE'], 'REPORTED_DATETIME')
    defect_df = convert_datetime(defect_df, ['DEFER_DATE', 'DEFER_HOUR', 'DEFER_MINUTE'], 'DEFER_DATETIME')
    defect_df = convert_datetime(defect_df, ['DEFER_TO_DATE', 'DEFER_TO_HOUR', 'DEFER_TO_MINUTE'], 'DEFER_TO_DATETIME')
    defect_df = convert_datetime(defect_df, ['RESOLVED_DATE', 'RESOLVED_HOUR', 'RESOLVED_MINUTE'], 'RESOLVED_DATETIME')

    defect_df = defect_df[defect_df.DEFECT.notnull() & defect_df.REPORTED_DATETIME.notnull()]

    if corrupted_dataset:
        # filter out spurious entries
        print("Filtering out invalid entries...", file=sys.stderr)
        trax_df.drop(trax_df.columns[13:21], axis=1, inplace=True)

    print("Fixing small stuff...", file=sys.stderr)
    defect_df.replace({'MEL_CALENDAR_DAYS_FLAG': "NO"}, {'MEL_CALENDAR_DAYS_FLAG': "N"}, inplace=True)
    defect_df.columns = map(str.lower, defect_df.columns)
    trax_df.columns = map(lambda x: x.lower().replace(' ', '_').replace('-', '_'), trax_df.columns)

    print("Indexing...", file=sys.stderr)
    ids = defect_df[['defect_type', 'defect', 'defect_item']].apply(lambda x: f'{x[0]}-{str(x[1])}-{str(x[2])}', axis=1)
    defect_df.index = ids

    print(defect_df.dtypes)
    print(defect_df.head())

    print("Writing...", file=sys.stderr)
    pickle.dump([defect_df, ata_df, mel_df, trax_df], open(output_file, 'wb'))


if __name__ == '__main__':
    main()
