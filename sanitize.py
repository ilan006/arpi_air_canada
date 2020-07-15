"""
This script sanitizes the workshop data by identifying problems, filtering out invalid data and formatting
fields with sensible types.
"""
import pandas as pd
from datetime import datetime
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

    defect_df = defect_df.astype({'DEFECT': 'Int64', 'PARAGRAPH': 'Int64', 'MDDR': 'Int64',
                                  'REPORTED_HOUR': 'Int64', 'REPOTED_MINUTE': 'Int64',
                                  'RECURRENT': 'Int64', 'SCHEDULE_DAYS': 'Int64'}, copy=False)

    print("Converting dates...", file=sys.stderr)
    defect_df = convert_datetime(defect_df, ['REPORTED_DATE', 'REPORTED_HOUR', 'REPOTED_MINUTE'], 'REPORTED_DATETIME')
    defect_df = convert_datetime(defect_df, ['DEFER_DATE', 'DEFER_HOUR', 'DEFER_MINUTE'], 'DEFER_DATETIME')
    defect_df = convert_datetime(defect_df, ['DEFER_TO_DATE', 'DEFER_TO_HOUR', 'DEFER_TO_MINUTE'], 'DEFER_TO_DATETIME')
    defect_df = convert_datetime(defect_df, ['RESOLVED_DATE', 'RESOLVED_HOUR', 'RESOLVED_MINUTE'], 'RESOLVED_DATETIME')

    print("Fixing small stuff...", file=sys.stderr)
    defect_df.replace({'MEL_CALENDAR_DAYS_FLAG': "NO"}, {'MEL_CALENDAR_DAYS_FLAG': "N"}, inplace=True)
    defect_df.columns = map(str.lower, defect_df.columns)
    trax_df.columns = map(str.lower, trax_df.columns)

    print(defect_df.dtypes)
    print(defect_df.head())

    print("Writing...", file=sys.stderr)
    pickle.dump([defect_df, ata_df, mel_df, trax_df], open(output_file, 'wb'))


if __name__ == '__main__':
    main()
