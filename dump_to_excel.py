"""
This script sanitizes the workshop data by identifying problems, filtering out invalid data and formatting
fields with sensible types.
"""
import pandas as pd
import pickle
import sys


def main():
    if len(sys.argv) != 3:
        print("Usage: prog input_file.pkl output_file.xlsx")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(open(input_file, 'rb'))

    print("Writing (this can take a few minutes)...", file=sys.stderr, end=' ', flush=True)
    with pd.ExcelWriter(output_file) as writer:
        defect_df_train.to_excel(writer, sheet_name='defect_train', index=False, freeze_panes=(1, 0))
        defect_df_dev.to_excel(writer, sheet_name='defect_dev', index=False, freeze_panes=(1, 0))
        defect_df_test.to_excel(writer, sheet_name='defect_test', index=False, freeze_panes=(1, 0))
        trax_df.to_excel(writer, sheet_name='trax', index=False, freeze_panes=(1, 0))
        ata_df.to_excel(writer, sheet_name='ata', index=False, freeze_panes=(1, 0))
        mel_df.to_excel(writer, sheet_name='mel', index=False, freeze_panes=(1, 0))

    print("done.", file=sys.stderr)


if __name__ == '__main__':
    main()
