"""
This script combines datasets, one for 2018 and the other for 2019. It should also remove duplicates.
"""
import pickle
import sys

import pandas as pd


def main():
    if len(sys.argv) != 4:
        print("Usage: prog dataset_2018.pkl dataset_2019.pkl output_file.pkl", file=sys.stderr)
        sys.exit(1)

    dataset_2018 = sys.argv[1]
    dataset_2019 = sys.argv[2]
    output_file = sys.argv[3]

    [defect_df_2018, ata_df, mel_df, trax_df] = pickle.load(open(dataset_2018, 'rb'))  # better dataset
    [defect_df_2019, _, _, _] = pickle.load(open(dataset_2019, 'rb'))

    defect_df_2018.columns = map(str.lower, defect_df_2018.columns)
    defect_df_2019.columns = map(str.lower, defect_df_2019.columns)

    print("Removing duplicates...", file=sys.stderr)
    defect_df_full = pd.concat([defect_df_2018, defect_df_2019], sort=True)
    defect_df_full.drop_duplicates(['defect_type', 'defect', 'defect_item'], inplace=True)

    print("Sorting...", file=sys.stderr)
    defect_df_full.sort_values(by=['ac', 'reported_datetime'], inplace=True)

    print("Rearranging columns...", file=sys.stderr)
    proper_order = ['defect_type', 'defect', 'defect_item', 'defect_description', 'status', 'ac', 'reported_datetime',
                    'chapter', 'section', 'paragraph', 'i_f_s_d',  'defect_category', 'mddr', 'defer', 'defer_date',
                    'defer_hour', 'defer_minute', 'defer_to_date', 'defer_to_hour', 'defer_to_minute', 'mel',
                    'mel_number', 'reported_date', 'reported_hour', 'repoted_minute', 'resolution_description',
                    'resolution_category', 'resolved_date', 'resolved_hour', 'resolved_minute', 'created_date',
                    'modified_date', 'recurrent', 'schedule_hours', 'schedule_cycles', 'schedule_days', 'skill',
                    'man_hours', 'man_required', 'sdr', 'repeat_number', 'completed_number', 'defer_notes',
                    'release_to_service_by', 'mel_calendar_days_flag', 'pn_incident', 'ongoing_trouble_shooting',
                    'planning_dept', 'planning_sub_dept', 'schedule_hours_repeat', 'schedule_cycles_repeat',
                    'schedule_days_repeat', 'mel_sub', 'moc_item_description', 'asr', 'mel_alert']

    cols: list = defect_df_full.columns.tolist()

    for col_label in [proper_order[i] for i in range(len(proper_order) - 1, -1, -1) if proper_order[i] in cols]:
        cols.remove(col_label)
        cols.insert(0, col_label)

    defect_df_full = defect_df_full[cols]

    print("Writing...", file=sys.stderr)
    pickle.dump([defect_df_full, ata_df, mel_df, trax_df], open(output_file, 'wb'))
    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    main()
