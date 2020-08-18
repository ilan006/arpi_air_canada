"""Provide a few simple stats for the dataset"""

import argparse
import numpy as np
import pandas as pd
import pickle
import re
import sys

no_letter_pattern = re.compile("^[^a-zA-Z]*$")
def text_stats(series, fout, exclude_word_list):
    exclude_list = set()
    if exclude_word_list is not None:
        with open(exclude_word_list) as fin:
            exclude_list = set(fin.read().split('\n'))

    vocabulary = dict()
    total = 0
    rows_empty = 0
    for txt in series:
        if type(txt) != str:
            rows_empty += 1
            continue
        txt = txt.replace(",", "")
        txt = txt.replace(".", "")
        txt = txt.replace("(", "")
        txt = txt.replace(")", "")
        for token in txt.split(' '):
            if not no_letter_pattern.match(token) and token.lower() not in exclude_list:
                total += 1
                nb = vocabulary.get(token, 0)
                vocabulary[token] = nb + 1
    avg = total / len(vocabulary)
    med = sorted(vocabulary.values())[round(len(vocabulary)/2)]

    print(f"{rows_empty} empty rows", file=sys.stderr)

    print(f"average: {avg}\nmedian: {med}\n", file=fout)
    for word, count in sorted(vocabulary.items(), key=lambda item: item[1], reverse=True):
        print(f"{count}\t{word}", file=fout)
    

# parse args
parser = argparse.ArgumentParser("A sample program.")
parser.add_argument("input_file", help="A pickle input file, e.g. aircan-data-split-clean.pkl.")
parser.add_argument("--description_stats_output_file", help="Output file for simple text stats on the defect description field.")
parser.add_argument("--word_exclude_file", help="Use with description_stats to ignore words from a given dictionary file.") # e.g. en_dict.txt

args = parser.parse_args()

with open(args.input_file, 'rb') as fin:
    [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)
    print(f"Read # samples: {len(defect_df_train)} train, {len(defect_df_dev)} dev, {len(defect_df_test)} test.")

defect_df_full = pd.concat([defect_df_train, defect_df_test, defect_df_dev], sort=False)

print(f"Nb  of defects: {len(defect_df_full)}")
text_len = defect_df_full.defect_description.apply(lambda s: len(s) if not pd.isnull(s) else np.nan)
nb_toks = defect_df_full.defect_description.apply(lambda s: len(re.split(r'[\s,\.:;]', s)) if not pd.isnull(s) else np.nan)

print(f"Avg text len in chars: {text_len.mean():.1f}")
print(f"Avg text len in tokens: {nb_toks.mean():.1f} +- {nb_toks.std():.1f}")

if args.description_stats_output_file is not None:
    with open(args.description_stats_output_file, 'w') as fout:
        text_stats(defect_df_full.defect_description, fout, args.word_exclude_file)
