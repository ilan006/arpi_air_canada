"""Simple Levenshtein based spell checker"""

import argparse
import numpy as np
import pandas as pd
import pickle
import re
from tqdm import tqdm

MAX_EDIT_RATIO = 1/5
def capped_levenshtein(token: str, dictionary: set):
    tok_len = len(token)
    if tok_len * MAX_EDIT_RATIO < 1 or token in dictionary:
        return None

    candidates = list()
    for word in dictionary:
        word_len = len(word)
        if word_len * MAX_EDIT_RATIO < 1:
            continue
        cap = min(word_len, tok_len) * MAX_EDIT_RATIO

        tab = list()
        for i in range(tok_len+1):
            tab.append([i])
        for j in range(1, word_len+1):
            tab[0].append(j)

        for i in range(1, tok_len+1):
            local_minimum = max(tok_len, word_len)
            for j in range(1, word_len+1):
                edit = 1
                if token[i-1] == word[j-1]:
                    edit = 0
                current = min(tab[i-1][j]+1, tab[i][j-1]+1, tab[i-1][j-1]+edit)

                swap = 2
                if i > 1 and j > 1 and token[i-1] == word[j-2] and token[i-2] == word[j-1]:
                    swap = 1
                    current = min(current, tab[i-2][j-2]+swap)
                
                tab[i].append(current)
                local_minimum = min(local_minimum, tab[i][j])
            if local_minimum > cap:
                break

        if len(tab[tok_len]) == word_len+1 and tab[tok_len][word_len] <= cap:
            candidates.append((word, tab[tok_len][word_len]))

    candidates = sorted(candidates, key=lambda c:c[1])
#    print(f"{token} -> {candidates}")

    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        return candidates[0][0]
    else:
        if candidates[0][1] == candidates[1][1]:  # ambiguity
            return None
        else:
            return candidates[0][0]


def spell_check(series, fout, en_dict_file):
    en_dict = set()
    with open(en_dict_file) as fin:
        for word in fin.read().split():
            if len(word) * MAX_EDIT_RATIO >= 1:
                en_dict.add(word)

    spell_check_dict = dict()
    rows_empty = 0

    for txt in tqdm(series):
        if type(txt) != str:
            rows_empty += 1
            continue
        txt = txt.replace(",", "")
        txt = txt.replace(".", "")
        txt = txt.replace(";", "")
        txt = txt.replace(":", "")
        txt = txt.replace('"', "")
        txt = txt.replace("(", "")
        txt = txt.replace(")", "")
        txt = txt.replace("[", "")
        txt = txt.replace("]", "")
        for token in txt.split():
            token = token.lower()

            if token.isalpha() and token not in en_dict and not token in spell_check_dict:
                spell_check_dict[token] = capped_levenshtein(token, en_dict)

    for token, corrected in spell_check_dict.items():
        if corrected is not None:
            print(f"{token}\t{corrected}", file=fout)

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_input_file", help="A pickle input file, e.g. aircan-data-split-clean.pkl.")
    
    parser.add_argument("spell_check_output_file", help="Output file for the spell checking.")
    parser.add_argument("--en_dictionary", default="small_resources/en_dict_short.txt", help="Specify a different dictionary to check words against.")
    args = parser.parse_args()
    
    with open(args.corpus_input_file, 'rb') as fin:
        [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)
    
    defect_df_full = pd.concat([defect_df_train, defect_df_test, defect_df_dev], sort=False)
    
    with open(args.spell_check_output_file, 'w') as fout:
        spell_check(defect_df_full.defect_description, fout, args.en_dictionary)
