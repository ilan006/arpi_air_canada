A few (modest) resources. Nothing large.
* `acronyms_1.tsv`: A TSV file containing acronym mapping from Wikipedia and CASA sources. See website.
* `airport_codes.tsv`: A file mapping airport codes (YUL - Montreal) to their names.
* `en_dict.txt`: An english dictionary containing inflected forms
* `en_dict_short.txt`: An english dictionary containing only the 55k of the most frequent Wikipedia tokens.
* `valid_cluster_ids.txt`: A list of cluster ids that were deemed valid by humans.
* `spelling_full.txt`: A list of word corrections output by `preprocessing/spell_check.py` the third column is a confidence value based on the data. 2 means that the correction was found in the data (so, not always accurate), 1 means it is straight out of the `en_dict.txt` file.
* `spelling_short.txt`: Same list as above but computed with the shorter english dictionary (saves computating time).
