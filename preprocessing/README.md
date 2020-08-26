# Text preprocessing modules

## Spell checking
The `spell_check.py` tool computes a list of word corrections from the defect description and the resolution description of the data.
Use it from the parent directory.

Usage:
```
python preprocessing/spell_check.py aircan-data-split-clean.pkl spelling_output_file
```

You can specify an other english dictionary than the default on using `--en_dictionary PATH`

Depending on the size of the dictionary, the operation may be long and processor intensive.
If the spell check output file is already generated, use the following to load it and skip the computation:
```
from preprocessing.spell_check import load_spell_dict

spelling = load_spell_dict('path_to_the_file')
```
