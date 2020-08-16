# Tenth Montreal Problem Solving Workshop / Air Canada
This code:
* has been used to prepare the data for the workshop
* shows how to load the data and work with the pandas [library](http://pandas.pydata.org/) to manipulate the data
* shows how to code a dummy clusterer and how to evaluate it 

## Sample code

A sample clusterer is provided in the file `sample_clusterer.py` Its sole purpose is to show how to load the data 
and manipulate it, then cluster it and evaluate your algorithm. Feel free to do whatever you like with this.
The evaluation framework can also be modified!

Similarly, for labeling, take a look at `sample_labeler.py` (and also at `sample_clusterer.py` which shows how to 
use pandas.) 

## Data preparation
To prepare the data, the following recipe was used, starting from the original data set in Excel form, to produce the 
final dataset used in the workshop (`aircan-data-split-clean.pkl` and the equivalent `aircan-data-split-clean.xlsx`).

**You don't have to rerun this, just use the pickle provided, unless you have difficulty loading the pickle.**

The data split was 82.5% train, 7.3% dev, 10.2% test.

```
export INPUT_DIR=/your/input/directory
export OUTPUT_DIR=/your/output/directory

python import_excel.py ${INPUT_DIR}/10-july-2020/IVADO\ Data\ July\ 10\ 2020.xlsx ${OUTPUT_DIR}/aircan-data-2018-raw.pkl
python import_excel.py ${INPUT_DIR}/15-june-2020/IVADO\ Data\ 15\ June\ 2020.xlsx ${OUTPUT_DIR}/aircan-data-2019-raw.pkl
python sanitize.py ${OUTPUT_DIR}/aircan-data-2018-raw.pkl ${OUTPUT_DIR}/aircan-data-2018-clean.pkl
python sanitize.py ${OUTPUT_DIR}/aircan-data-2019-raw.pkl ${OUTPUT_DIR}/aircan-data-2019-clean.pkl
python combine_datasets.py ${OUTPUT_DIR}/aircan-data-2018-clean.pkl ${OUTPUT_DIR}/aircan-data-2019-clean.pkl ${OUTPUT_DIR}/aircan-data-full-clean.pkl
python split_dataset.py  ${OUTPUT_DIR}/aircan-data-full-clean.pkl  ${OUTPUT_DIR}/aircan-data-split-clean.pkl
python dump_to_excel.py  ${OUTPUT_DIR}/aircan-data-split-clean.pkl ${OUTPUT_DIR}/aircan-data-split-clean.xlsx
```

---
Fabrizio G
