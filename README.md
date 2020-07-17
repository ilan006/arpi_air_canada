# Tenth Montreal Problem Solving Workshop / Air Canada
This code:
* has been used to prepare the data for the workshop
* shows how to load the data and work with the pandas [library](http://pandas.pydata.org/) to manipulate the data
* shows how to code a dummy clusterer and how to evaluate it 

## Data preparation
To prepare the data, the following recipe was used, starting from the original data set in Excel form, to produce the 
final dataset used in the workshop (`aircan-data-split-clean.pkl` and `aircan-data-split-clean.xlsx`).

```
python import_excel.py ${INPUT_DIR}/10-july-2020/IVADO\ Data\ July\ 10\ 2020.xlsx ${OUTPUT_DIR}/aircan-data-2018-raw.pkl`
python import_excel.py ${INPUT_DIR}/15-june-2020/IVADO\ Data\ 15\ June\ 2020.xlsx ${OUTPUT_DIR}/aircan-data-2019-raw.pkl
python sanitize.py ${OUTPUT_DIR}/aircan-data-2018-raw.pkl ${OUTPUT_DIR}/aircan-data-2018-clean.pkl
python sanitize.py ${OUTPUT_DIR}/aircan-data-2019-raw.pkl ${OUTPUT_DIR}/aircan-data-2019-clean.pkl
python combine_datasets.py ${OUTPUT_DIR}/aircan-data-2018-clean.pkl ${OUTPUT_DIR}/aircan-data-2019-clean.pkl ${OUTPUT_DIR}/aircan-data-full-clean.pkl
python split_dataset.py  ${OUTPUT_DIR}/aircan-data-full-clean.pkl  ${OUTPUT_DIR}/aircan-data-split-clean.pkl
python dump_to_excel.py  ${OUTPUT_DIR}/aircan-data-split-clean.pkl ${OUTPUT_DIR}/aircan-data-split-clean.xlsx
```

## Sample code

---
Fabrizio G
