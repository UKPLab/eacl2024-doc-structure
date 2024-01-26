# Structure Infusion Benchmark - Data Handling

This repository provides the scripts and notebooks to transform the following datasets into the ITG format:

- QASPER ([A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers](https://aclanthology.org/2021.naacl-main.365), Dasigi et al., NAACL 2021)
- Evidence Inference ([Evidence Inference 2.0: More Data, Better Models](http://aclanthology.lst.uni-saarland.de/2020.bionlp-1.13/), DeYoung et al., BioNLP 2020)

The scripts are grouped by the dataset and depend on the `config.py`, `histogram.py`, and `metadata_analyzer.py`
helpers.

## Setup
Create a new conda environment and install the requirements:
```
conda create -n itg_datasets python=3.10
conda activate itg_datasets
pip install -r requirements.txt
```

## Running the scripts

Download the datasets and unpack them
Adapt the file paths in `path_config_local.json`.

### QASPER

- Download the QASPER dataset at https://allenai.org/data/qasper and unpack it.
- Adapt the file paths in `path_config_local.json`. `path.QASPER": "/path/to/unzipped/qasper` should point to a folder containing the dataset content (`qasper-dev-v0.3.json` etc).

```
cd qasper
export PYTHONPATH='../'; python qasper_transform_into_itg_format.py
```

### Evidence Inference
- Download the Evidence Inference dataset as a zip file from https://github.com/jayded/evidence-inference.
- Unpack it and adapt the file paths in `path_config_local.json`. `path.EVIDENCE-INFERENCE": "/path/to/unzipped/EVIDENCE-INFERENCE-dataset` should point to a folder containing a directory named `evidence-inference` with the contents of the github repository.

```
cd evidence_inference
export PYTHONPATH='../'; python evidence_inference_transform_into_itg_format.py
```
