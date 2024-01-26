# Infusion

This repo contains code for conducting experiments with LED and LongT5 and structure infusion methods on downstream tasks. It is based on the [intertext graph](https://github.com/UKPLab/intertext-graph) as a source document. 

The main building blocks are **Tasks** (found in `evaluation/tasks/`) and **Models** (found in `structformer/led.py` and `structformer/longt5.py`). The exact setup is specified in the Hydra config (`config/`).

## Installation
Make a fresh conda environment and install the requirements:
```
conda create -n infusion python=3.10
conda activate infusion
pip install -r requirements.txt
```

## Setup

### Download datasets

Please download the data [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4111). Unzip the `infusion_datasets.zip` and move the contents to the `data/datasets/` directory. There should be three folders in `datasets/`: `EVIDENCE-INFERENCE-ITG`, `F1000-ITG` and `QASPER-ITG`.

You should have the following files:
```
analysis  # Contains scripts to produce plots and tables
config  # Contains the hydra config files
data
├── datasets  # Contains the downstream task datasets in ITG format
│   ├── EVIDENCE-INFERENCE-ITG
│   │   ├── deep-train.jsonl
│   │   ├── deep-dev.jsonl
│   │   ├── deep-test.jsonl
│   ├── QASPER-ITG
│   │   ├── deep-train.jsonl
│   │   ├── deep-dev.jsonl
│   │   ├── deep-test.jsonl
│   ├── F1000-ITG  # Use this to pretrain for probing
│   │   ├── dev
│   │   ├── test
│   │   ├── train
├── models  # Model checkpoints will be written here
├── plots  # Plots will be written here
├── predictions  # Predictions will be written here
├── probing_results  # Copy of the results from probing to do correlation analysis
│   ├──  in_n_out
│   │   ├──  f1000rd-full
│   │   │   ├──  led-base-16384
│   │   │   │   ├──  many folders with results
│   │   │   ├──  long-t5-tglobal-base
│   │   │   │   ├──  many folders with results
├── pytorch_lightning  # Output folder for lightning
├── results  # Results from experiments done for the paper. New results will be written here
│   ├──  many results files
├── tensorboard  # Tensorboard will log here
evaluation 
├── tasks  # Contains modules with task objects for QASPER and Evidence Inference, implementing data loading and evaluation
├── baselines.py  # Contains oracle baseline models
├── common.py  # Base classes for tasks, models, instances, etc.
├── run.py  # Implements training and evaluation
├── util.py  # Utility functions for evaluation
modeling
├── modeling_led.py  # Slight modification of LED from huggingface
structformer  # Contains modules for ITG-specific data processing
run_evaluation.py  # Main entry point into training and evaluation
```

## Running experiments
To run an experiment, use the `run_evaluation.py` script. For example, to run the LED model on Evidence Inference with node type position embeddings and node depth tokens, run:
```
python run_evaluation.py model=led_for_evidence_inference task=evidence_inference position_embeddings.mode=node_types input_sequence.mode=text_with_node_depths
```
If you are working on a cpu-only system, run
```
python run_evaluation.py model=led_for_evidence_inference task=evidence_inference position_embeddings.mode=node_types input_sequence.mode=text_with_node_depths accelerator=cpu
```

## Running Pretraining

To run pretraining, use the `Scaffold Task for Pretraining`:

Run this to pretrain LED on the evidence inference dataset
```
python run_evaluation.py model=led_for_scaffold_task_for_pretraining task=scaffold_task_for_pretraining task.datasets=['EVIDENCE-INFERENCE-ITG/deep]
``` 

This will save a model in `data/models` with a config specific hash. Find the hash in the results files in `data/results`.

To use the model in a downstream task, run this, replacing `<hash>` with the hash of the model you want to use:
```
python run_evaluation.py model=led_for_evidence_inference task=evidence_inference load_model=True hash_to_load=<hash>
```


## Reproducing the paper plots
To reproduce the plots and table from the paper, run the following commands:
```
cd analysis
python end_task_table.py
python separate_probing_to_end_task_correlation.py
```
