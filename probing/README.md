# Probing Kit

## Dependency setup

    conda create --name longdoc python=3.10
    conda activate longdoc
    pip install allennlp transformers torch git+https://github.com/UKPLab/intertext-graph.git
    # Or
    pip install -r requirements.txt

## Dataset setup

Download the probing dataset (probing_datasets.zip) [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4111).
 
Unzip the `probing_datasets.zip` file and copy the files from the `intertext_graph` folder to `out/f1000rd-full/led-base-16384/intertext_graph` and `out/f1000rd-full/long-t5-tglobal-base/intertext_graph`.

    python create_dataset.py --model allenai/led-base-16384
    python create_dataset.py --model google/long-t5-tglobal-base

To run probing experiments with special token infusion, a separate dataset is needed. This can be created with:

    python create_dataset.py --model allenai/led-base-16384 --infusion_method node_boundaries --probe_structure_tags
    python create_dataset.py --model google/long-t5-tglobal-base --infusion_method node_boundaries --probe_structure_tags

For node type / node depth infusion, replace `node_boundaries` with `node_types` or `node_depth`.

## Run probes

    # You can set different run ids for different experiments, e.g. 001_led, or 002_lt5
    python run_probing.py 001_led --model allenai/led-base-16384 --batch_size 4

To run the atomic or random control add `--atomic` or `--random`.

To run only a selection of probes specify e.g. `--probes node_type position`.

The results can be found under `out/<model_name>/<run_id>/<probe_name>/probing.json`, e.g. `out/led-base-16384/001_led/node_type/probing.json`.

### Infusion

**Embeddings**

To add node type embedding structure infusion, run (assuming that the probing code is in the same directory as the infusion code)

    export PYTHONPATH='../:../infusion'; python run_probing.py 001_led --model allenai/led-base-16384 --batch_size 4 --position_embeddings_mode node_types

For node depth infusion, replace `node_types` with `node_depths`.

**Special Tokens**

To add node boundary token structure infusion, add

`--infusion_method node_boundaries --probe_structure_tags`. 

For node type / node depth infusion, replace `node_boundaries` with `node_types` or `node_depth`.

### Loading a pretraing model

If you pretrained a model with the infusion code and want to load it, add 

`--state_dict_path <path_to_model>`

### Batch size

On an A100 40 GB GPU we recommend batch size 4 for the vanilla and random configuration, and 64 for the atomic control.

## Re-create plots

    cd scripts/paper_plots
    export PYTHONPATH='../../'; python probing_plot.py  # Bar plot of baseline probes, not used in paper
    export PYTHONPATH='../../'; python infused_probing_plot.py  # Fig. 5
    export PYTHONPATH='../../'; python layer_utilization.py  # Fig. 7
    export PYTHONPATH='../../'; probing_table.py  # Table 2 and Table 4 

## Adding new self-hosted models

The following describes the addition of new self-hosted models (i.e. models with access to hidden states) for probing experiments. If you want to do experiments on API-served models, this likely requires a re-implementation of dataset creation and probing experiments.

Adding a new model for probing experiments requires implementing model-specific classification as well as producing a model-specific dataset. We need model-specific datasets because the probing datasets are pre-tokenized using the model-specific tokenization. 

### Producing a new dataset

Producing a new dataset should be fairly simple if you are using a huggingface based model. Simply run 
```
python create_dataset.py --model <hf_model_name> --max_length <max_doc_length_in_tokens>
```
