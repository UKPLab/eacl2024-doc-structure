# Document Structure in Long Document Transformers

-------
<img src="/img/eyecatcher.png" alt="image" width="300" height="auto">

--------

### Jan Buchmann, Max Eichler, Jan-Micha Bodensohn, Ilia Kuznetsov and Iryna Gurevych

[UKP Lab](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp), [TU Darmstadt](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp)

This repository contains the code for the paper "Document Structure in Long Document Transformers", accepted at EACL 2024 (link will follow soon).

The corresponding data can be found [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4111)

From the abstract of the paper: 

> Long documents often exhibit structure with hierarchically organized elements of different functions, such as section headers and paragraphs. Despite the omnipresence of document structure, its role in natural language processing (NLP) remains opaque. Do long-document Transformer models acquire an internal representation of document structure during pre-training? How can structural information be communicated to a model after pre-training, and how does it influence downstream performance? To answer these questions, we develop a novel suite of probing tasks to assess structure-awareness of long-document Transformers, propose general-purpose structure infusion methods, and evaluate the effects of structure infusion on QASPER and Evidence Inference, two challenging long-document NLP tasks.

We build our experiments on [Intertext Graphs](https://github.com/UKPLab/intertext-graph) (ITG) [1] as the common data format and employed two long document transformers: [LED](https://huggingface.co/docs/transformers/model_doc/led) [2] and [LongT5](https://huggingface.co/google/long-t5-tglobal-base) [3]. We performed downstream task experiments on [QASPER](https://allenai.org/data/qasper) [4] and [Evidence Inference](https://evidence-inference.ebm-nlp.com/) [5]. 

**Contact**: Jan Buchmann, jan.buchmann@tu-darmstadt.de
- UKP Lab: http://www.ukp.tu-darmstadt.de/
- TU Darmstadt: http://www.tu-darmstadt.de/

## Repository Structure

The repository is split into 3 parts. Each of these has a README that explains usage and a requirements.txt with dependencies.

### infusion

The infusion repository contains the code and data for downstream task experiments and pretraining. This includes the downstream task datasets in ITG format.

### probing

The probing repository contains the code for probing experiments and the probing datasets. To be able to use position embedding structure infusion, the infusion repository must be available.

### structure_datasets

The structure_datasets repository contains the code to create the downstream task datasets in the intertext graph format. Note that the the downstream task datasets in ITG format are available [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4111), so you should not need to recreate them.

## Usage

To reproduce the experiments from the paper, please download the datasets in ITG format [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4111) and unpack the zip files. 

Move the contents of `infusion_datasets` to `infusion/data/datasets/`. See `infusion/README.md` for basic instructions to run experiments.

Move the contents of `probing_datasets` to `probing/data/`. See `probing/README.md` for basic instructions to run experiments.

## Citation

If you happen to find our paper or this repository useful, please consider citing

[add citation here when we have the link] 

## References

[1] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. 2021. A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4599–4610, Online. Association for Computational Linguistics.

[2] Iz Beltagy, Matthew E. Peters, and Arman Cohan. "Longformer: The long-document transformer." arXiv preprint arXiv:2004.05150 (2020).

[3] Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo Ni, Yun-Hsuan Sung, and Yinfei Yang. 2022. LongT5: Efficient Text-To-Text Transformer for Long Sequences. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 724–736, Seattle, United States. Association for Computational Linguistics.

[4] Jay DeYoung, Eric Lehman, Benjamin Nye, Iain Marshall, and Byron C. Wallace. 2020. Evidence Inference 2.0: More Data, Better Models. In Proceedings of the 19th SIGBioMed Workshop on Biomedical Language Processing, pages 123–132, Online. Association for Computational Linguistics.

[5] Ilia Kuznetsov, Jan Buchmann, Max Eichler, Iryna Gurevych; Revise and Resubmit: An Intertextual Model of Text-based Collaboration in Peer Review. Computational Linguistics 2022; 48 (4): 949–986. doi: https://doi.org/10.1162/coli_a_00455

## Disclaimer

This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
