# SentenceLDA: Discriminative and Robust Document Representation with Sentence Level Topic Model (EACL 2024)

![Screenshot from 2024-02-07 16-09-38](https://github.com/cth127/sentencelda/assets/44315098/750000b4-78b9-4ed2-8d35-af7403fa20c3)


We introduce the new sentence-level topic model, SentenceLDA.
We found out that the SentenceLDA returns more discriminative, robust and interpretable document representation.

As our implementation is significantly based on the python implementation of [GaussianLDA](https://github.com/markgw/gaussianlda), we also follow their license, GPL-3.0.

## Installation
1. install requirements with `pip install -r requirements.txt`
2. install `choldate` with `pip install git+https://github.com/jcrudy/choldate.git`

## Run SentenceLDA
1. Put your corpus file on `/data` as a line-breaked `.txt` file. (See `/data/sample.txt`)
2. Run `train.py` with arguments `-d` (data path) and `-n` (the number of topics).
3. Run `generate.py` to check the topics in natural language form.
4. Check the `result.json`.

```bib
@inproceedings{
cha2024sentencelda,
title={SentenceLDA: Discriminative and Robust Document Representation with Sentence Level Topic Model},
author={Taehun Cha and Donghun Lee},
booktitle={18th Conference of the European Chapter of the Association for Computational Linguistics},
year={2024},
url={https://openreview.net/forum?id=wyx9hXM7NF}
}
```
