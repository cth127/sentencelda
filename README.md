# SentenceLDA: Discriminative and Robust Document Representation with Sentence Level Topic Model (EACL 2024)

![Screenshot from 2024-02-07 16-09-38](https://github.com/cth127/sentencelda/assets/44315098/750000b4-78b9-4ed2-8d35-af7403fa20c3)

[Paper](https://aclanthology.org/2024.eacl-long.31/)

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
@inproceedings{cha-lee-2024-sentencelda,
    title = "{S}entence{LDA}: Discriminative and Robust Document Representation with Sentence Level Topic Model",
    author = "Cha, Taehun  and
      Lee, Donghun",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.31",
    pages = "521--538",
    abstract = "A subtle difference in context results in totally different nuances even for lexically identical words. On the other hand, two words can convey similar meanings given a homogeneous context. As a result, considering only word spelling information is not sufficient to obtain quality text representation. We propose SentenceLDA, a sentence-level topic model. We combine modern SentenceBERT and classical LDA to extend the semantic unit from word to sentence. By extending the semantic unit, we verify that SentenceLDA returns more discriminative document representation than other topic models, while maintaining LDA{'}s elegant probabilistic interpretability. We also verify the robustness of SentenceLDA by comparing the inference results on original and paraphrased texts. Additionally, we implement one possible application of SentenceLDA on corpus-level key opinion mining by applying SentenceLDA on an argumentative corpus, DebateSum.",
}
```
