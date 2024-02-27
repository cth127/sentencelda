# Reproducing the experiments in our paper

0. For each folder in `sentencelda_submit/topicmodel` (except `data`) run `train.py`

- For Text Classification
1. For each folder in `sentencelda_submit/topicmodel` (except `data` and `senclu`) run `infer.py`
2. Run `sentencelda_submit/classification/train.py`
3. Run `sentencelda_submit/classification/write_summary.py` and see the results.

- For Context-robustness Test
1. For each folder in `sentencelda_submit/topicmodel` (except `data` and `senclu`) run `infer_lexical.py`
2. For each folder in `sentencelda_submit/topicmodel` (except `data` and `senclu`) run `infer_syntactic.py`
3. Run `sentencelda_submit/robustness/robustnesss.py`
