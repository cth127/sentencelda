import pickle
from pathlib import Path
import json
import os

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


TARGET = {"20newsgroup": {"computer": ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'],
               "ride": ['rec.autos', 'rec.motorcycles'],
               "sports": ['rec.sport.baseball', 'rec.sport.hockey'],
               "science": ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
               "religion": ['alt.atheism', 'soc.religion.christian', 'talk.religion.misc'],
               "politics": ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc']},
          "nyt": {"arts": ['dance', 'music', 'movies', 'television'],
                  "business": ['economy', 'energy_companies', 'international_business', 'stocks_and_bonds'],
                  "politics": ['abortion', 'federal_budget', 'gay_rights', 'gun_control', 'immigration',
                               'law_enforcement', 'military', 'surveillance', 'the_affordable_care_act'],
                  "science": ['cosmos', 'environment'],
                  "sports": ['baseball', 'basketball', 'football', 'golf', 'hockey', 'soccer', 'tennis']}}


def write_json(res, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


def topic_distribution(x_data):
    ret = list()
    for line in x_data:
        if line.sum() != 0:
            ret.append(line / line.sum())
        else:
            ret.append(line)
    return np.array(ret)


def train(category, dataset, num_topics, it):
    ctm_path = Path(__file__).parents[1] / f'topicmodel/ctm/params/iter{it}/{dataset}/{num_topics}topic/{category}'
    lda_path = Path(__file__).parents[1] / f'topicmodel/bleilda/params/iter{it}/{dataset}/{num_topics}topic/{category}'
    glda_path = Path(__file__).parents[1] / f'topicmodel/gaussianlda/params/iter{it}/{dataset}/{num_topics}topic/{category}'
    senclu_path = Path(__file__).parents[1] / f'topicmodel/senclu/params/iter{it}/{dataset}/{num_topics}topic/{category}.pkl'
    slda_path = Path(__file__).parents[1] / f'topicmodel/sentence_gaussianlda/params/iter{it}/{dataset}/{num_topics}topic/{category}'
    ctm_test_path = Path(__file__).parents[1] / f'topicmodel/ctm/infer/iter{it}/{dataset}/{num_topics}topic/{category}.pkl'
    lda_test_path = Path(__file__).parents[1] / f'topicmodel/bleilda/infer/iter{it}/{dataset}/{num_topics}topic/{category}.pkl'
    glda_test_path = Path(__file__).parents[1] / f'topicmodel/gaussianlda/infer/iter{it}/{dataset}/{num_topics}topic/{category}.pkl'
    senclu_test_path = Path(__file__).parents[1] / f'topicmodel/senclu/infer/iter{it}/{dataset}/{num_topics}topic/{category}.pkl'
    slda_test_path = Path(__file__).parents[1] / f'topicmodel/sentence_gaussianlda/infer/iter{it}/{dataset}/{num_topics}topic/{category}.pkl'

    train_data = pd.read_csv(Path(__file__).parents[1] / f'topicmodel/data/{dataset}/train.csv')
    train_labels = train_data.loc[train_data["target_str"].isin(TARGET[dataset][category])]['target_str'].to_list()
    label_map = {i:n for n, i in enumerate(set(train_labels))}
    train_y = [label_map[i] for i in train_labels]
    test_data = pd.read_csv(Path(__file__).parents[1] / f'topicmodel/data/{dataset}/test.csv')
    test_labels = test_data.loc[test_data["target_str"].isin(TARGET[dataset][category])]['target_str'].to_list()
    test_y = [label_map[i] for i in test_labels]

    with open(ctm_path / 'ctm.pkl', 'rb') as f:
        train_X1 = topic_distribution(pickle.load(f))
    with open(lda_path / 'lda_comp.pkl', 'rb') as f:
        train_X2 = topic_distribution(pickle.load(f))
    with open(glda_path / 'table_counts_per_doc.pkl', 'rb') as f:
        train_X3 = topic_distribution(pickle.load(f))
    with open(senclu_path, 'rb') as f:
        train_X4 = topic_distribution(pickle.load(f))
    with open(slda_path / 'table_counts_per_doc.pkl', 'rb') as f:
        train_X5 = topic_distribution(pickle.load(f))

    with open(ctm_test_path, 'rb') as f:
        test_X1 = topic_distribution(pickle.load(f))
    with open(lda_test_path, 'rb') as f:
        test_X2 = topic_distribution(pickle.load(f))
    with open(glda_test_path, 'rb') as f:
        test_X3 = topic_distribution(pickle.load(f))
    with open(senclu_test_path, 'rb') as f:
        test_X4 = pickle.load(f)
    with open(slda_test_path, 'rb') as f:
        test_X5 = topic_distribution(pickle.load(f))

    clf1 = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=1000).fit(train_X1, train_y)
    clf2 = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=1000).fit(train_X2, train_y)
    clf3 = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=1000).fit(train_X3, train_y)
    clf4 = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=1000).fit(train_X4, train_y)
    clf5 = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=1000).fit(train_X5, train_y)

    out1 = clf1.predict(test_X1)
    out2 = clf2.predict(test_X2)
    out3 = clf3.predict(test_X3)
    out4 = clf4.predict(test_X4)
    out5 = clf5.predict(test_X5)

    acc1 = accuracy_score(test_y, out1)
    acc2 = accuracy_score(test_y, out2)
    acc3 = accuracy_score(test_y, out3)
    acc4 = accuracy_score(test_y, out4)
    acc5 = accuracy_score(test_y, out5)

    f11_mac = f1_score(test_y, out1, average='macro')
    f12_mac = f1_score(test_y, out2, average='macro')
    f13_mac = f1_score(test_y, out3, average='macro')
    f14_mac = f1_score(test_y, out4, average='macro')
    f15_mac = f1_score(test_y, out5, average='macro')
    f11_mic = f1_score(test_y, out1, average='micro')
    f12_mic = f1_score(test_y, out2, average='micro')
    f13_mic = f1_score(test_y, out3, average='micro')
    f14_mic = f1_score(test_y, out4, average='micro')
    f15_mic = f1_score(test_y, out5, average='micro')

    acc_result = {"Contextual TM": round(acc1 * 100, 2),
                  "Blei LDA": round(acc2 * 100, 2),
                  "Gaussian LDA": round(acc3 * 100, 2),
                  "SenClu": round(acc4 * 100, 2),
                  "Sentence LDA": round(acc5 * 100, 2)}
    f1_result = {'macro': {"Contextual TM": round(f11_mac * 100, 2),
                           "Blei LDA": round(f12_mac * 100, 2),
                           "Gaussian LDA": round(f13_mac * 100, 2),
                           "SenClu": round(f14_mac * 100, 2),
                           "Sentence LDA": round(f15_mac * 100, 2)},
                 'micro': {"Contextual TM": round(f11_mic * 100, 2),
                           "Blei LDA": round(f12_mic * 100, 2),
                           "Gaussian LDA": round(f13_mic * 100, 2),
                           "SenClu": round(f14_mic * 100, 2),
                           "Sentence LDA": round(f15_mic * 100, 2)}}
    feature_importance = dict()
    for clf, name in zip([clf1, clf2, clf3, clf4], ["Contextual TM", "Blei LDA", "Gaussian LDA", "SenClu", "Sentence LDA"]):
        feature_importance[name] = dict()
        if len(TARGET[dataset][category]) == 2:
            feature_importance[name][list(label_map.keys())[0]] = clf.coef_.argsort().tolist()[0]
            feature_importance[name][list(label_map.keys())[1]] = clf.coef_.argsort()[:, ::-1].tolist()[0]
        else:
            for line, cat in zip(clf.coef_.argsort()[:, ::-1], label_map.keys()):
                feature_importance[name][cat] = line.tolist()
    return acc_result, f1_result, feature_importance


if __name__ == "__main__":
    result_path = Path(__file__).parents[0] / 'result'
    for it in range(5):
        os.makedirs(result_path / f'iter{it}/feature/', exist_ok=True)
        for dataset in ['20newsgroup', 'nyt']:
            for num_topics in [10, 20]:
                acc_result = dict()
                f1_result = dict()
                feature_importance = dict()
                for category in TARGET[dataset].keys():
                    acc_ret, f1_ret, feat = train(category, dataset, num_topics, it)
                    acc_result[category] = acc_ret
                    f1_result[category] = f1_ret
                    feature_importance[category] = feat
                write_json(acc_result, result_path / f"iter{it}/acc_{dataset}_{num_topics}topic.json")
                write_json(f1_result, result_path / f"iter{it}/f1_{dataset}_{num_topics}topic.json")
                write_json(feature_importance, result_path / f"iter{it}/feature/feat_{dataset}_{num_topics}topic.json")
