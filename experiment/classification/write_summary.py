import numpy as np
from pathlib import Path
import json
import copy


TARGET = {"20newsgroup": ["computer", "ride", "sports", "science", "religion", "politics"],
          "nyt": ["arts", "business", "politics", "science", "sports"]}

TARGET2 = {"20newsgroup": {"computer": ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'],
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

MODEL = ["Blei LDA", "Gaussian LDA", "Contextual TM", "SenClu", "Sentence LDA"]


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def write_summary(metric, dataset, topics):
    for i in range(5):
        if metric == "acc":
            file_path = Path(__file__).parents[0] / f"result/iter{i}/{metric}_{dataset}_{topics}topic.json"
        else:
            file_path = Path(__file__).parents[0] / f"result/iter{i}/f1_{dataset}_{topics}topic.json"
        file = load_json(file_path)
        if i == 0:
            tmp = {key: list() for key in MODEL}
            results = {key: copy.deepcopy(tmp) for key in TARGET[dataset]}
        for k, v in file.items():
            if metric == "acc":
                for k2, v2 in v.items():
                    results[k][k2].append(v2)
            elif metric == "f1mac":
                for k2, v2 in v["macro"].items():
                    results[k][k2].append(v2)
            elif metric == "f1mic":
                for k2, v2 in v["micro"].items():
                    results[k][k2].append(v2)
    for k, v in results.items():
        means = list()
        for k2, v2 in v.items():
            results[k][k2] = '{:.2f}'.format(round(np.mean(v2), 2)) + "\% {\std(" + '{:.2f}'.format(round(np.std(v2), 2)) + ")}"
            means.append(round(np.mean(v2), 2))
        max_mean = max(means)
        second = means[np.argsort(means)[-2]]
        for k2, v2 in results[k].items():
            if str(max_mean) in v2:
                results[k][k2] = r'\textbf{' + v2 + '}'
            elif str(second) in v2:
                results[k][k2] = r'\underline{' + v2 + '}'
    for category in TARGET[dataset]:
        print("& " + category.capitalize() + f" ({len(TARGET2[dataset][category])})" + " & " + " & ".join(results[category].values()) + " \\\\")


if __name__ == "__main__":
    for topic in [10, 20]:
        for dataset in ["20newsgroup", "nyt"]:
            print(topic, dataset)
            write_summary("acc", dataset, topic)
