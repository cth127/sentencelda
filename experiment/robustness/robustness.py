from pathlib import Path
import pickle
import json
from scipy.stats import rankdata, kendalltau

import numpy as np


MODEL = ['bleilda', 'gaussianlda', 'ctm', 'sentence_gaussianlda']
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


def compute_distribution(array):
    for n, i in enumerate(array):
        if sum(i) != 0:
            array[n] /= sum(i)
    return array


def compute_stats(distributions1, distributions2):
    stats = list()
    diffs = list()
    for dist1, dist2 in zip(distributions1, distributions2):
        stat, _ = kendalltau(rankdata(dist1, method='min'), rankdata(dist2, method='min'))
        diff = sum(abs(dist1 - dist2)) / 2
        if not np.isnan(stat):
            stats.append(stat)
        diffs.append(diff)
    return stats, diffs


def compute(model, dataset, category, num_topics, mode):
    path1 = Path(__file__).parents[1] / f'topicmodel/{model}/infer/iter0/{dataset}/{num_topics}topic'
    path2 = Path(__file__).parents[1] / f'topicmodel/{model}/{mode}/iter0/{dataset}/{num_topics}topic'
    with open(path1 / f'{category}.pkl', 'rb') as f:
        distribution1 = compute_distribution(pickle.load(f))
    with open(path2 / f'{category}.pkl', 'rb') as f:
        distribution2 = compute_distribution(pickle.load(f))
    stats, diffs = compute_stats(distribution1, distribution2)
    return stats, diffs


def compare(mode):
    output_path = Path(__file__).parents[0]
    result = dict()
    for dataset, v in TARGET.items():
        result[dataset] = dict()
        for num_topics in [10, 20]:
            result[dataset][f'{num_topics} topics'] = dict()
            for model in MODEL:
                stats_list = list()
                differences_list = list()
                for category in v.keys():
                    stats, diffs = compute(model, dataset, category, num_topics, mode)
                    differences_list.extend(diffs)
                    stats_list.extend(stats)
                line = f"""tau: {round(np.mean(stats_list), 4)} (± {round(np.std(stats_list), 4)})
diff: {round(np.mean(differences_list), 4)} (± {round(np.std(differences_list), 4)})"""
                result[dataset][f'{num_topics} topics'][model] = line
    write_json(result, output_path / f'compare_{mode}.json')


if __name__ == "__main__":
    for mode in ["syntactic", "lexical"]:
        print(mode)
        compare(mode)
