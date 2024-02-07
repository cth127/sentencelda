import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path
import os, sys, tqdm, pickle
import parmap

sys.path.insert(0, Path(__file__).parents[1].as_posix())
from sentence_gaussianlda import GaussianLDA


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


def infer(category, dataset, num_topics, it):
    model_path = Path(__file__).parents[0] / f"params/iter{it}/{dataset}/{num_topics}topic/{category}"
    output_path = Path(__file__).parents[0] / f"lexical/iter{it}/{dataset}/{num_topics}topic"
    data_path = Path(__file__).parents[1] / "data" / dataset
    os.makedirs(output_path, exist_ok=True)

    encoder = SentenceTransformer('all-mpnet-base-v2', device="cuda")

    data = pd.read_csv(data_path / "lexical.csv")
    data = data.loc[data["target_str"].isin(TARGET[dataset][category])]
    corpus = [i.split(" / ") for i in data["data"]]

    model = GaussianLDA.load(encoder, model_path)
    output = np.zeros((len(corpus), num_topics))

    for n, i in enumerate(tqdm.tqdm(corpus)):
        topics = model.sample(i, 20)
        for j in topics:
            output[n, j] += 1
    with open(output_path / f"{category}.pkl", "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    for it in range(1):
        for dataset in ["20newsgroup", "nyt"]:
            for num_topics in [10, 20]:
                for category in list(TARGET[dataset].keys()):
                    print(f"*******************{category}*********************")
                    infer(category, dataset, num_topics, it)
