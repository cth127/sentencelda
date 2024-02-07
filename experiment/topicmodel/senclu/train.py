from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from pathlib import Path
import sys
import os
import pickle

sys.path.insert(0, Path(__file__).parents[1].as_posix())
from senclu import SenClu


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


def train(category, dataset, num_topics, it):
    train_output_path = Path(__file__).parents[0] / f"params/iter{it}/{dataset}/{num_topics}topic"
    test_output_path = Path(__file__).parents[0] / f"infer/iter{it}/{dataset}/{num_topics}topic"
    data_path = Path(__file__).parents[1] / "data" / dataset
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)

    train_data = pd.read_csv(data_path / "train.csv")
    test_data = pd.read_csv(data_path / "test.csv")
    train_data = train_data.loc[train_data["target_str"].isin(TARGET[dataset][category])]
    train_corpus = [i.split(" / ") for i in train_data["data"]]
    test_data = test_data.loc[test_data["target_str"].isin(TARGET[dataset][category])]
    test_corpus = [i.split(" / ") for i in test_data["data"]]

    folder = f"./tmp/{dataset}/{category}/"
    trainer = SenClu()
    train_res, test_res = trainer.fit_transform(train_docs=train_corpus, test_docs=test_corpus, nTopics=num_topics, loadAndStoreInFolder=folder)

    with open(train_output_path / f"{category}.pkl", "wb") as f:
        pickle.dump(train_res, f)
    with open(test_output_path / f"{category}.pkl", "wb") as f:
        pickle.dump(test_res, f)


if __name__ == "__main__":
    for it in range(5):
        for dataset in ["20newsgroup", "nyt"]:
            for num_topics in [10, 20]:
                for category in list(TARGET[dataset].keys()):
                    print(f"*******************{category}*********************")
                    train(category, dataset, num_topics, it)
