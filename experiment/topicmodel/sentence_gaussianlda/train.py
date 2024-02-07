from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, Path(__file__).parents[1].as_posix())
from sentence_gaussianlda import GaussianLDATrainer


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
    output_path = Path(__file__).parents[0] / f"params/iter{it}/{dataset}/{num_topics}topic/{category}"
    data_path = Path(__file__).parents[1] / "data" / dataset

    encoder = SentenceTransformer('all-mpnet-base-v2', device="cuda")
    data = pd.read_csv(data_path / "train.csv")
    data = data.loc[data["target_str"].isin(TARGET[dataset][category])]
    corpus = [i.split(" / ") for i in data["data"]]

    trainer = GaussianLDATrainer(corpus, encoder, num_topics, 0.1, 0.1, save_path=output_path)
    trainer.sample(20)


if __name__ == "__main__":
    for it in range(5):
        for dataset in ["20newsgroup", "nyt"]:
            for num_topics in [10, 20]:
                for category in list(TARGET[dataset].keys()):
                    print(f"*******************{category}*********************")
                    train(category, dataset, num_topics, it)
