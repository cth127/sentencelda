from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from pathlib import Path
import os, sys, pickle
from nltk.corpus import stopwords

sys.path.insert(0, Path(__file__).parents[1].as_posix())
from ctm.util import WhiteSpacePreprocessingStopwords

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
PATH_NAME = {10: 0.9, 20: 0.95}


def infer(category, dataset, num_topics, it):
    name = f"contextualized_topic_model_nc_{num_topics}_tpm_0.0_tpv_{PATH_NAME[num_topics]}_hs_prodLDA_ac_(100, 100)_do_softplus_lr_0.2_mo_0.002_rp_0.99"
    model_path = Path(__file__).parents[0] / f"params/iter{it}/{dataset}/{num_topics}topic/{category}/{name}"
    tp_path = Path(__file__).parents[0] / f"params/iter{it}/{dataset}/{num_topics}topic/{category}"
    output_path = Path(__file__).parents[0] / f"syntactic/iter{it}/{dataset}/{num_topics}topic"
    data_path = Path(__file__).parents[1] / "data" / dataset
    os.makedirs(output_path, exist_ok=True)

    data = pd.read_csv(data_path / "syntactic.csv")
    data = data.loc[data["target_str"].isin(TARGET[dataset][category])]
    corpus = [i.replace(" / ", " ") for i in data["data"]]

    with open(tp_path / "tp.pkl", "rb") as f:
        tp = pickle.load(f)
    stop_words = stopwords.words('english')
    sp = WhiteSpacePreprocessingStopwords(corpus, stopwords_list=stop_words)
    preproc_docs, unpreproc_docs, vocab, retained_indices = sp.preprocess()

    test_set = tp.transform(text_for_contextual=unpreproc_docs, text_for_bow=preproc_docs)
    ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=num_topics, num_epochs=20)
    ctm.load(model_dir=model_path, epoch=19)
    output = ctm.get_doc_topic_distribution(test_set, n_samples=20)

    with open(output_path / f"{category}.pkl", "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    for it in range(1):
        for dataset in ["nyt", "20newsgroup"]:
            for num_topics in [10, 20]:
                for category in list(TARGET[dataset].keys()):
                    print(f"*******************{category}*********************")
                    infer(category, dataset, num_topics, it)
