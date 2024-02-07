import numpy as np
import pandas as pd
import gensim.downloader
from nltk.corpus import stopwords
from pathlib import Path
import re, os, sys, tqdm, pickle

sys.path.insert(0, Path(__file__).parents[1].as_posix())
from gaussianlda import GaussianLDA


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


STOPWORDS = stopwords.words('english')


def proc_docs(docs, vocab):
    for n, doc in enumerate(docs):
        doc = doc.lower()
        doc = re.sub(r'[^a-z0-9 ]', '', doc)
        doc = doc.split(" ")
        doc = [vocab[i] for i in doc if i in vocab.keys() and i not in STOPWORDS]
        docs[n] = doc
    return docs


def infer(word2vec, category, dataset, num_topics, it):
    model_path = Path(__file__).parents[0] / f"params/iter{it}/{dataset}/{num_topics}topic/{category}"
    output_path = Path(__file__).parents[0] / f"infer/iter{it}/{dataset}/{num_topics}topic"
    data_path = Path(__file__).parents[1] / "data" / dataset
    os.makedirs(output_path, exist_ok=True)

    data = pd.read_csv(data_path / "test.csv")
    data = data.loc[data["target_str"].isin(TARGET[dataset][category])]['data'].to_list()
    vocab_dict = word2vec.key_to_index
    vocab_list = list(vocab_dict.keys())
    embeddings = word2vec.vectors
    corpus = proc_docs(data, vocab_dict)
    model = GaussianLDA.load(embeddings, vocab_list, model_path)

    output = np.zeros((len(corpus), num_topics))
    for n, i in enumerate(tqdm.tqdm(corpus)):
        topics = model.sample(i, 20)
        for j in topics:
            output[n, j] += 1
    with open(output_path / f"{category}.pkl", "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    for it in range(5):
        for dataset in ["20newsgroup", "nyt"]:
            for num_topics in [10, 20]:
                word2vec = gensim.downloader.load('word2vec-google-news-300')
                for category in list(TARGET[dataset].keys()):
                    print(f"*******************{category}*********************")
                    infer(word2vec, category, dataset, num_topics, it)
