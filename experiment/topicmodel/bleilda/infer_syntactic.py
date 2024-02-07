import pandas as pd
import gensim
from nltk.corpus import stopwords
from pathlib import Path
import re, os, pickle
import numpy as np


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


def proc_docs(docs):
    for n, doc in enumerate(docs):
        doc = doc.lower()
        doc = re.sub(r'[^a-z0-9 ]', '', doc)
        doc = doc.split(" ")
        doc = [i for i in doc if i not in STOPWORDS and i != '']
        docs[n] = doc
    return docs


def infer(category, dataset, num_topics, it):
    model_path = Path(__file__).parents[0] / f"params/iter{it}/{dataset}/{num_topics}topic/{category}"
    output_path = Path(__file__).parents[0] / f"syntactic/iter{it}/{dataset}/{num_topics}topic"
    data_path = Path(__file__).parents[1] / "data" / dataset
    os.makedirs(output_path, exist_ok=True)

    data = pd.read_csv(data_path / "syntactic.csv")
    data = proc_docs(data.loc[data["target_str"].isin(TARGET[dataset][category])]['data'].to_list())

    ldamodel = gensim.models.ldamodel.LdaModel.load(model_path.as_posix() + "/model.gensim")
    dictionary = ldamodel.id2word
    corpus = [dictionary.doc2bow(text) for text in data]

    output = ldamodel.get_document_topics(corpus)
    result = np.zeros((len(corpus), num_topics))
    for n, i in enumerate(output):
        for j in i:
            result[n][j[0]] = j[1]
    with open(output_path / f"{category}.pkl", 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    for it in range(1):
        for num_topics in [10, 20]:
            for dataset in ["nyt", "20newsgroup"]:
                for category in list(TARGET[dataset].keys()):
                    print(f"*******************{category}*********************")
                    infer(category, dataset, num_topics, it)
