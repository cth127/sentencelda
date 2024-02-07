from sentence_transformers import SentenceTransformer
from pathlib import Path
import argparse
from pysbd import Segmenter

from src.chol import GaussianLDATrainer


def train(num_topics, data_path):
    output_path = Path(__file__).parents[0] / f"params"

    encoder = SentenceTransformer('all-mpnet-base-v2', device="cuda")
    segmenter = Segmenter(language='en', clean=True)
    with open(data_path, 'r') as f:
        data = f.readlines()
    corpus = [segmenter.segment(i) for i in data]
    trainer = GaussianLDATrainer(corpus, encoder, num_topics, 0.1, 0.1, save_path=output_path)
    trainer.sample(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_topics", dest="num_topics", action="store", default=10)
    parser.add_argument("-d", "--data_path", dest="data_path", action="store", default="./data/sample.txt")
    args = parser.parse_args()
    train(int(args.num_topics), args.data_path)
