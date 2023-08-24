import argparse
from data import MovieLensDataset
from bertdata import *

import ipdb

# MovieLensDataset
path = '/home/doolee13/recsysData/movieLens/ml-1m/ratings.dat'
parser = argparse.ArgumentParser()
parser.add_argument('--directory', default=path)
parser.add_argument('--min-rating', default=4, type=int)
parser.add_argument('--min-uc', default=5, type=int)
parser.add_argument('--min-sc', default=0, type=int)
parser.add_argument('--split', default='leave_one_out', type=str)
parser.add_argument('--split-seed', default=42, type=int)
parser.add_argument('--eval-set-size', default=500, type=int)

# BertDataLoader
parser.add_argument('--dataloader-random-seed', default=0.0, type=float)
parser.add_argument('--negative-sampler-code', default='random', type=str)
parser.add_argument('--train-negative-sample-size', default=0, type=int)
parser.add_argument('--train-negative-sampling-seed', default=0, type=int)
parser.add_argument('--test-negative-sample-size', default=100, type=int)
parser.add_argument('--test-negative-sampling-seed', default=42, type=int)
parser.add_argument('--train-batch-size', default=128, type=int)
parser.add_argument('--val-batch-size', default=128, type=int)
parser.add_argument('--test-batch-size', default=128, type=int)

# BertModel
parser.add_argument('--max-len', default=100, type=int)
parser.add_argument('--bert-mask-prob', default=0.15, type=float)

args = parser.parse_args()

if __name__ == '__main__':
    dataset = MovieLensDataset(args)
    dataloader = BertDataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    