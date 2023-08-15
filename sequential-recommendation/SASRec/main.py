import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random
import numpy as np
import time
from tqdm import tqdm

from utils import *
from sampler import *
from model import *

# comfortable path set for developing 
path = '/home/doolee13/recsysData/movieLens/ml-1m/ratings.dat'
parser = argparse.ArgumentParser()
parser.add_argument('--directory', default= path)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--learning-rate', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--num-blocks', default=200, type=int)
parser.add_argument('--seed', default=42)
parser.add_argument('--hidden-units', default=50, type=int)
parser.add_argument('--dropout-rate', default=0.2, type=float)
parser.add_argument('--num-heads', default=1, type=int)
parser.add_argument('--num-epochs', default=201, type=int)
parser.add_argument('--l2-emb', default=0.0, type=float)

args = parser.parse_args()

if __name__ == '__main__':
    dataset = data_partition(args.directory)
    [train, valid, test, user_total, item_total] = dataset
    num_batch = len(train) // args.batch_size

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sampler = WarpSampler(train, user_total, item_total, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    # weight init done internally 
    model = SASRec(user_total, item_total, device, args).to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in range(1, args.num_epochs + 1):
        for step in tqdm(range(num_batch)):
            user, seq, pos, neg = sampler.next_batch()
            user, seq, pos, neg = np.array(user), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(user, seq, pos, neg)
            pos_y= torch.ones(pos_logits.shape, device=device)
            neg_y= torch.zeros(neg_logits.shape, device=device)
            
            optimizer.zero_grad()
            indices = np.where(pos !=0)
            loss = criterion(pos_logits[indices], pos_y[indices]) 
            loss += criterion(neg_logits[indices], neg_y[indices])
            # l2 regularization for item embedding table necessary? 
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()
            print(f'loss in epoch {epoch} iteration {step} : {loss.item()}')
        
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')            
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            
            t0 = time.time()
            model.train()
    
    sampler.close()
    print('Done!')