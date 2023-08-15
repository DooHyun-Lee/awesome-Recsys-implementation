import pandas as pd
from collections import defaultdict
import copy
import random
import numpy as np
import sys

def data_partition(dir, rating_filter = False):
    User = defaultdict(list)
    train, valid, test = {}, {}, {}
    # load and preprocess user/item data
    column_names = ['userid', 'movieid', 'rating', 'timestep']
    df = pd.read_csv(dir, sep='::', names=column_names)
    df.sort_values(by=['userid', 'timestep'])
    # only consider items with ratings >=3 
    if rating_filter:
        df = df[df['rating']>=3]
    
    for idx, row in df.iterrows():
        User[row['userid']].append(row['movieid'])

    for user in User:
        len_interaction = len(User[user])
        if len_interaction < 3:
            train[user] = User[user]
            valid[user] = []
            test[user] = []
        else:
            train[user] = User[user][:-2]
            valid[user] = []
            valid[user].append(User[user][-2])
            test[user] = []
            test[user].append(User[user][-1])
    user_total, item_total = df['userid'].max(), df['movieid'].max()
    return [train, valid, test, user_total, item_total]

def evaluate(model, dataset, args):
    [train, valid, test, user_total, item_total] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if user_total > 10000:
        users = random.sample(range(1, user_total + 1), 10000)
    else:
        users = range(1, user_total+1)
    for u in users:
        if len(train[u]) <1 or len(test[u]) <1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32) # this will contain train + val
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -=1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0) # don't know why adding zero? 
        item_idx = [test[u][0]]
        for _ in range(100): # adding random 100 items
            t = np.random.randint(1, item_total+1)
            while t in rated:
                t = np.random.randint(1, item_total+1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # result comes in [1, item_len] 

        # item_idx contains target item in 0 idx
        rank = predictions.argsort().argsort()[0].item() 
        valid_user += 1
        
        # calculate NDCG@10, HT@10
        if rank < 10:
            NDCG += 1/np.log2(rank+2)
            HT +=1
        if valid_user % 100 ==0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def evaluate_valid(model, dataset, args):
    [train, valid, test, user_total, item_total] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if user_total > 10000:
        users = random.sample(range(1, user_total+1), 10000)
    else:
        users = range(1, user_total+1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u])<1 :
            continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -=1
            if idx == -1:
                break
        
        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, item_total + 1)
            while t in rated:
                t = np.random.randint(1, item_total + 1)
                item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < 10:
            NDCG += 1/ np.log2(rank + 2)
            HT += 1
        if valid_user % 100 ==0 :
            print('.', end='')
            sys.stdout.flush()
    return NDCG / valid_user, HT / valid_user