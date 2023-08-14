import pandas as pd
from collections import defaultdict

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