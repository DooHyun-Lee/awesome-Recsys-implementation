import torch.nn as nn
import pandas as pd
import numpy as np

class MovieLensDataset(nn.Module):
    def __init__(self, args):
        super(MovieLensDataset, self).__init__()

        self.path = args.directory
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc # user interaction threshold
        self.min_sc = args.min_sc # item interaction threshold
        self.split = args.split
        self.split_seed = args.split_seed
        self.eval_set_size = args.eval_set_size


    def load_ratigs_df(self):
        column_names = ['userid', 'movieid', 'rating', 'timestep']
        df = pd.read_csv(self.path, sep='::', names=column_names)
        df.sort_values(by=['userid', 'timestep'])
        return df

    # select rows only with ratings higher than min rating
    def make_implicit(self, df):
        print('turning into implicit ratings')
        df = df[df['rating'] >= self.min_rating]
        return df

    # select rows only with user and items that has interacted over threshold
    def filter_triplets(self, df):
        print('filtering triplets')
        if self.min_sc >0 :
            item_sizes = df.groupby('movieid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['movieid'].isin(good_items)]
        
        if self.min_uc >0 :
            user_sizes = df.groupby('userid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['userid'].isin(good_users)]

        return df

    # create user-idx, item-idx and densify into idx
    def densify_index(self, df):
        print('densifying index')
        user_dict = {u:i for i, u in enumerate(set(df['userid']))}
        item_dict = {u:i for i, u in enumerate(set(df['movieid']))}
        df['userid'] = df['userid'].map(user_dict)
        df['movieid'] = df['movieid'].map(item_dict)
        return df, user_dict, item_dict

    # split data into train, val, test
    def split_df(self, df, user_count):
        if self.split == 'leave_one_out':
            print('splitting')
            user_group = df.groupby('userid')
            user2items = user_group.apply(lambda d: list(d.sort_values(by='timestep')['movieid']))
            train, val, test = {}, {}, {}
            for user in range(user_count):
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        elif self.split == 'holdout':
            print('splitting')
            np.random.seed(self.split_seed)
            eval_set_size = self.eval_set_size

            permuted_idx = np.random.permutation(user_count)
            train_user_idx = permuted_idx[:-2*eval_set_size]
            val_user_idx = permuted_idx[-2*eval_set_size : -eval_set_size]
            test_user_idx = permuted_idx[-eval_set_size: ]

            # split dataframes
            train_df = df.loc[df['userid'].isin(train_user_idx)]
            val_df = df.loc[df['userid'].isin(val_user_idx)]
            test_df = df.loc[df['userid'].isin(test_user_idx)]

            # df to dict
            train = dict(train_df.groupby('userid').apply(lambda d: list(d['movieid'])))
            val = dict(val_df.groupby('userid').apply(lambda d: list(d['movieid'])))
            test = dict(test_df.groupby('userid').apply(lambda d: list(d['movieid'])))
            
            return train, val, test
        else:
            raise NotImplementedError

    def preprocess(self):
        df = self.load_ratigs_df()
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df, user_dict, item_dict = self.densify_index(df)
        train, val, test = self.split_df(df, len(user_dict))
        dataset = {'train' : train, 
                   'val' : val, 
                   'test' : test, 
                   'user_dict' : user_dict, 
                   'item_dict' : item_dict} 
        return dataset

    # called by DataLoader 
    def load_dataset(self):
        dataset = self.preprocess()
        return dataset

