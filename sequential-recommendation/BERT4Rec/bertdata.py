import torch 
import random

from samplers import *

class BertDataloader():
    def __init__(self, args, dataset):
        # dataset : MovieLensDataset instance 
        dataset_dict = dataset.load_dataset()
        self.train = dataset_dict['train'] 
        self.val = dataset_dict['val']
        self.test = dataset_dict['test']
        self.user_dict = dataset_dict['user_dict']
        self.item_dict = dataset_dict['item_dict']
        self.user_count = len(self.user_dict)
        self.num_items = len(self.item_dict)

        self.args = args
        seed = args.dataloader_random_seed
        self.max_len = args.max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.num_items + 1
        code = args.negative_sampler_code
        # args.train_negative_sample_size
        # args.train_negative_sampling_seed
        # args.test_negative_sample_size
        # args.test_negative_sampling_seed
        # args.train_batch_size
        # args.val_batch_size
        # args.test_batch_size
        self.rng = random.Random(seed)
        
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.num_items, args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed)
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.num_items, args.test_negative_sample_size,
                                                          args.test_negative_sampling_seed)
        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_eval_loader(mode='val')
        test_loader = self._get_eval_loader(mode='test')
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                                 shuffle = True, pin_memory = True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.num_items, self.rng)
        return dataset
    
    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset

class BertTrainDataset(torch.utils.data.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        # u2seq : self.train (dictionary)
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                # 80% mask, 10% random, 10% original
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0) # we don't need to predict

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

class BertEvalDataset(torch.utils.data.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples):
        # u2seq : self.train, u2answer : self.val or self.test
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples 

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len : ]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)