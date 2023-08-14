import random
import numpy as np 
from multiprocessing import Process, Queue

# l, r defines range
# s is a set
def random_negative(l, r, s):
    idx = np.random.randint(l, r)
    while idx in s:
        idx = np.random.randint(l,r)
    return idx

def sample_function(train, user_total, item_total, batch_size, maxlen, result_queue):
    def sample():
        user = np.random.randint(1, user_total +1)
        # pick user until sequence bigger than 1
        # since pos is one step behind seq
        while len(train[user]) <= 1:
            user = np.random.randint(1, user_total + 1)

        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)
        nxt = train[user][-1]
        idx = maxlen - 1

        user_set = set(train[user])
        for i in reversed(train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            # if not padding item
            # select item among not in current set
            if nxt != 0:
                neg[idx] = random_negative(1, item_total +1, user_set)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return (user, seq, pos, neg)

    while True:
        batch = []
        for i in range(batch_size):
            batch.append(sample())
        result_queue.put(zip(*batch))

class WarpSampler():
    def __init__(self, train, user_total, item_total, batch_size = 128, maxlen=200, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(
                train,
                user_total,
                item_total,
                batch_size,
                maxlen, 
                self.result_queue
                ))
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

