import numpy as np

import utils

if __name__ == '__main__':
    seed = 42
    utils.set_seed(seed)
    data = utils.jload("./data/sharegpt_V3_format_with_real_response.jsonl")
    
    # random sample 40k
    N = 50000
    data_mask = np.random.choice(len(data), N, replace=False)
    data = [data[i] for i in data_mask]
    data = data[:N]
    data_train = data[:40000]
    data_train = utils.jsort(data_train, key="id", integer=True)
    data_val = data[40000:]
    data_val = utils.jsort(data_val, key="id", integer=True)

    # save to json
    utils.jdump(data_train, "./data/sharegpt-train-40k.json")
    utils.jdump(data_val, "./data/sharegpt-val-10k.json")
