import numpy as np
import pandas as pd
import time
import multiprocessing as mp
from tqdm import tqdm
import pickle
import math
from os.path import isfile
from const import CACHE_FOLDER

def _i2i_30k_sim_core(job_id, user_id_list, dataset):
    """
    Core function to calculate item-to-item similarity based on intervals of 30k (30,000 milliseconds)
    for a subset of users. This function is executed in parallel processes.

    :param job_id: the ID of the current subprocess job
    :param user_id_list: list of user IDs to process in this subprocess
    :param dataset: dictionary of user_id -> [(article_id, timestamp), ...]
    :return: a dictionary of item-to-item similarity results for the given user subset
    """
    _item_counts_dic = {}
    _i2i_30k_sim = {}

    start_time = time.time()
    for user_id in user_id_list:
        item_dt = dataset[user_id]
        ts_list = pd.Series([ts for _, ts in item_dt])
        # Identify indices where timestamp difference from previous item is exactly 30000
        idx_list = [idx for idx, val in dict(ts_list - ts_list.shift(1) == 30000).items() if val]

        for idx in idx_list:
            i_art_id, _ = item_dt[idx]
            j_art_id, _ = item_dt[idx - 1]

            # Update similarity counts for both directions: i->j and j->i
            _i2i_30k_sim.setdefault(i_art_id, {})
            _i2i_30k_sim[i_art_id].setdefault(j_art_id, 0)
            _i2i_30k_sim[i_art_id][j_art_id] += 1

            _i2i_30k_sim.setdefault(j_art_id, {})
            _i2i_30k_sim[j_art_id].setdefault(i_art_id, 0)
            _i2i_30k_sim[j_art_id][i_art_id] += 1

    print('Subtask[{}]: i2i_30k similarity calculation completed. ({} seconds)'.format(
        job_id, '%.2f' % (time.time() - start_time))
    )

    return _i2i_30k_sim

def i2i_30k_sim(dataset, n_cpu, offline, max_related=50):
    """
    Calculate item-to-item similarity based on a 30k interval pattern across the entire dataset.
    Utilizes parallel processing. If results are cached, load from file instead of recalculating.

    :param dataset: dictionary of user_id -> [(article_id, timestamp), ...]
    :param n_cpu: number of CPU cores to use for parallel processing
    :param offline: boolean indicating if offline mode is used
    :param max_related: maximum number of related items to consider (default 50)
    :return: i2i similarity dictionary {item_id: {'sorted_keys': [...], 'related_arts': {...}}}
    """
    filename = 'i2i_30k_sim_{}.pkl'.format('offline' if offline else 'online')
    if isfile(CACHE_FOLDER + filename):
        print('Loading precomputed i2i_30k similarity from file {}'.format(filename))
        return pickle.load(open(CACHE_FOLDER + filename, 'rb'))

    # Compute similarity
    start_time = time.time()
    print('Starting i2i_30k similarity calculation')
    i2i_sim_3k = {}
    n_block = (len(dataset.keys()) - 1) // n_cpu + 1
    keys = list(dataset.keys())
    pool = mp.Pool(processes=n_cpu)
    results = [
        pool.apply_async(_i2i_30k_sim_core, args=(i, keys[i * n_block:(i + 1) * n_block], dataset))
        for i in range(0, n_cpu)
    ]
    pool.close()
    pool.join()

    # Merge results from all subprocesses
    for result in results:
        _i2i_sim_3k = result.get()

        for art_id, related_art_id_dic in _i2i_sim_3k.items():
            i2i_sim_3k.setdefault(art_id, {})
            for related_art_id, value in related_art_id_dic.items():
                i2i_sim_3k[art_id].setdefault(related_art_id, 0)
                i2i_sim_3k[art_id][related_art_id] += value

    print('Sorting in descending order')
    for art_id, related_arts in tqdm(i2i_sim_3k.items()):
        # Sort related items by similarity values in descending order
        sorted_and_topK = sorted(related_arts.items(), key=lambda x: x[1], reverse=True)
        i2i_sim_3k[art_id] = {
            'sorted_keys': [art_id for art_id, _ in sorted_and_topK],
            'related_arts': dict(sorted_and_topK)
        }

    print('i2i_30k similarity calculation completed ({} seconds)'.format('%.2f' % (time.time() - start_time)))
    print('Saving i2i_30k similarity data to file {}'.format(filename))
    pickle.dump(i2i_sim_3k, open(CACHE_FOLDER + filename, 'wb'))
    return i2i_sim_3k
