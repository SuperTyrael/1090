from code.calc_i2i import i2i_30k_sim
from tqdm import tqdm
import multiprocessing as mp
import time
import math
import pandas as pd
import numpy as np

def get_clicked_items(items):
    """
    Extract the set of article_ids that the user has clicked on.
    """
    return {art_id for art_id, _ in items}

def _calc_sim(dataset, articles_dic, cpu_cores, offline):
    """
    Calculate various similarity measures before performing recall.
    Currently only i2i_30k_sim is implemented.
    """
    num = len([i2i_30k_sim])  # Counting number of similarity functions involved
    start_time = time.time()
    print('Pre-recall calculations start ({} tasks)'.format(num))

    sims = {}
    sims['i2i_30k_sim'] = i2i_30k_sim(dataset, cpu_cores, offline)

    print('Pre-recall calculations end ({} seconds)'.format('%.2f' % (time.time() - start_time)))
    return sims

def _is_recall_target(last_clicked_timestamp, art_id, articles_dic, lag_hour_max=27, lag_hour_min=3):
    """
    Determine if an article should be considered for recall based on its creation timestamp
    relative to the user's last clicked timestamp. The article must be within 3 to 27 hours
    before the last_clicked_timestamp.
    """
    lag_max = lag_hour_max * 60 * 60 * 1000
    lag_min = lag_hour_min * 60 * 60 * 1000
    if articles_dic[art_id]['created_at_ts'] < (last_clicked_timestamp - lag_max):
        return False
    if articles_dic[art_id]['created_at_ts'] > (last_clicked_timestamp - lag_min):
        return False
    return True

def _recall_hot_items(dataset, train_dataset, test_users, articles_dic, topK=10):
    """
    Recall top-K hot items for each user/timestamp pair.
    Hot items are defined by their overall frequency in the dataset.
    Only consider items that fit the time constraints (3 to 27 hours before the last click).
    """
    result = {}
    start_time = time.time()
    lag_hour_min = 3
    lag_hour_max = 27

    # Count overall article frequencies
    hot_items = {}
    for _, items in tqdm(dataset.items()):
        for art_id, _ in items:
            hot_items.setdefault(art_id, 0)
            hot_items[art_id] += 1

    sorted_hot_items = sorted(hot_items.items(), key=lambda x: x[1], reverse=True)

    for user_id, ts_set in tqdm(test_users.items()):
        for last_clicked_timestamp in ts_set:
            user_items = train_dataset[user_id][last_clicked_timestamp]
            clicked_items = get_clicked_items(user_items)
            recommend_items = []

            for art_id, _ in sorted_hot_items:
                if art_id in clicked_items:
                    continue

                if not _is_recall_target(last_clicked_timestamp, art_id, articles_dic, lag_hour_min=lag_hour_min, lag_hour_max=lag_hour_max):
                    continue

                recommend_items.append(art_id)
                if len(recommend_items) >= topK:
                    break

            result.setdefault(user_id, {})
            result[user_id][last_clicked_timestamp] = recommend_items

    print('Hot item recall completed ({}s) range: [{}-{}]'.format('%.2f' % (time.time() - start_time), lag_hour_min, lag_hour_max))
    return result

def _recall_i2i_30k_sim_items(dataset, test_users, articles_dic, i2i_30k_sim, topK=25):
    """
    Recall top-K items based on i2i_30k similarity for each user/timestamp pair.
    Only consider items meeting the time constraints (0 to 27 hours before).
    Also, filter out items the user has already clicked.
    """
    result = {}
    start_time = time.time()
    lag_hour_min = 0
    lag_hour_max = 27

    for user_id, ts_set in tqdm(test_users.items()):
        for last_clicked_timestamp in ts_set:
            user_items = dataset[user_id][last_clicked_timestamp]
            clicked_items = get_clicked_items(user_items)
            recommend_items = {}

            for art_id, _ in user_items:
                if art_id not in i2i_30k_sim:
                    break

                # Retrieve candidate items from i2i_30k_sim
                candidate_list = i2i_30k_sim[art_id]['sorted_keys']
                for candidate_art_id in candidate_list:
                    if candidate_art_id in clicked_items:
                        continue

                    if not _is_recall_target(last_clicked_timestamp, art_id, articles_dic, lag_hour_min=lag_hour_min, lag_hour_max=lag_hour_max):
                        continue

                    # Only consider pairs with at least a similarity score of 2
                    if i2i_30k_sim[art_id]['related_arts'][candidate_art_id] < 2:
                        break

                    recommend_items.setdefault(candidate_art_id, 0)
                    recommend_items[candidate_art_id] += i2i_30k_sim[art_id]['related_arts'][candidate_art_id]

            result.setdefault(user_id, {})
            # Sort by score and keep topK
            result[user_id][last_clicked_timestamp] = [art_id for art_id, _ in sorted(recommend_items.items(), key=lambda x: x[1], reverse=True)[:topK]]

    print('i2i_30k_sim recall completed ({}s) range: [{}-{}]'.format('%.2f' % (time.time() - start_time), lag_hour_min, lag_hour_max))
    return result

def calc_and_recall(dataset, train_dataset, test_users, articles_dic, cpu_cores, offline, answers=None):
    """
    Main function to calculate similarities and perform recall steps:
    - Hot items recall
    - i2i_30k_sim recall

    If in offline mode and answers are provided, compute recall accuracy.
    """
    sims = _calc_sim(dataset, articles_dic, cpu_cores, offline)
    num = len([_recall_hot_items, _recall_i2i_30k_sim_items])
    start_time = time.time()
    print('Starting recall steps ({} tasks)'.format(num))

    recalls = {}
    recalls['hot'] = _recall_hot_items(dataset, train_dataset, test_users, articles_dic)
    recalls['i2i_30k_sim'] = _recall_i2i_30k_sim_items(train_dataset, test_users, articles_dic, sims['i2i_30k_sim'])

    if offline and answers is not None:
        test_users_count = np.sum([len(ts_list) for _, ts_list in test_users.items()])
        for recall_name, result in recalls.items():
            accuracy = 0
            # Determine the maximum length of recommendation lists for indexing
            max_list_len = np.max([len(items) for _, ts_dict in result.items() for _, items in ts_dict.items()])
            recall_counts = np.repeat(0, max_list_len)

            for user_id, ts_dict in result.items():
                for last_clicked_timestamp, items in ts_dict.items():
                    if answers[user_id][last_clicked_timestamp] in items:
                        accuracy += 1
                        recall_counts[items.index(answers[user_id][last_clicked_timestamp])] += 1

            print('Recall [{}]: Accuracy = {}%'.format(recall_name, '%.2f' % (accuracy * 100 / test_users_count)))
            print('Recall [{}]: Detailed hit counts: {}'.format(recall_name, recall_counts))

        # Calculate total accuracy across all recall methods
        total_accuracy = 0
        for user_id, ts_list in test_users.items():
            for last_clicked_timestamp in ts_list:
                for _, result in recalls.items():
                    if answers[user_id][last_clicked_timestamp] in result[user_id][last_clicked_timestamp]:
                        total_accuracy += 1
                        break

        print('Total combined recall accuracy = {}%'.format('%.2f' % (total_accuracy * 100 / test_users_count)))

    print('Recall steps completed ({}s)'.format('%.2f' % (time.time() - start_time)))
    return recalls
