## File1 ##
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from const import CACHE_FOLDER

def neg_sampling(ds, min=1, max=5):
    """
    Perform negative sampling on the given dataset to balance positive and negative samples.
    """
    start_time = time.time()
    pos_ds = ds.loc[ds['answer'] == 1]
    neg_ds = ds.loc[ds['answer'] == 0]

    def _neg_sampling_func(x):
        # Determine the number of negative samples to pick based on given min/max constraints
        n_sampling = len(x)
        n_sampling = min if n_sampling < min else (max if n_sampling > max else n_sampling)
        return x.sample(n=n_sampling, replace=False)

    # Apply negative sampling by user and by article, then remove duplicates
    neg_ds = pd.concat([
        neg_ds.groupby(['user_id', 'last_clicked_timestamp']).apply(_neg_sampling_func),
        neg_ds.groupby('article_id').apply(_neg_sampling_func),
    ]).drop_duplicates()

    # Combine positive and sampled negative samples
    ret = pd.concat([pos_ds, neg_ds]).reset_index(drop=True)
    print('Negative sampling completed ({}s, {} -> {} samples)'.format(
        '%.2f' % (time.time() - start_time), len(ds), len(ret)
    ))
    return ret

def get_user_features(raw_data, train_dataset, test_users, articles_dic):
    """
    Compute various user-level features based on their interaction history and article metadata.
    """

    def calc_avg_words_count(items):
        return np.average([articles_dic[item[0]]['words_count'] for item in items])

    def calc_min_words_count(items):
        return np.min([articles_dic[item[0]]['words_count'] for item in items])

    def calc_max_words_count(items):
        return np.max([articles_dic[item[0]]['words_count'] for item in items])

    def calc_lag_between_created_at_ts_and_clicked_ts(items, articles_dic):
        # Time lag between the creation of the last clicked article and the click timestamp
        item = items[-1]
        return (item[1] - articles_dic[item[0]]['created_at_ts']) / (1000 * 60 * 60 * 24)

    def calc_lag_between_two_click(items):
        # Time gap between the last two clicked articles
        if len(items) > 1:
            return (items[-1][1] - items[-2][1]) / (1000 * 60 * 60 * 24)
        else:
            return np.nan

    def calc_lag_between_two_articles(items, articles_dic):
        # Time gap between the creation of the last two clicked articles
        if len(items) > 1:
            return (articles_dic[items[-1][0]]['created_at_ts'] - articles_dic[items[-2][0]]['created_at_ts']) / (1000 * 60 * 60 * 24)
        else:
            return np.nan

    df_users = pd.DataFrame(list(test_users.keys()), columns=['user_id'])

    # Compute user-level statistics
    _data = []
    for user_id, ts_set in tqdm(test_users.items()):
        for last_clicked_timestamp in ts_set:
            _data.append((
                user_id,
                last_clicked_timestamp,
                calc_avg_words_count(train_dataset[user_id][last_clicked_timestamp]),
                calc_min_words_count(train_dataset[user_id][last_clicked_timestamp]),
                calc_max_words_count(train_dataset[user_id][last_clicked_timestamp]),
                calc_lag_between_created_at_ts_and_clicked_ts(train_dataset[user_id][last_clicked_timestamp], articles_dic),
                calc_lag_between_two_click(train_dataset[user_id][last_clicked_timestamp]),
                calc_lag_between_two_articles(train_dataset[user_id][last_clicked_timestamp], articles_dic),
            ))

    df1 = pd.DataFrame(_data, columns=[
        'user_id', 'last_clicked_timestamp',
        'avg_words_count', 'min_words_count', 'max_words_count',
        'lag_between_created_at_ts_and_clicked_ts',
        'lag_between_two_click', 'lag_between_two_articles'
    ])

    # Compute the mode (most frequent category) for environment/device attributes
    columns = [
        'user_id', 'click_environment', 'click_deviceGroup',
        'click_os', 'click_country', 'click_region', 'click_referrer_type'
    ]
    df2 = df_users.merge(raw_data.get_all_click_log())[columns].groupby('user_id').agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()

    return df1.merge(df2)

def create_train_data(raw_data, train_dataset, test_users, articles_dic, recall_results, offline, y_answer):
    """
    Create training (or test) dataset by merging recall results, user features, and article metadata.
    """
    start_time = time.time()
    keys_ds = []

    # For each user and each timestamp in test_users, retrieve recommended articles from all recall methods
    for user_id, ts_set in test_users.items():
        for last_clicked_timestamp in ts_set:
            items = np.concatenate([result[user_id][last_clicked_timestamp] for _, result in recall_results.items()])
            keys_ds.append(list(zip(
                np.repeat(user_id, len(items)),
                np.repeat(last_clicked_timestamp, len(items)),
                items
            )))

    ds = pd.DataFrame(np.concatenate(keys_ds), columns=['user_id', 'last_clicked_timestamp', 'article_id'], dtype=np.int64).drop_duplicates()

    # If offline mode, merge with correct answers and perform negative sampling
    if offline:
        answer_keys_ds = []
        for user_id, ts_list in y_answer.items():
            for last_clicked_timestamp, art_id in ts_list.items():
                answer_keys_ds.append((user_id, last_clicked_timestamp, art_id))

        answers = pd.DataFrame(answer_keys_ds, columns=['user_id', 'last_clicked_timestamp', 'article_id'], dtype=np.int64)
        answers['answer'] = 1
        ds = ds.merge(answers, how='left').fillna({'answer': 0})
        ds['answer'] = ds['answer'].astype(np.int8)

        ds = neg_sampling(ds)

    # Merge with article features and user features
    ds = ds.merge(raw_data.get_articles()).merge(get_user_features(raw_data, train_dataset, test_users, articles_dic))

    # Add new features
    ds['lag_period_last_article'] = ds['last_clicked_timestamp'] - ds['created_at_ts']
    ds['diff_words_last_article'] = ds['avg_words_count'] - ds['words_count']

    ds.to_csv(CACHE_FOLDER + '{}.csv'.format('train' if offline else 'test'), index=False)
    print('{} CSV file generated ({}s, {} samples)'.format(
        'Train' if offline else 'Test',
        '%.2f' % (time.time() - start_time), len(ds)
    ))


## File2 ##
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import time
from os.path import isfile
from const import CACHE_FOLDER

class DataHolder:
    """
    DataHolder class manages loading and preprocessing of user click logs, articles, and trains the dataset needed for modeling.
    """
    def __init__(self, articles, train_click_log, test_click_log, trainB_click_log=None):
        self.articles = articles
        self.train_click_log = train_click_log
        self.test_click_log = test_click_log
        self.trainB_click_log = trainB_click_log

        # Combine logs from train, trainB (if available), and test
        if self.trainB_click_log is not None:
            self.all_click_log = pd.concat([self.train_click_log, self.trainB_click_log, self.test_click_log], ignore_index=True)
        else:
            self.all_click_log = pd.concat([self.train_click_log, self.test_click_log], ignore_index=True)

        self.all_click_log.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'], inplace=True)

        print('Loaded from train_click_log: {} (UserId=[{},{}])'.format(
            len(self.train_click_log), self.train_click_log['user_id'].min(), self.train_click_log['user_id'].max()
        ))
        print('Loaded from test_click_log: {} (UserId=[{},{}])'.format(
            len(self.test_click_log), self.test_click_log['user_id'].min(), self.test_click_log['user_id'].max()
        ))

        if self.trainB_click_log is not None:
            print('Loaded from trainB_click_log: {} (UserId=[{},{}])'.format(
                len(self.trainB_click_log), self.trainB_click_log['user_id'].min(), self.trainB_click_log['user_id'].max()
            ))

        print('Using all_click_log: {} records (UserId=[{},{}])'.format(
            len(self.all_click_log),
            self.all_click_log['user_id'].min(), self.all_click_log['user_id'].max()
        ))

        # Convert DataFrame into a dictionary {user_id: [(article_id, timestamp), ...]}
        filename = 'dataset.pkl'
        if isfile(CACHE_FOLDER + filename):
            print('Loading dataset from file {}'.format(filename))
            self.dataset = pickle.load(open(CACHE_FOLDER + filename, 'rb'))
        else:
            start_time = time.time()
            _t = (self.all_click_log.sort_values('click_timestamp')
                                .groupby('user_id')
                                .apply(lambda x: list(zip(x['click_article_id'], x['click_timestamp'])))
                                .reset_index()
                                .rename(columns={0: 'item_dt_list'}))

            self.dataset = dict(zip(_t['user_id'], _t['item_dt_list']))
            print('Dataset creation completed ({}s)'.format('%.2f' % (time.time() - start_time)))
            print('Saving dataset to file {}'.format(filename))
            pickle.dump(self.dataset, open(CACHE_FOLDER + filename, 'wb'))

        # Generate a dictionary of {user_id: [timestamps]} to facilitate training
        filename = 'train_users_dic.pkl'
        if isfile(CACHE_FOLDER + filename):
            print('Loading train_users_dic from file {}'.format(filename))
            self.train_users_dic = pickle.load(open(CACHE_FOLDER + filename, 'rb'))
        else:
            start_time = time.time()
            self.train_users_dic = {}
            for user_id, items in tqdm(self.dataset.items()):
                ts_list = pd.Series([item[1] for item in items])
                # Collect timestamps where next timestamp - current timestamp = 30000
                self.train_users_dic[user_id] = list(ts_list.loc[ts_list.shift(-1) - ts_list == 30000])

            print('train_users_dic creation completed ({}s)'.format('%.2f' % (time.time() - start_time)))
            print('Saving train_users_dic to file {}'.format(filename))
            pickle.dump(self.train_users_dic, open(CACHE_FOLDER + filename, 'wb'))

    def get_articles(self):
        return self.articles

    def get_train_click_log(self):
        return self.train_click_log

    def get_test_click_log(self):
        return self.test_click_log

    def get_all_click_log(self):
        return self.all_click_log

    def get_user_list(self):
        return self.train_click_log['user_id'].unique()

    def get_item_dt_groupby_user(self):
        return self.dataset

    def users_df2dic(self, df_users):
        # Convert a DataFrame of users and timestamps to a dictionary {user_id: set_of_timestamps}
        _t = (df_users.sort_values('click_timestamp')
                      .groupby('user_id')
                      .apply(lambda x: set(x['click_timestamp']))
                      .reset_index()
                      .rename(columns={0: 'ts_set'}))

        return dict(zip(_t['user_id'], _t['ts_set']))

    def get_test_users(self, offline, samples=100000):
        """
        Get a dictionary of test users and their last clicked timestamps.
        If offline mode is True, randomly sample 'samples' users from the training dictionary.
        Otherwise, use the max timestamp per user from the test set.
        """
        if offline:
            users = []
            for user_id, ts_list in self.train_users_dic.items():
                if len(ts_list) > 0:
                    users.append((user_id, ts_list[-1]))

            np.random.seed(42)
            idx_list = np.random.choice(len(users), samples, replace=False)
            selected_users = [users[idx] for idx in idx_list]

            return self.users_df2dic(pd.DataFrame(selected_users, columns=['user_id', 'click_timestamp']))
        else:
            return self.users_df2dic(
                self.test_click_log.groupby('user_id').max('click_timestamp').reset_index()[['user_id', 'click_timestamp']]
            )

    def take_last(self, items, last=1):
        """
        Split the user's article list into two parts:
        all but the last 'last' items and the last item.
        """
        if len(items) <= last:
            return items.copy(), items[0]
        else:
            return items[:-last], items[-last]

    def get_train_dataset_and_answers(self, test_users):
        """
        From test_users, retrieve the training dataset (all items before the target)
        and the correct answer (the next clicked article) for each user/timestamp pair.
        """
        start_time = time.time()
        train_dataset = {}
        y_answer = {}

        for user_id, ts_set in tqdm(test_users.items()):
            items = self.dataset[user_id]
            for last_clicked_timestamp in ts_set:
                idx = [item[1] for item in items].index(last_clicked_timestamp)
                train_dataset.setdefault(user_id, {})
                train_dataset[user_id][last_clicked_timestamp] = items[0:idx+1]
                y_answer.setdefault(user_id, {})
                y_answer[user_id][last_clicked_timestamp] = items[idx+1][0]

        print('Train dataset and answers extraction completed ({}s)'.format('%.2f' % (time.time() - start_time)))
        return train_dataset, y_answer

    def get_train_dataset_for_online(self, test_users):
        """
        Prepare a dataset for online inference: 
        since we don't know the next article, use all items as the 'history' for each user/timestamp.
        """
        start_time = time.time()
        train_dataset = {}

        for user_id, ts_set in tqdm(test_users.items()):
            items = self.dataset[user_id]
            for last_clicked_timestamp in ts_set:
                train_dataset.setdefault(user_id, {})
                train_dataset[user_id][last_clicked_timestamp] = items

        print('Test dataset (online) preparation completed ({}s)'.format('%.2f' % (time.time() - start_time)))
        return train_dataset
