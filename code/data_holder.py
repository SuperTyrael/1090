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