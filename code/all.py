import pandas as pd
import numpy as np
import sys
from data_holder import DataHolder
from tqdm import tqdm
import pickle
import math
import time
import multiprocessing as mp
import pickle
import lightgbm as lgb
from const import RAW_DATA_FOLDER, OUTPUT_FOLDER, CACHE_FOLDER
from recaller import calc_and_recall
from csv_handler import create_train_data
from sklearn.model_selection import train_test_split

from code.calc_i2i import i2i_30k_sim
from tqdm import tqdm
import multiprocessing as mp
import time
import math
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import time
from os.path import isfile
from const import CACHE_FOLDER

RAW_DATA_FOLDER = '../data/'
OUTPUT_FOLDER = '../prediction_result/'
CACHE_FOLDER = '../user_data/'


class DataHolder:
    def __init__(self, articles, train_click_log, test_click_log, trainB_click_log=None):
        self.articles = articles
        self.train_click_log = train_click_log
        self.test_click_log = test_click_log
        self.trainB_click_log = trainB_click_log
        self.all_click_log = self.train_click_log.append(self.trainB_click_log).append(self.test_click_log) if self.trainB_click_log is not None else self.train_click_log.append(self.test_click_log)
        self.all_click_log.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'], inplace=True)

        print('从train_click_log读取{}件(UserId=[{},{}])'.format(len(self.train_click_log), self.train_click_log['user_id'].min(), self.train_click_log['user_id'].max()))
        print('从test_click_log读取{}件(UserId=[{},{}])'.format(len(self.test_click_log), self.test_click_log['user_id'].min(), self.test_click_log['user_id'].max()))

        if self.trainB_click_log is not None:
            print('从trainB_click_log读取{}件(UserId=[{},{}])'.format(len(self.trainB_click_log), self.trainB_click_log['user_id'].min(), self.trainB_click_log['user_id'].max()))

        print('使用训练集all_click_log共{}件(UserId=[{},{}])'.format(len(self.all_click_log), self.all_click_log['user_id'].min(), self.all_click_log['user_id'].max()))

        # DataFrame对象转换成字典
        filename = 'dataset.pkl'
        if isfile(CACHE_FOLDER + filename):
            print('直接从文件{}中读取dataset'.format(filename))
            self.dataset = pickle.load(open(CACHE_FOLDER + filename, 'rb'))
        else:
            start_time = time.time()
            _t = self.all_click_log.sort_values('click_timestamp').groupby('user_id')\
                .apply(lambda x: list(zip(x['click_article_id'], x['click_timestamp'])))\
                .reset_index()\
                .rename(columns={0: 'item_dt_list'})

            self.dataset = dict(zip(_t['user_id'], _t['item_dt_list']))
            print('dataset对象完毕({}秒)'.format('%.2f' % (time.time() - start_time)))

            print('保存dataset至文件{}中'.format(filename))
            pickle.dump(self.dataset, open(CACHE_FOLDER + filename, 'wb'))

        # 生成可供训练用的(user_id, timestamp)字典
        filename = 'train_users_dic.pkl'
        if isfile(CACHE_FOLDER + filename):
            print('直接从文件{}中读取train_users_dic'.format(filename))
            self.train_users_dic = pickle.load(open(CACHE_FOLDER + filename, 'rb'))
        else:
            start_time = time.time()
            self.train_users_dic = {}
            for user_id, items in tqdm(self.dataset.items()):
                ts_list = pd.Series([item[1] for item in items])
                self.train_users_dic[user_id] = list(ts_list.loc[ts_list.shift(-1) - ts_list == 30000])

            print('train_users_dic对象完毕({}秒)'.format('%.2f' % (time.time() - start_time)))

            print('保存train_users_dic至文件{}中'.format(filename))
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
        _t = df_users.sort_values('click_timestamp').groupby('user_id')\
            .apply(lambda x: set(x['click_timestamp']))\
            .reset_index()\
            .rename(columns={0: 'ts_set'})

        return dict(zip(_t['user_id'], _t['ts_set']))

    def get_test_users(self, offline, samples=100000):
        if offline:
            # 一维数组化
            users = []
            for user_id, ts_list in self.train_users_dic.items():
                # for ts in ts_list:
                #     users.append((user_id, ts))
                if len(ts_list) > 0:
                    users.append((user_id, ts_list[-1]))
                
            np.random.seed(42)
            idx_list = np.random.choice(len(users), samples, replace=False)
            selected_users = [users[idx] for idx in idx_list]

            # 字典化
            return self.users_df2dic(pd.DataFrame(selected_users, columns=['user_id', 'click_timestamp']))
        else:
            return self.users_df2dic(self.test_click_log.groupby('user_id').max('click_timestamp').reset_index()[['user_id', 'click_timestamp']])

    def take_last(self, items, last=1):
        if len(items) <= last:
            return items.copy(), items[0]
        else:
            return items[:-last], items[-last]

    def get_train_dataset_and_answers(self, test_users):
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

        print('训练集和答案分割完毕({}秒)'.format('%.2f' % (time.time() - start_time)))
        return train_dataset, y_answer

    def get_train_dataset_for_online(self, test_users):
        start_time = time.time()
        train_dataset = {}

        for user_id, ts_set in tqdm(test_users.items()):
            items = self.dataset[user_id]
            for last_clicked_timestamp in ts_set:
                train_dataset.setdefault(user_id, {})
                train_dataset[user_id][last_clicked_timestamp] = items

        print('测试集制作完毕({}秒)'.format('%.2f' % (time.time() - start_time)))
        return train_dataset

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
    _item_counts_dic = {}
    _i2i_30k_sim = {}

    start_time = time.time()
    for user_id in user_id_list:
        item_dt = dataset[user_id]
        ts_list = pd.Series([ts for _, ts in item_dt])
        idx_list = [idx for idx, val in dict(ts_list - ts_list.shift(1) == 30000).items() if val]

        for idx in idx_list:
            i_art_id, _ = item_dt[idx]
            j_art_id, _ = item_dt[idx - 1]

            _i2i_30k_sim.setdefault(i_art_id, {})
            _i2i_30k_sim[i_art_id].setdefault(j_art_id, 0)
            _i2i_30k_sim[i_art_id][j_art_id] += 1

            _i2i_30k_sim.setdefault(j_art_id, {})
            _i2i_30k_sim[j_art_id].setdefault(i_art_id, 0)
            _i2i_30k_sim[j_art_id][i_art_id] += 1

    print('子任务[{}]: 完成i2i_30k相似度的计算。({}秒)'.format(job_id, '%.2f' % (time.time() - start_time)))

    return _i2i_30k_sim

def i2i_30k_sim(dataset, n_cpu, offline, max_related=50):
    filename = 'i2i_30k_sim_{}.pkl'.format('offline' if offline else 'online')
    if isfile(CACHE_FOLDER + filename):
        print('直接从文件{}中读取计算好的i2i_30k相似度'.format(filename))
        return pickle.load(open(CACHE_FOLDER + filename, 'rb'))

    # 计算相似度
    start_time = time.time()
    print('开始计算i2i_30k相似度')
    i2i_sim_3k = {}
    n_block = (len(dataset.keys()) - 1) // n_cpu + 1
    keys = list(dataset.keys())
    pool = mp.Pool(processes=n_cpu)
    results = [pool.apply_async(_i2i_30k_sim_core, args=(i, keys[i * n_block:(i + 1) * n_block], dataset)) for i in range(0, n_cpu)]
    pool.close()
    pool.join()

    for result in results:
        _i2i_sim_3k = result.get()

        for art_id, related_art_id_dic in _i2i_sim_3k.items():
            i2i_sim_3k.setdefault(art_id, {})
            for related_art_id, value in related_art_id_dic.items():
                i2i_sim_3k[art_id].setdefault(related_art_id, 0)
                i2i_sim_3k[art_id][related_art_id] += value

    print('逆序排序')
    for art_id, related_arts in tqdm(i2i_sim_3k.items()):
        sorted_and_topK = sorted(related_arts.items(), key=lambda x: x[1], reverse=True)
        i2i_sim_3k[art_id] = {
            'sorted_keys': [art_id for art_id, _ in sorted_and_topK],
            'related_arts': dict(sorted_and_topK)
        } 

    print('i2i_30k相似度计算完毕({}秒)'.format('%.2f' % (time.time() - start_time)))
    print('保存i2i_30k相似度数据至文件{}中'.format(filename))
    pickle.dump(i2i_sim_3k, open(CACHE_FOLDER + filename, 'wb'))
    return i2i_sim_3k

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from const import CACHE_FOLDER

def neg_sampling(ds, min=1, max=5):
    start_time = time.time()
    pos_ds = ds.loc[ds['answer'] == 1]
    neg_ds = ds.loc[ds['answer'] == 0]

    def _neg_sampling_func(x):
        n_sampling = len(x)
        n_sampling = min if n_sampling < min else (max if n_sampling > max else n_sampling)
        return x.sample(n=n_sampling, replace=False)

    neg_ds = pd.concat([
        neg_ds.groupby(['user_id', 'last_clicked_timestamp']).apply(_neg_sampling_func),
        neg_ds.groupby('article_id').apply(_neg_sampling_func),
        ]).drop_duplicates()

    ret = pd.concat([pos_ds, neg_ds]).reset_index(drop=True)
    print('负采样处理完毕({}秒, {}->{}件)'.format('%.2f' % (time.time() - start_time), len(ds), len(ret)))
    return ret

def get_user_features(raw_data, train_dataset, test_users, articles_dic):
    def calc_avg_words_count(items):
        return np.average([articles_dic[item[0]]['words_count'] for item in items])

    def calc_min_words_count(items):
        return np.min([articles_dic[item[0]]['words_count'] for item in items])

    def calc_max_words_count(items):
        return np.max([articles_dic[item[0]]['words_count'] for item in items])

    def calc_lag_between_created_at_ts_and_clicked_ts(items, articles_dic):
        item = items[-1]
        return (item[1] - articles_dic[item[0]]['created_at_ts']) / (1000 * 60 * 60 * 24)

    def calc_lag_between_two_click(items):
        if len(items) > 1:
            return (items[-1][1] - items[-2][1]) / (1000 * 60 * 60 * 24)
        else:
            return np.nan

    def calc_lag_between_two_articles(items, articles_dic):
        if len(items) > 1:
            return (articles_dic[items[-1][0]]['created_at_ts'] - articles_dic[items[-2][0]]['created_at_ts']) / (1000 * 60 * 60 * 24)
        else:
            return np.nan

    df_users = pd.DataFrame(list(test_users.keys()), columns=['user_id'])

    # 计算
    # 1. 用户看新闻的平均字数
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

    df1 = pd.DataFrame(_data, columns=['user_id', 'last_clicked_timestamp', 'avg_words_count', 'min_words_count', 'max_words_count', 'lag_between_created_at_ts_and_clicked_ts', 'lag_between_two_click', 'lag_between_two_articles'])

    # 计算用户使用设备，环境等的众数
    columns = ['user_id','click_environment','click_deviceGroup','click_os','click_country','click_region','click_referrer_type']
    df2 = df_users.merge(raw_data.get_all_click_log())[columns].groupby('user_id').agg(lambda x: x.value_counts().index[0]).reset_index()

    return df1.merge(df2)
    
def create_train_data(raw_data, train_dataset, test_users, articles_dic, recall_results, offline, y_answer):
    start_time = time.time()
    keys_ds = []

    for user_id, ts_set in test_users.items():
        for last_clicked_timestamp in ts_set:
            items = np.concatenate([result[user_id][last_clicked_timestamp] for _, result in recall_results.items()])
            keys_ds.append(list(zip(np.repeat(user_id, len(items)), np.repeat(last_clicked_timestamp, len(items)), items)))

    ds = pd.DataFrame(np.concatenate(keys_ds), columns=['user_id', 'last_clicked_timestamp', 'article_id'], dtype=np.int64).drop_duplicates()

    if offline:
        answer_keys_ds = []
        # 拼接正确答案标签
        for user_id, ts_list in y_answer.items():
            for last_clicked_timestamp, art_id in ts_list.items():
                answer_keys_ds.append((user_id, last_clicked_timestamp, art_id))

        answers = pd.DataFrame(answer_keys_ds, columns=['user_id', 'last_clicked_timestamp', 'article_id'], dtype=np.int64)
        # 将正确答案融合进数据集
        answers['answer'] = 1
        ds = ds.merge(answers, how='left').fillna({'answer': 0})
        ds['answer'] = ds['answer'].astype(np.int8)

        # 负采样
        ds = neg_sampling(ds)

    ds = ds.merge(raw_data.get_articles()).merge(get_user_features(raw_data, train_dataset, test_users, articles_dic))

    # 新特征
    ds['lag_period_last_article'] = ds['last_clicked_timestamp'] - ds['created_at_ts']
    ds['diff_words_last_article'] = ds['avg_words_count'] - ds['words_count']
    ds.to_csv(CACHE_FOLDER + '{}.csv'.format('train' if offline else 'test'), index=False)
    print('{}用的csv文件生成完毕({}秒, {}件)'.format('训练' if offline else '测试', '%.2f' % (time.time() - start_time), len(ds)))


def get_clicked_items(items):
    return { art_id for art_id, _ in items }

def _calc_sim(dataset, articles_dic, cpu_cores, offline):
    # 计算各种相似度
    num = len([i2i_30k_sim])

    start_time = time.time()
    print('召回前的计算处理开始({}件)'.format(num))

    sims = {}
    sims['i2i_30k_sim'] = i2i_30k_sim(dataset, cpu_cores, offline)

    print('召回前的计算处理结束({}秒)'.format('%.2f' % (time.time() - start_time)))

    return sims


def _is_recall_target(last_clicked_timestamp, art_id, articles_dic, lag_hour_max=27, lag_hour_min=3):
    # 热度文章在用户最后一次点击时刻起，前3小时~27小时内的文章
    lag_max = lag_hour_max * 60 * 60 * 1000
    lag_min = lag_hour_min * 60 * 60 * 1000
    if articles_dic[art_id]['created_at_ts'] < (last_clicked_timestamp - lag_max):
        return False

    if articles_dic[art_id]['created_at_ts'] > (last_clicked_timestamp - lag_min):
        return False

    return True

def _recall_hot_items(dataset, train_dataset, test_users, articles_dic, topK=10):
    result = {}
    start_time = time.time()
    lag_hour_min = 3
    lag_hour_max = 27

    hot_items = {}
    for _, items in tqdm(dataset.items()):
        for art_id, _ in items:
            hot_items.setdefault(art_id, 0)
            hot_items[art_id] += 1

    sorted_hot_items = sorted(hot_items.items(), key=lambda x: x[1], reverse=True)

    for user_id, ts_set in tqdm(test_users.items()):
        for last_clicked_timestamp in ts_set:
            items = train_dataset[user_id][last_clicked_timestamp]
            clicked_items = get_clicked_items(items)
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

    print('hot召回处理完毕({}秒) 限制：[{}-{}]'.format('%.2f' % (time.time() - start_time), lag_hour_min, lag_hour_max))
    return result

def _recall_i2i_30k_sim_items(dataset, test_users, articles_dic, i2i_30k_sim, topK=25):
    result = {}
    start_time = time.time()
    lag_hour_min = 0
    lag_hour_max = 27

    for user_id, ts_set in tqdm(test_users.items()):
        for last_clicked_timestamp in ts_set:
            items = dataset[user_id][last_clicked_timestamp]
            clicked_items = get_clicked_items(items)
            recommend_items = {}

            for art_id, _ in items:
                if art_id not in i2i_30k_sim:
                    break

                recommand_art_id_list = i2i_30k_sim[art_id]['sorted_keys']
                for recommend_art_id in recommand_art_id_list:
                    if recommend_art_id in clicked_items:
                        continue

                    if not _is_recall_target(last_clicked_timestamp, art_id, articles_dic, lag_hour_min=lag_hour_min, lag_hour_max=lag_hour_max):
                        continue

                    if i2i_30k_sim[art_id]['related_arts'][recommend_art_id] < 2:
                        break

                    recommend_items.setdefault(recommend_art_id, 0)
                    recommend_items[recommend_art_id] += (i2i_30k_sim[art_id]['related_arts'][recommend_art_id])
            
            result.setdefault(user_id, {})
            result[user_id][last_clicked_timestamp] = [art_id for art_id, _ in sorted(recommend_items.items(), key=lambda x: x[1], reverse=True)[:topK]]

    print('i2i_30k_sim召回处理完毕({}秒) 限制：[{}-{}]'.format('%.2f' % (time.time() - start_time), lag_hour_min, lag_hour_max))
    return result

def calc_and_recall(dataset, train_dataset, test_users, articles_dic, cpu_cores, offline, answers=None):
    sims = _calc_sim(dataset, articles_dic, cpu_cores, offline)
    num = len([_recall_hot_items, _recall_i2i_30k_sim_items])

    start_time = time.time()
    print('召回处理开始({}件)'.format(num))

    recalls = {}
    recalls['hot'] = _recall_hot_items(dataset, train_dataset, test_users, articles_dic)
    recalls['i2i_30k_sim'] = _recall_i2i_30k_sim_items(train_dataset, test_users, articles_dic, sims['i2i_30k_sim'])

    if offline and answers is not None:
        test_users_count = np.sum([len(ts_list) for _, ts_list in test_users.items()])
        for recall_name, result in recalls.items():
            accuracy = 0
            recall_counts = np.repeat(0, np.max([len(items) for _, ts_list in result.items() for _, items in ts_list.items()]))
            for user_id, ts_list in result.items():
                for last_clicked_timestamp, items in ts_list.items():
                    if answers[user_id][last_clicked_timestamp] in items:
                        accuracy += 1 
                        recall_counts[items.index(answers[user_id][last_clicked_timestamp])] += 1
            
            print('召回处理[{}]的召回率为{}%'.format(recall_name, '%.2f' % (accuracy * 100 / test_users_count)))
            print('召回处理[{}]的详细召回命中计数: {}'.format(recall_name, recall_counts))

        total_accuracy = 0
        for user_id, ts_list in test_users.items():
            for last_clicked_timestamp in ts_list:
                for _, result in recalls.items():
                    if answers[user_id][last_clicked_timestamp] in result[user_id][last_clicked_timestamp]:
                        total_accuracy += 1
                        break

        print('所有召回处理的总召回率为{}%'.format('%.2f' % (total_accuracy * 100 / test_users_count)))

    print('召回处理结束({}秒)'.format('%.2f' % (time.time() - start_time)))

    return recalls


def read_raw_data(filename, cb=None):
    data = pd.read_csv(RAW_DATA_FOLDER + filename)
    return cb(data) if cb is not None else data

def read_all_raw_data(filenames=['articles.csv', 'train_click_log.csv', 'testA_click_log.csv']):
    return DataHolder(*[read_raw_data(filename) for filename in filenames])

def calc_mrr_and_hit(recommend_dict, y, k=5):
    #assert len(recommend_dict) == len(y)
    sum_mrr = 0.0
    sum_hit = 0.0
    sum_hit_detail = np.repeat(0.0, 5)
    user_cnt = len(recommend_dict.keys())

    for user_id, recommend_items in recommend_dict.items():
        answer = y[user_id] if user_id in y else -1
        if (answer in recommend_items) and (recommend_items.index(answer) < k):
            sum_hit += 1
            sum_mrr += 1 / (recommend_items.index(answer) + 1)
            sum_hit_detail[recommend_items.index(answer)] += 1

    return (sum_mrr / user_cnt), (sum_hit / user_cnt), (sum_hit_detail / user_cnt)

def create_submission(recommend_dict):
    _data = [{'user_id': user_id,
        'article_1': art_id_list[0],
        'article_2': art_id_list[1],
        'article_3': art_id_list[2],
        'article_4': art_id_list[3],
        'article_5': art_id_list[4]} for user_id, art_id_list in tqdm(recommend_dict.items())]
    _t = pd.DataFrame(_data)
    _t.sort_values('user_id', inplace=True)
    _t.to_csv(OUTPUT_FOLDER + 'result.csv', index=False)

def handler(offline=True):
    cpu_cores = mp.cpu_count()
    print('使用CPU核心数: {}'.format(cpu_cores))
    print('开始{}数据验证处理'.format('线下' if offline else '线上'))
    raw_data = read_all_raw_data()
    test_users = raw_data.get_test_users(offline)

    _user_id_list = list(test_users.keys())
    user_id_min = np.min(_user_id_list)
    user_id_max = np.max(_user_id_list)
    print('获得{}用户集合{}件 [{} ~ {}]'.format('验证' if offline else '测试', len(test_users), user_id_min, user_id_max))

    dataset = raw_data.get_item_dt_groupby_user()

    if offline:
        train_dataset, y_answer = raw_data.get_train_dataset_and_answers(test_users)
    else:
        train_dataset = raw_data.get_train_dataset_for_online(test_users)
        y_answer = None

    print('训练数据({}件)'.format(np.sum([len(ts_list) for user_id, ts_list in train_dataset.items()])))

    articles_dic = dict(list(raw_data.get_articles().apply(lambda x: (x['article_id'], dict(x)), axis=1)))
    print('获得文章字典({}件)'.format(len(articles_dic.keys())))

    recall_results = calc_and_recall(dataset, train_dataset, test_users, articles_dic, cpu_cores, offline, y_answer)
    create_train_data(raw_data, train_dataset, test_users, articles_dic, recall_results, offline, y_answer)

def make_train_data():
    handler()

def make_test_data():
    handler(False)

def prepare_dataset(df):
    agg_column = [column for column in df.columns if column != 'user_id'][0]
    df.sort_values('user_id', inplace=True)
    grp_info = df.groupby('user_id', as_index=False).count()[agg_column].values
    y = df['answer'] if 'answer' in df.columns else None
    return df.drop(columns=['answer']) if 'answer' in df.columns else df, grp_info, y

def make_recommend_dict(X_val, y_pred):
    X_val['pred'] = y_pred
    _t = X_val.groupby('user_id')\
        .apply(lambda x: list(x.sort_values('pred', ascending=False)['article_id'].head(5)))\
        .reset_index()\
        .rename(columns={0: 'item_list'})

    recommend_dict = dict(zip(_t['user_id'], _t['item_list']))    
    return recommend_dict

def test():
    df_train = pd.read_csv(CACHE_FOLDER + 'train.csv')

    clf = lgb.LGBMRanker(random_state=777, n_estimators=1000)

    users = df_train['user_id'].unique()
    train_users, _test_users = train_test_split(users, test_size=0.2, random_state=98)
    test_users, val_users = train_test_split(_test_users, test_size=0.5, random_state=38)
    df_new_train = df_train.merge(pd.DataFrame(train_users, columns=['user_id']))
    df_test = df_train.merge(pd.DataFrame(test_users, columns=['user_id']))
    df_val = df_train.merge(pd.DataFrame(val_users, columns=['user_id']))

    X_train, X_grp_train, y_train = prepare_dataset(df_new_train)
    X_test, X_grp_test, y_test = prepare_dataset(df_test)
    X_val, X_grp_val, _ = prepare_dataset(df_val)

    def handle_columns(X):
        return X.drop(columns=['user_id', 'article_id'])

    _X_train = handle_columns(X_train)

    clf.fit(_X_train, y_train, group=X_grp_train, eval_set=[(handle_columns(X_test), y_test)], eval_group=[X_grp_test], eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ], early_stopping_rounds=50, verbose=False)
    print('Best iteration: {}'.format(clf.best_iteration_))


    for X, X_grp, df, title in [(X_test, X_grp_test, df_test, 'Test Set'), (X_val, X_grp_val, df_val, 'Validation Set')]:
        print('[{}]'.format(title))
        y_pred = clf.predict(handle_columns(X), group=X_grp, num_iteration=clf.best_iteration_)
        recommend_dict = make_recommend_dict(X, y_pred)
        answers = dict(df.loc[df['answer'] == 1, ['user_id', 'article_id']].values)
        mrr, hit, details = calc_mrr_and_hit(recommend_dict, answers)
        print('MRR: {} / HIT: {}'.format(mrr, hit))
        print(' / '.join(['%.2f' % detail for detail in details]))

    for column, score in sorted(zip(_X_train.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True):
        print('{}: {}'.format(column, score))

def run():
    df_train = pd.read_csv(CACHE_FOLDER + 'train.csv')
    df_test = pd.read_csv(CACHE_FOLDER + 'test.csv')

    clf = lgb.LGBMRanker(random_state=777, n_estimators=1000)

    users = df_train['user_id'].unique()
    train_users, eval_users = train_test_split(users, test_size=0.2, random_state=77)
    df_new_train = df_train.merge(pd.DataFrame(train_users, columns=['user_id']))
    df_eval = df_train.merge(pd.DataFrame(eval_users, columns=['user_id']))

    X_train, X_grp_train, y_train = prepare_dataset(df_new_train)
    X_eval, X_grp_eval, y_eval = prepare_dataset(df_eval)
    X_test, X_grp_test, _ = prepare_dataset(df_test)

    def handle_columns(X):
        return X.drop(columns=['user_id', 'article_id'])

    _X_train = handle_columns(X_train)

    clf.fit(_X_train, y_train, group=X_grp_train, eval_set=[(handle_columns(X_eval), y_eval)], eval_group=[X_grp_eval], eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ], early_stopping_rounds=50, verbose=False)
    print('Best iteration: {}'.format(clf.best_iteration_))
    y_pred = clf.predict(handle_columns(X_test), group=X_grp_test, num_iteration=clf.best_iteration_)
    
    for column, score in sorted(zip(_X_train.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True):
        print('{}: {}'.format(column, score))

    recommend_dict = make_recommend_dict(X_test, y_pred)

    create_submission(recommend_dict)

if __name__ == "__main__":
    make_train_data()
    test()
    make_test_data()
    # run()
