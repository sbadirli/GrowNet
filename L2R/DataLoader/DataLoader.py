import datetime
import os

import pandas as pd
import numpy as np
from sklearn import preprocessing


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class L2R_DataLoader:

    def __init__(self, path):
        """
        :param path: str
        """
        self.path = path
        self.pickle_path = path[:-3] + 'pkl'
        self.df = None
        self.num_pairs = None
        self.num_sessions = None

    def get_num_pairs(self):
        if self.num_pairs is not None:
            return self.num_pairs
        self.num_pairs = 0
        for _, Y in self.generate_batch_per_query(self.df):
            Y = Y.reshape(-1, 1)
            pairs = Y - Y.T
            pos_pairs = np.sum(pairs > 0, (0, 1))
            neg_pairs = np.sum(pairs < 0, (0, 1))
            assert pos_pairs == neg_pairs
            self.num_pairs += pos_pairs + neg_pairs
        return self.num_pairs

    def get_num_sessions(self):
        return self.num_sessions

    def _load_mslr(self):
        print(get_time(), "load file from {}".format(self.path))
        df = pd.read_csv(self.path, sep=" ", header=None)
        df.drop(columns=df.columns[-1], inplace=True)
        self.num_features = len(df.columns) - 2
        self.num_paris = None
        print(get_time(), "finish loading from {}".format(self.path))
        print("dataframe shape: {}, features: {}".format(df.shape, self.num_features))
        return df

    def _parse_feature_and_label(self, df):
        """
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        print(get_time(), "parse dataframe ...", df.shape)
        for col in range(1, len(df.columns)):
            if ':' in str(df.iloc[:, col][0]):
                df.iloc[:, col] = df.iloc[:, col].apply(lambda x: x.split(":")[1])
        df.columns = ['rel', 'qid'] + [str(f) for f in range(1, len(df.columns) - 1)]

        for col in [str(f) for f in range(1, len(df.columns) - 1)]:
            df[col] = df[col].astype(np.float32)

        print(get_time(), "finish parsing dataframe")
        self.df = df
        self.num_sessions = len(df.qid.unique())
        return df

    def generate_query_pairs(self, df, qid):
        """
        :param df: pandas.DataFrame, contains column qid, rel, fid from 1 to self.num_features
        :param qid: query id
        :returns: numpy.ndarray of x_i, y_i, x_j, y_j
        """
        df_qid = df[df.qid == qid]
        rels = df_qid.rel.unique()
        x_i, x_j, y_i, y_j = [], [], [], []
        for r in rels:
            df1 = df_qid[df_qid.rel == r]
            df2 = df_qid[df_qid.rel != r]
            df_merged = pd.merge(df1, df2, on='qid')
            df_merged.reindex(np.random.permutation(df_merged.index))
            y_i.append(df_merged.rel_x.values.reshape(-1, 1))
            y_j.append(df_merged.rel_y.values.reshape(-1, 1))
            x_i.append(df_merged[['{}_x'.format(i) for i in range(1, self.num_features + 1)]].values)
            x_j.append(df_merged[['{}_y'.format(i) for i in range(1, self.num_features + 1)]].values)
        return np.vstack(x_i), np.vstack(y_i), np.vstack(x_j), np.vstack(y_j)

    def generate_query_pair_batch(self, df=None, batchsize=2000):
        """
        :param df: pandas.DataFrame, contains column qid
        :returns: numpy.ndarray of x_i, y_i, x_j, y_j
        """
        if df is None:
            df = self.df
        x_i_buf, y_i_buf, x_j_buf, y_j_buf = None, None, None, None
        qids = df.qid.unique()
        np.random.shuffle(qids)
        for qid in qids:
            x_i, y_i, x_j, y_j = self.generate_query_pairs(df, qid)
            if x_i_buf is None:
                x_i_buf, y_i_buf, x_j_buf, y_j_buf = x_i, y_i, x_j, y_j
            else:
                x_i_buf = np.vstack((x_i_buf, x_i))
                y_i_buf = np.vstack((y_i_buf, y_i))
                x_j_buf = np.vstack((x_j_buf, x_j))
                y_j_buf = np.vstack((y_j_buf, y_j))
            idx = 0
            while (idx + 1) * batchsize <= x_i_buf.shape[0]:
                start = idx * batchsize
                end = (idx + 1) * batchsize
                yield x_i_buf[start: end, :], y_i_buf[start: end, :], x_j_buf[start: end, :], y_j_buf[start: end, :]
                idx += 1

            x_i_buf = x_i_buf[idx * batchsize:, :]
            y_i_buf = y_i_buf[idx * batchsize:, :]
            x_j_buf = x_j_buf[idx * batchsize:, :]
            y_j_buf = y_j_buf[idx * batchsize:, :]

        yield x_i_buf, y_i_buf, x_j_buf, y_j_buf

    def generate_query_batch(self, df, batchsize):
        """
        :param df: pandas.DataFrame, contains column qid
        :returns: numpy.ndarray qid, rel, x_i
        """
        idx = 0
        while idx * batchsize < df.shape[0]:
            r = df.iloc[idx * batchsize: (idx + 1) * batchsize, :]
            yield r.qid.values, r.rel.values, r[['{}'.format(i) for i in range(1, self.num_features + 1)]].values
            idx += 1


    def generate_batch_per_query(self, df=None):
        """
        :param df: pandas.DataFrame
        :return: X for features, y for relavance
        :rtype: numpy.ndarray, numpy.ndarray
        """
        if df is None:
            df = self.df
        qids = df.qid.unique()
        np.random.shuffle(qids)
        for qid in qids:
            df_qid = df[df.qid == qid]
            yield df_qid[['{}'.format(i) for i in range(1, self.num_features + 1)]].values, df_qid.rel.values

    def load(self):
        """
        :return: pandas.DataFrame
        """
        if os.path.isfile(self.pickle_path):
            print(get_time(), "load from pickle file {}".format(self.pickle_path))
            self.df = pd.read_pickle(self.pickle_path)
            self.num_features = len(self.df.columns) - 2
            self.num_paris = None
            self.num_sessions = len(self.df.qid.unique())
        else:
            self.df = self._parse_feature_and_label(self._load_mslr())
            self.df.to_pickle(self.pickle_path)
        return self.df

    def train_scaler_and_transform(self):
        """Learn a scalar and apply transform."""
        feature_columns = [str(i) for i in range(1, self.num_features + 1)]
        X_train = self.df[feature_columns]
        #scaler = preprocessing.StandardScaler().fit(X_train)
        scaler = preprocessing.MinMaxScaler().fit(X_train)
        self.df[feature_columns] = scaler.transform(X_train)
        return self.df, scaler

    def apply_scaler(self, scaler):
        print(get_time(), "apply scaler to transform feature for {}".format(self.path))
        feature_columns = [str(i) for i in range(1, self.num_features + 1)]
        X_train = self.df[feature_columns]
        self.df[feature_columns] = scaler.transform(X_train)
        return self.df