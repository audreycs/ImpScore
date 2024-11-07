import numpy as np
import logging
import os
import pandas as pd
import random


class LoadDataset:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.total_df = self.load_dataset(dataset_dir)
        self.train_df, self.valid_df, self.test_df = self.split_train_valid_test()
        self.train_pairs = self.prepare_train_data()
        self.valid_pairs, self.test_pairs = self.prepare_valid_test_data()
        self.save_test_file()

    def load_dataset(self, dataset_dir):
        logging.info(f'Loading dataset from {dataset_dir}...')
        file = os.path.join(dataset_dir, 'all_data.csv')
        df = pd.read_csv(file, header=0, sep=',', encoding='utf-8')
        # calculate the number of samples in each class in column 'pragmatic'
        num_pair = len(df)
        logging.info(f'Number of positive and negative pairs: {num_pair*2}')
        return df

    def split_train_valid_test(self, valid_ratio=0.05):
        logging.info('Splitting dataset into train, valid, and test sets...')

        total_num = len(self.total_df)
        train_num = int(total_num * (1-valid_ratio*2))
        valid_num = int(total_num * valid_ratio)
        test_num = total_num - train_num - valid_num

        # randomly select training set
        train_index = random.sample(range(total_num), train_num)
        valid_index = random.sample(list(set(range(total_num)) - set(train_index)), valid_num)
        test_index = list(set(range(total_num)) - set(train_index) - set(valid_index))

        train_df = self.total_df.iloc[train_index]
        valid_df = self.total_df.iloc[valid_index]
        test_df = self.total_df.iloc[test_index]

        train_df_num_pos = len(train_df)
        train_df_num_neg = len(train_df)
        valid_df_num_pos = len(valid_df)
        valid_df_num_neg = len(valid_df)
        test_df_num_pos = len(test_df)
        test_df_num_neg = len(test_df)

        logging.info(f'Train set: {train_df_num_pos} positive pairs, {train_df_num_neg} negative pairs')
        logging.info(f'Valid set: {valid_df_num_pos} positive pairs, {valid_df_num_neg} negative pairs')
        logging.info(f'Test set: {test_df_num_pos} positive pairs, {test_df_num_neg} negative pairs')

        return train_df, valid_df, test_df

    def prepare_train_data(self) -> np.array:
        """
        Prepare training data for model training
        :return: train_pairs: np.array, shape=(num_pairs, 2, 2), each sample contains an implicit pair and explicit pair.
        """
        train_pairs = []
        for index, row in self.train_df.iterrows():
            pos_imp, pos_exp = row['pos_implicit'], row['pos_explicit']
            neg_imp, neg_exp = row['neg_implicit'], row['neg_explicit']
            pos_pair = [pos_imp, pos_exp]
            neg_pair = [neg_imp, neg_exp]
            train_pairs.append((pos_pair, neg_pair))

        train_pairs = np.array(train_pairs)
        return train_pairs

    def prepare_valid_test_data(self):
        """
        Prepare validation and test data for model evaluation
        :return: valid_pairs: np.array, shape=(num_pairs, 3), each sample contains an implicit pair or explicit pair, differed by the last element.
        :return: test_pairs: np.array, shape=(num_pairs, 3), each sample contains an implicit pair or explicit pair, differed by the last element.
        """
        valid_pairs = []
        test_pairs = []

        for index, row in self.valid_df.iterrows():
            pos_imp, pos_exp = row['pos_implicit'], row['pos_explicit']
            neg_imp, neg_exp = row['neg_implicit'], row['neg_explicit']
            pos_pair = [pos_imp, pos_exp, 1]
            neg_pair = [neg_imp, neg_exp, 0]
            valid_pairs.append(pos_pair)
            valid_pairs.append(neg_pair)

        for index, row in self.test_df.iterrows():
            pos_imp, pos_exp = row['pos_implicit'], row['pos_explicit']
            neg_imp, neg_exp = row['neg_implicit'], row['neg_explicit']
            pos_pair = [pos_imp, pos_exp, 1]
            neg_pair = [neg_imp, neg_exp, 0]
            test_pairs.append(pos_pair)
            test_pairs.append(neg_pair)

        valid_pairs = np.array(valid_pairs)
        test_pairs = np.array(test_pairs)

        return valid_pairs, test_pairs

    def save_test_file(self):
        test_df = pd.DataFrame(self.test_pairs, columns=['pos_implicit', 'pos_explicit', 'pragmatic_label'])
        test_df.to_csv(os.path.join(self.dataset_dir, 'test_data.csv'), index=False, encoding='utf-8')
        logging.info("")
        logging.info(f'Saved test pairs to {self.dataset_dir}...')
