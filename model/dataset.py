import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import gzip
import pickle


class DtmlClassificationDataset(Dataset):
    def __init__(self, stock_df, macro_df, window_size, target_window_size, target_criteria, target_min_true,
                 unq_stock_cd, void=False, unit_split=252, target_index=None):
        # variable init
        self._stock_data = None
        self._target_data = None
        self._macro_data = None
        self._raw_target_data = None
        self._date = None
        self._stock_cd_idx = None
        self.unq_date = None
        self.unq_stock_cd = None
        self.unit_split = unit_split
        self.target_index = target_index

        # void class
        if void:
            pass
        else:
            # assert
            assert stock_df.columns[0] == 'date', 'First column of stock_df must be "date".'
            assert stock_df.columns[1] == 'stock_cd', 'Second column of stock_df must be "stock_cd".'
            assert stock_df.columns[2] == 'target_adj_close_price', 'Third column of stock_df must be "target_adj_close_price".'
            if self.target_index is not None:
                assert self.target_index in macro_df.columns, 'macro_df must have target_index column'

            # sort
            stock_df = stock_df.sort_values(['stock_cd', 'date']).reset_index(drop=True)
            macro_df = macro_df.sort_values(['date']).reset_index(drop=True)

            # date array 생성
            unq_date = stock_df['date'].unique()

            # 주가정보 학습 데이터 생성(종목 * 날짜 * window * 변수)
            _stock_data = torch.zeros((len(unq_stock_cd), len(unq_date)-target_window_size, window_size, len(stock_df.columns)-3))
            _target_data = torch.zeros((len(unq_stock_cd), len(unq_date)-target_window_size))
            _raw_target_data = torch.zeros((len(unq_stock_cd), len(unq_date)-target_window_size, target_window_size))
            _date = torch.zeros((len(unq_stock_cd), len(unq_date)-target_window_size)).long()
            _stock_cd_idx = torch.zeros((len(unq_stock_cd), len(unq_date)-target_window_size))
            for i, cd in enumerate(tqdm(unq_stock_cd)):
                # 특정 date에 정보가 없을 경우 0으로 채움
                date_df = pd.DataFrame({'stock_cd': cd, 'date': unq_date})
                tmp_df = stock_df.loc[stock_df['stock_cd'].values == cd, :]
                tmp_df = date_df.merge(tmp_df, how='left', on=['stock_cd', 'date'])
                tmp_df = tmp_df.fillna(0)

                # _stock_data
                for j in range(window_size):
                    for k in range(len(stock_df.columns)-3):
                        _stock_data[i, :, j, k] = torch.from_numpy(tmp_df.iloc[:-target_window_size, k+3].shift(window_size-(j+1)).fillna(method='bfill').values)

                # _target_data & _stock_cd_idx & _date
                for j in range(len(unq_date)-target_window_size):
                    # _target_data
                    tmp_target = tmp_df['target_adj_close_price'][j:j+target_window_size+1].values
                    if self.target_index is not None:
                        tmp_index = np.cumprod(macro_df[self.target_index][j:j+target_window_size+1].values + 1)  # 수익률 값으로 들어오므로, cumprod 취해줌

                    if tmp_target[0] != 0:
                        tmp_target = tmp_target / tmp_target[0]
                        if self.target_index is not None:
                            tmp_index = tmp_index / tmp_index[0]
                            tmp_target = tmp_target - tmp_index + 1
                        _target_data[i, j] = 1 if sum(tmp_target > target_criteria + 1) >= target_min_true else 0
                        _raw_target_data[i, j, :] = torch.from_numpy(tmp_target[1:])
                    else:
                        _target_data[i, j] = 0

                    # _stock_cd_idx & _date
                    _stock_cd_idx[i, j] = i
                    _date[i, j] = tmp_df['date'][j]

            # 매크로정보 학습 데이터 생성(날짜 * window * 변수)
            _macro_data = torch.zeros((len(unq_date)-target_window_size, window_size, len(macro_df.columns)-1))
            for j in range(window_size):
                for k in range(len(macro_df.columns)-1):
                    _macro_data[:, j, k] = torch.from_numpy(macro_df.iloc[:-target_window_size, k + 1].shift(window_size - (j + 1)).fillna(method='bfill').values)

            # (날짜 * 종목 * window * 변수) 형태로 데이터 변경
            _stock_data = _stock_data.transpose(0, 1)
            _target_data = _target_data.transpose(0, 1)
            _raw_target_data = _raw_target_data.transpose(0, 1)
            _stock_cd_idx = np.swapaxes(_stock_cd_idx, 0, 1)
            _date = _date.transpose(0, 1)

            # window size 이전 날짜 날리기
            self._stock_data = _stock_data[(window_size-1):, :, :, :]
            self._macro_data = _macro_data[(window_size-1):, :, :]
            self._target_data = _target_data[(window_size-1):, :].long()
            self._raw_target_data = _raw_target_data[(window_size-1):, :, :]
            self._stock_cd_idx = _stock_cd_idx[(window_size-1):, :]
            self._date = _date[(window_size-1):, :]

            # date, stock_cd 저장하기
            self.unq_date = np.array(unq_date[(window_size-1):-target_window_size])
            self.unq_stock_cd = unq_stock_cd

    def __len__(self):
        return self._target_data.size(0)

    def __getitem__(self, idx):
        return self._stock_data[idx], self._macro_data[idx], self._target_data[idx], self._raw_target_data[idx], \
               self._stock_cd_idx[idx], self._date[idx]

    def get_subset(self, date_list=None, stock_cd_list=None):
        if date_list is None:
            date_list = self.unq_date
        if stock_cd_list is None:
            stock_cd_list = self.unq_stock_cd

        # date_list, stock_cd_list가 현재 데이터셋에 포함되어있는지 확인
        assert all(np.isin(date_list, self.unq_date)) == True, 'date_list has uncovered date.'
        assert all(np.isin(stock_cd_list, self.unq_stock_cd)) == True, 'stock_cd_list has uncovered stock_cd.'

        # 해당 조건의 index 확인
        date_index = np.where(np.isin(self.unq_date, date_list) == True)[0]
        stock_cd_index = np.where(np.isin(self.unq_stock_cd, stock_cd_list) == True)[0]

        # void dataset 생성
        subset = DtmlClassificationDataset(stock_df=None, macro_df=None, window_size=None, target_window_size=None,
                                           target_criteria=None, target_min_true=None, unq_stock_cd=None, void=True)
        subset._stock_data = torch.index_select(torch.index_select(self._stock_data, 0, torch.from_numpy(date_index)), 1, torch.from_numpy(stock_cd_index))
        subset._target_data = torch.index_select(torch.index_select(self._target_data, 0, torch.from_numpy(date_index)), 1, torch.from_numpy(stock_cd_index))
        subset._macro_data = torch.index_select(self._macro_data, 0, torch.from_numpy(date_index))
        subset._raw_target_data = torch.index_select(torch.index_select(self._raw_target_data, 0, torch.from_numpy(date_index)), 1, torch.from_numpy(stock_cd_index))
        subset._date = torch.index_select(torch.index_select(self._date, 0, torch.from_numpy(date_index)), 1, torch.from_numpy(stock_cd_index))
        subset._stock_cd_idx = torch.index_select(torch.index_select(self._stock_cd_idx, 0, torch.from_numpy(date_index)), 1, torch.from_numpy(stock_cd_index))
        subset.unq_date = date_list
        subset.unq_stock_cd = stock_cd_list

        return subset

    def merge(self, merge_dataset):
        if self.unq_date is None:
            self._stock_data = merge_dataset._data
            self._target_data = merge_dataset._target_data
            self._macro_data = merge_dataset._macro_data
            self._raw_target_data = merge_dataset._raw_target_data
            self._date = merge_dataset._date
            self._stock_cd_idx = merge_dataset._stock_cd_idx
            self.unq_date = merge_dataset.unq_date
            self.unq_stock_cd = merge_dataset.unq_stock_cd
            self.unit_split = merge_dataset.save_unit_days
        else:
            # unq_stock_cd 확인
            assert all(merge_dataset.unq_stock_cd == self.unq_stock_cd) == True, 'unq_stock_cd of merge_dataset doesn\'t match'

            # merge_dataset에서 중복 date 삭제
            merge_date = merge_dataset.unq_date[~np.isin(merge_dataset.unq_date, self.unq_date)]
            if len(merge_date) == 0:
                return None
            else:
                merge_dataset = merge_dataset.get_subset(merge_date)

            # merge
            self._stock_data = torch.concat([self._stock_data, merge_dataset._data])
            self._target_data = torch.concat([self._target_data, merge_dataset._target_data])
            self._macro_data = torch.concat([self._macro_data, merge_dataset._macro_data])
            self._raw_target_data = torch.concat([self._raw_target_data, merge_dataset._raw_target_data])
            self._date = torch.concat([self._date, merge_dataset._date])
            self._stock_cd_idx = torch.concat([self._stock_cd_idx, merge_dataset._stock_cd_idx])
            self.unq_date = np.concatenate((self.unq_date, merge_dataset.unq_date))

    def save(self, fname):
        n_split = len(self.unq_date) // self.unit_split + 1
        for i in range(n_split):
            ith_split = self.get_subset(date_list=self.unq_date[i * self.unit_split:(i+1) * self.unit_split])
            with gzip.open(f'./data/{fname}_{i}.pickle', 'wb') as f:
                pickle.dump(ith_split, f)

    def load(self, fname):
        i = 0
        while True:
            try:
                with gzip.open(f'./data/{fname}_{i}.pickle', 'rb') as f:
                    tmp_dataset = pickle.load(f)
                    self.merge(tmp_dataset)
                    del tmp_dataset
            except FileNotFoundError as e:
                break
            i = i + 1
