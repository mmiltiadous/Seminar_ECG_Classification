import os

from mlpy.lib.utils.path import tag_path, makedirs
from mlpy.configure import DIR_DATA_UCR15
from mlpy.datasets import ucr_uea

from configure import DIR_LOG
from exp.vis import tsne

import pandas as pd
import numpy as np


def run(data_name, dir_data, log_dir):
    log_path = makedirs(os.path.join(log_dir, data_name))

    df_train = pd.read_csv('/Users/fiarresga/Desktop/2024/SADL/DATA_SADL/mydata2715/mydata2715_train.csv')
    classes = df_train.iloc[:, 0].values
    time_series_data = df_train.iloc[:, 1:].values
    y_tr = np.array(classes)
    x_tr = np.array(time_series_data)

    df_test = pd.read_csv('/Users/fiarresga/Desktop/2024/SADL/DATA_SADL/mydata2715/mydata2715_test.csv')
    classes = df_test.iloc[:, 0].values
    time_series_data = df_test.iloc[:, 1:].values
    y_te = np.array(classes)
    x_te = np.array(time_series_data)

    tsne(x_tr, y_tr, x_te, y_te, log_path)


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 4)
    log_dir = makedirs(os.path.join(DIR_LOG, tag))

    data_name = 'Two_Patterns'

    run(data_name, DIR_DATA_UCR15, log_dir)