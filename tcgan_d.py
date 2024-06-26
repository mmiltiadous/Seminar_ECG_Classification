import os
import fire
import numpy as np
import pandas as pd
from time import time
import tensorflow as tf
from tensorflow.keras import layers

import sklearn


from mlpy.datasets import ucr_uea
from mlpy.lib.utils.path import makedirs, tag_path
from mlpy.lib.utils.log import set_logging
from mlpy.configure import DIR_DATA_UCR15
from mlpy.lib.tfops.base import tf_keras_set_gpu_allow_growth
from mlpy.datasets.ucr_uea.data_names import UCR85_DATASETS

from configure import DIR_LOG
from tcgan.model.tcgan import TCGAN, TCGANConfig

from tcgan.lib.eval import EvaluatorGAN, EvaluatorClf


tf_keras_set_gpu_allow_growth()

""" 

Replace the last layer of GAN.discriminator with softmax layer, train it from scratch.
The resulted network = a FCN.

"""


class Classifier(object):
    def __init__(self, log_dir, logger, model_obj, model_cfg_obj, idx_layer, data_name, data_dir,
                 model_cfg_kwargs=None):
        self.log_dir = log_dir
        self.logger = logger
        self.model_obj = model_obj
        self.model_cfg_obj = model_cfg_obj
        self.idx_layer = idx_layer
        self.data_name = data_name
        self.data_dir = data_dir
        self.model_cfg_kwargs = model_cfg_kwargs if model_cfg_kwargs is not None else {}

        self._load_data()
        self._init_model()

    def _load_data(self):
        df_train = pd.read_csv('/Users/fiarresga/Desktop/2024/SADL/DATA_SADL/mydata1000/mydata1000_train.csv')
        #df_train = df_train[df_train.iloc[:, 0] != 3]
        df_train = df_train[df_train.iloc[:, 0] != 2]
        #df_train = df_train.iloc[:10000, :]
        print('\n DF TRAIN SHAPE', df_train.shape, '\n')
        classes = df_train.iloc[:, 0].values
        time_series_data = df_train.iloc[:, 1:].values
        y_tr = np.array(classes)
        x_tr = np.array(time_series_data)

        df_val = pd.read_csv('/Users/fiarresga/Desktop/2024/SADL/DATA_SADL/mydata1000/mydata1000_val.csv')
        #df_val = df_val[df_val.iloc[:, 0] != 3]
        df_val = df_val[df_val.iloc[:, 0] != 2]
        print('\n DF VAL SHAPE', df_val.shape, '\n')
        classes = df_val.iloc[:, 0].values
        #CHANGED FOR T-SNE
        #time_series_data = df_val.iloc[:, 2:].values
        time_series_data = df_val.iloc[:, 1:].values
        y_val = np.array(classes)
        x_val = np.array(time_series_data)

        df_test = pd.read_csv('/Users/fiarresga/Desktop/2024/SADL/DATA_SADL/mydata1000/mydata1000_test.csv')
        #df_test = df_test[df_test.iloc[:, 0] != 3]
        df_test = df_test[df_test.iloc[:, 0] != 2]
        print('\n DF TEST SHAPE', df_test.shape, '\n')
        classes = df_test.iloc[:, 0].values
        #counts_te = df_test.iloc[:, 1].values
        #time_series_data = df_test.iloc[:, 2:].values
        #CHANGED FOR T-SNE PLOT
        counts_te = 0
        time_series_data = df_test.iloc[:, 1:].values
        y_te = np.array(classes)
        x_te = np.array(time_series_data)

        x_tr = x_tr[..., np.newaxis]
        x_te = x_te[..., np.newaxis]
        x_val = x_val[..., np.newaxis]


        y_tr = tf.one_hot(y_tr, depth=4)
        y_te = tf.one_hot(y_te, depth=4)
        y_val = tf.one_hot(y_val, depth=4)


        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te
        self.x_val = x_val
        self.y_val = y_val
        self.counts_te = counts_te
        self.n_classes = 4
        self.input_shape = self.x_tr.shape[1:]

    def _init_model(self):
        #self.input_shape = self.x_tr.shape[1:]
        self.cfg = self.model_cfg_obj(self.input_shape, self.log_dir, self.logger, **self.model_cfg_kwargs)
        self.evaluator = EvaluatorGAN(self.cfg)
        self.pre_base = self.model_obj(self.cfg, self.evaluator)
        self.pre_base.fit(self.x_tr, self.x_te)

        self.base_model = self.pre_base.discriminator

    def run(self):
        inputs = self.base_model.input
        base_output = self.base_model.layers[self.idx_layer].output

        # new model
        flat = layers.Flatten()(base_output)
        predictions = layers.Dense(self.n_classes, activation='softmax')(flat)
        model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

        if self.cfg.verbose:
            print(f"#trainable_variables={len(model.trainable_variables)}")

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        t = time()
        history = model.fit(self.x_tr, self.y_tr,
                            batch_size=self.cfg.batch_size, epochs=self.cfg.epochs, verbose=self.cfg.verbose,
                            validation_data=(self.x_val, self.y_val))
        t_tr = time() - t

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(self.cfg.train_dir, 'history.csv'))
        print('\nSAVE ', self.cfg.ckpt_dir)
        model.save(self.cfg.ckpt_dir)
        model.save('/Users/fiarresga/Desktop/2024/SADL/sadl2024-tcgan/saved_model.h5')

        _, acc_tr = model.evaluate(self.x_tr, self.y_tr)
        t = time()
        _, acc_te = model.evaluate(self.x_te, self.y_te)
        t_te = time() - t

        res = {
            'data_name': self.data_name,
            'acc_train': acc_tr,
            'acc_test': acc_te,
            'n_params': model.count_params(),
            'time_train': t_tr,
            'time_test': t_te
        }

        return res


class ExperimentClf(object):
    def __init__(self,
                 tag,
                 model_obj,
                 model_cfg_obj,
                 data_dir,
                 data_name_list,
                 log_dir,
                 model_cfg_kwargs=None):
        self.tag = tag
        self.model_obj = model_obj
        self.model_cfg_obj = model_cfg_obj
        self.data_dir = data_dir
        self.data_name_list = data_name_list
        self.log_dir = log_dir
        self.model_cfg_kwargs = model_cfg_kwargs if model_cfg_kwargs is not None else {}

    def run(self, i_run=0):
        log_dir = makedirs(os.path.join(self.log_dir, str(i_run)))
        logger = set_logging(self.tag, log_dir)

        records = []
        n_datasets = len(self.data_name_list)
        for i, data_name in enumerate(self.data_name_list):
            logger.info(f"****** [{i+1}/{n_datasets}] process dataset {data_name}")
            _log_dir = os.path.join(log_dir, data_name)
            tf.keras.backend.clear_session()

            clf = Classifier(_log_dir, logger, self.model_obj, self.model_cfg_obj, -3, data_name, self.data_dir,
                             self.model_cfg_kwargs)
            res = clf.run()
            print(res)
            records.append(res)
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(log_dir, 'metrics.csv'), index=False)


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    log_dir = makedirs(os.path.join(DIR_LOG, tag))
    logger = set_logging(tag, log_dir)
    # data_name_list = UCR85_DATASETS
    data_name_list = ['ecg']

    model_cfg_kwargs = dict(acc_threshold_to_train_d=0.75, kernel_size=10)
    exp = ExperimentClf(tag, TCGAN, TCGANConfig, DIR_DATA_UCR15, data_name_list, log_dir,
                        model_cfg_kwargs=model_cfg_kwargs)
    fire.Fire(exp.run)

