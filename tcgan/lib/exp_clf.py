"""
    Extend the base clf. For example, the extractor can process some customized settings.
"""
import os
import json
from copy import deepcopy

import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from mlpy.lib.tfops.base import tf_keras_set_gpu_allow_growth
from mlpy.datasets import ucr_uea

from tcgan.lib.exp import ExpUnitClfData, Experiment
from tcgan.lib.utils import extract
from tcgan.lib.classfiers import *

tf_keras_set_gpu_allow_growth()

import pandas as pd


class ExpUnitClf(ExpUnitClfData):
    # res_eval_fnames = ['clf.json']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_model()

    def _load_model(self):
        #self.model = self.model_obj(self.model_cfg, None)
        #self.model = tf.keras.models.load_model('/Users/fiarresga/Desktop/2024/SADL/sadl2024-tcgan/saved_model.pb')
        #self.trained_epoch = self.model.load()
        pass

    def get_base_model(self):
        #return self.model.discriminator
        return tf.keras.models.load_model('/Users/fiarresga/Desktop/2024/SADL/sadl2024-tcgan/saved_model_2715.h5')

    def get_extractor(self, idx_layer, pool=None):
        base_model = self.get_base_model()

        inputs = base_model.input
        if isinstance(idx_layer, int):
            base_output = base_model.layers[idx_layer].output
        else:
            base_output = base_model.get_layer(idx_layer).output

        if pool is None:
            h = base_output
        else:
            kwargs = deepcopy(pool['kwargs'])
            if kwargs['pool_size'] < 1.0:
                kwargs['pool_size'] = int(base_output.shape[1] * kwargs['pool_size'])
            if kwargs['pool_size'] >= base_output.shape[1]:
                kwargs['pool_size'] = base_output.shape[1] // 2
            pool_layer = pool['obj'](**kwargs)
            h = pool_layer(base_output)
        if len(h.shape) <= 2:  # 1-dimensional vector, needn't Flatten
            flat = h
        else:
            flat = tf.keras.layers.Flatten()(h)

        extractor = tf.keras.models.Model(inputs=inputs, outputs=flat)
        extractor.trainable = False

        return extractor

    def extract_features(self, extractor_list):
        feature_list_tr = []
        feature_list_te = []
        feature_list_val = []
        for e in extractor_list:
            feat_tr = extract(e, self.x_tr, self.model_cfg.batch_size)
            feat_te = extract(e, self.x_te, self.model_cfg.batch_size)
            feat_val = extract(e, self.x_val, self.model_cfg.batch_size)
            feature_list_tr.append(feat_tr)
            feature_list_te.append(feat_te)
            feature_list_val.append(feat_val)

        feat_tr = np.hstack(feature_list_tr)
        feat_te = np.hstack(feature_list_te)
        feat_val = np.hstack(feature_list_val)

        return feat_tr, feat_te, feat_val

    def prepare_data(self):
        extractor_list = []
        for param in self.kwargs['extractor_params']:
            idx_layer = param['idx_layer']
            pool = param['pool']
            extractor = self.get_extractor(idx_layer, pool=pool)
            extractor_list.append(extractor)

        feat_tr, feat_te, feat_val = self.extract_features(extractor_list)

        norm = self.kwargs['norm']
        #CHANGED HERE
        #if norm is None:
            #pass
        #elif norm == 'znorm':
        print('\n Normalized \n')
        scaler = StandardScaler()
        scaler.fit(feat_tr, self.y_tr)
        feat_tr = scaler.transform(feat_tr)
        feat_te = scaler.transform(feat_te)
        feat_val = scaler.transform(feat_val)
        #else:
            #raise ValueError(f"norm={norm} can not be found!")
        return feat_tr, feat_te, feat_val

    def run(self):
        t_start = time()
        feat_tr, feat_te, feat_val = self.prepare_data()
        t_encode = time() - t_start

        clfs = self.kwargs['classifiers']
        res = {}
        res['time_encode'] = t_encode
        for name, clf in clfs.items():
            print(f"processing {name}")
            t_start = time()
            _res = clf(feat_tr, self.y_tr, feat_te, self.y_te, feat_val, self.y_val, self.counts_te, self.n_classes)
            t_clf = time() - t_start
            for k, v in _res.items():
                res[f'{name}_{k}'] = v
            res[f'time_{name}'] = t_clf

        out_file = self.kwargs['out_file']
        with open(os.path.join(self.model_cfg.eval_dir, out_file), 'a') as f:
            f.write(json.dumps(res) + "\n")

        tf.keras.backend.clear_session()


class ExpUnitClfUCR(ExpUnitClf):
    def _init_raw_data(self):
        """x_tr, y_tr, x_te, y_te, n_classes = ucr_uea.load_ucr(self.data_name, self.data_dir, one_hot=True)
        x_tr = x_tr[..., np.newaxis]
        x_te = x_te[..., np.newaxis]

        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te
        self.n_classes = n_classes
        self.input_shape = self.x_tr.shape[1:]"""

        df_train = pd.read_csv('/Users/fiarresga/Desktop/2024/SADL/DATA_SADL/mydata2715/mydata2715_train.csv')
        #df_train = df_train[df_train.iloc[:, 0] != 3]
        #df_train = df_train[df_train.iloc[:, 0] != 2]
        #df_train = df_train.iloc[:20000, :]
        print('\n DF TRAIN SHAPE', df_train.shape, '\n')
        classes = df_train.iloc[:, 0].values
        time_series_data = df_train.iloc[:, 1:].values
        y_tr = np.array(classes)
        x_tr = np.array(time_series_data)

        df_val = pd.read_csv('/Users/fiarresga/Desktop/2024/SADL/DATA_SADL/mydata2715/mydata2715_val.csv')
        #df_val = df_val[df_val.iloc[:, 0] != 3]
        #df_val = df_val[df_val.iloc[:, 0] != 2]
        classes = df_val.iloc[:, 0].values
        #CHANGED FOR T-SNE
        time_series_data = df_val.iloc[:, 1:].values
        #time_series_data = df_val.iloc[:, 2:].values
        y_val = np.array(classes)
        x_val = np.array(time_series_data)

        df_test = pd.read_csv('/Users/fiarresga/Desktop/2024/SADL/DATA_SADL/mydata2715/mydata2715_test.csv')
        #df_test = df_test[df_test.iloc[:, 0] != 3]
        #df_test = df_test[df_test.iloc[:, 0] != 2]
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


class ExperimentClf(Experiment):
    def __init__(self, tag, model_obj, model_cfg_obj, data_dir, data_name_list, log_dir,
                 exp_uni_obj=None,
                 model_cfg_kwargs=None,
                 **kwargs):
        if exp_uni_obj is None:
            exp_uni_obj = ExpUnitClfUCR
        training = False
        res_eval_fnames = [kwargs['out_file']]
        super().__init__(tag, model_obj, model_cfg_obj, data_dir, data_name_list, log_dir,
                         exp_uni_obj=exp_uni_obj,
                         res_eval_fnames=res_eval_fnames,
                         model_cfg_kwargs=model_cfg_kwargs,
                         training=training, **kwargs)
