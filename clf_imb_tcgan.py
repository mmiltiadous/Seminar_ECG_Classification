import os
import fire

from mlpy.configure import DIR_DATA_UCR15

from configure import DIR_LOG
from tcgan.model.tcgan import TCGAN, TCGANConfig
from tcgan.lib.classifiers_imb import *

from tcgan.lib.exp_clf import ExperimentClf

if __name__ == '__main__':
    tag = 'exp_tcgan_' # the pre-trained model
    log_dir = os.path.join(DIR_LOG, tag)

    # imbalanced ratios >= 30
    # data_name_list = ['50words', 'ECG5000', 'MedicalImages', 'ProximalPhalanxTW', 'WordsSynonyms']
    data_name_list = ['ecg']

    model_cfg = dict(acc_threshold_to_train_d=0.75, kernel_size=10, norm='znorm')

    exp_tag = 'base_imbalanced_eval'
    exp_cfg = {
        'extractor_params': [
            {
                'idx_layer': 'conv_3_dropout',  # that is equal to 'idx_layer': -3,
                'pool': {
                    'obj': tf.keras.layers.MaxPool1D,
                    'kwargs': {
                        'pool_size': 2,
                        'strides': 1
                    }}
            },
        ],
        'norm': None,
        'classifiers': {
            'softmax': softmax,
            'knn': knn,
            'lr': lr,
            'lsvc': lsvc,
            'svc': svc
        },
        'out_file': 'clf_{}.json'.format(exp_tag),
        'res_out_fname': 'res_clf_{}'.format(exp_tag)
    }

    clf = ExperimentClf(tag, TCGAN, TCGANConfig, DIR_DATA_UCR15, data_name_list, log_dir,
                        model_cfg_kwargs=model_cfg,
                        **exp_cfg)

    fire.Fire(clf.run)
