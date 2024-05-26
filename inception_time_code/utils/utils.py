from builtins import print
import numpy as np
import pandas as pd
import matplotlib
import random

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import os
import operator
import utils

from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from utils.constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


def check_if_file_exits(file_name):
    return os.path.exists(file_name)


def readucr(filename, delimiter=','):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def readsits(filename, delimiter=','):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, -1]
    X = data[:, :-1]
    return X, Y


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def read_dataset(root_dir, archive_name, dataset_name):
    datasets_dict = {}

    file_name = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/' + dataset_name
    x_train, y_train = readucr(file_name + '_train.csv')
    #change add validation
    x_val, y_val = readucr(file_name + '_val.csv')
    x_test, y_test = readucr(file_name + '_test.csv')

    # datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
    #                                y_test.copy())
    #change
    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(),
                                   x_val.copy(), y_val.copy(),
                                   x_test.copy(),y_test.copy())


    return datasets_dict


def read_all_datasets(root_dir, archive_name):
    datasets_dict = {}

    dataset_names_to_sort = []

    if archive_name == 'TSC':
        for dataset_name in DATASET_NAMES:
            root_dir_dataset = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
            file_name = root_dir_dataset + dataset_name
            x_train, y_train = readucr(file_name + '_train.csv')
            #change
            x_val, y_val = readucr(file_name + '_val.csv')
            x_test, y_test = readucr(file_name + '_test.csv')
            #change
            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), 
                                           x_val.copy(),y_val.copy(),
                                           x_test.copy(),y_test.copy())

            dataset_names_to_sort.append((dataset_name, len(x_train)))

        dataset_names_to_sort.sort(key=operator.itemgetter(1))

        for i in range(len(DATASET_NAMES)):
            DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    elif archive_name == 'InlineSkateXPs':

        for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
            root_dir_dataset = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'

            x_train = np.load(root_dir_dataset + 'x_train.npy')
            y_train = np.load(root_dir_dataset + 'y_train.npy')
            #change
            x_val = np.load(root_dir_dataset + 'x_val.npy')
            y_val = np.load(root_dir_dataset + 'y_val.npy')
            x_test = np.load(root_dir_dataset + 'x_test.npy')
            y_test = np.load(root_dir_dataset + 'y_test.npy')
            
            #change
            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), 
                                           x_val.copy(), y_val.copy(), 
                                           x_test.copy(),y_test.copy())
    # elif archive_name == 'SITS':
    #     return read_sits_xps(root_dir)
    else:
        print('error in archive name')
        exit()

    return datasets_dict


def calculate_metrics(y_true, y_pred, duration):

    res = pd.DataFrame(data=np.zeros((1, 12), dtype=float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 
                                'precision_0', 'recall_0',
                                'precision_1', 'recall_1',
                                'precision_2','recall_2',
                                'precision_3', 'recall_3',
                                'duration'])
    
    res['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

    for i in range(4):
        y_true_i = [1 if label == i else 0 for label in y_true]
        y_pred_i = [1 if label == i else 0 for label in y_pred]
        
        res[f'precision_{i}'] = precision_score(y_true_i, y_pred_i, zero_division=0)
        res[f'recall_{i}'] = recall_score(y_true_i, y_pred_i, zero_division=0)

    clas_rep = classification_report(y_true, y_pred, target_names=['class 0', 'class 1', 'class 2','class 3'], zero_division=0)
    
    res['duration'] = duration


    #Accuracies using conf matrix
    #change
    res2 = pd.DataFrame(data=np.zeros((1, 4), dtype=float), index=[0],
                       columns=['accuracy_0', 'accuracy_1', 'accuracy_2', 'accuracy_3'])

    confusion_matrix1 = confusion_matrix(y_true, y_pred)

    num_classes = confusion_matrix1.shape[0]

    # Initialize an array to hold the accuracy for each class
    class_accuracies = np.zeros(num_classes)

    # Calculate the accuracy for each class
    for i in range(num_classes):
        TP = confusion_matrix1[i, i]
        FN = np.sum(confusion_matrix1[i, : ]) - TP
        FP = np.sum(confusion_matrix1[:, i]) - TP
        TN = np.sum(confusion_matrix1) - (TP + FN + FP)
        
        class_accuracies[i] = (TP + TN) / (TP + TN + FP + FN)
    
    for i in range(num_classes):
        res2[f'accuracy_{i}'] = class_accuracies[i]

    return res, clas_rep, res2 #change

# #for runnning without other rhythm
# def calculate_metrics(y_true, y_pred, duration):

#     res = pd.DataFrame(data=np.zeros((1, 10), dtype=float), index=[0],
#                        columns=['precision', 'accuracy', 'recall', 
#                                 'precision_0', 'recall_0',
#                                 'precision_1', 'recall_1',
#                                 'precision_2','recall_2',
#                                 'duration'])
    
#     res['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
#     res['accuracy'] = accuracy_score(y_true, y_pred)
#     res['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

#     for i in range(3):
#         y_true_i = [1 if label == i else 0 for label in y_true]
#         y_pred_i = [1 if label == i else 0 for label in y_pred]
        
#         res[f'precision_{i}'] = precision_score(y_true_i, y_pred_i, zero_division=0)
#         res[f'recall_{i}'] = recall_score(y_true_i, y_pred_i, zero_division=0)

#     clas_rep = classification_report(y_true, y_pred, target_names=['class 0', 'class 1', 'class 2'], zero_division=0)
    
#     res['duration'] = duration


#     #Accuracies using conf matrix
#     #change
#     res2 = pd.DataFrame(data=np.zeros((1, 3), dtype=float), index=[0],
#                        columns=['accuracy_0', 'accuracy_1', 'accuracy_2'])

#     confusion_matrix1 = confusion_matrix(y_true, y_pred)

#     num_classes = confusion_matrix1.shape[0]

#     # Initialize an array to hold the accuracy for each class
#     class_accuracies = np.zeros(num_classes)

#     # Calculate the accuracy for each class
#     for i in range(num_classes):
#         TP = confusion_matrix1[i, i]
#         FN = np.sum(confusion_matrix1[i, : ]) - TP
#         FP = np.sum(confusion_matrix1[:, i]) - TP
#         TN = np.sum(confusion_matrix1) - (TP + FN + FP)
        
#         class_accuracies[i] = (TP + TN) / (TP + TN + FP + FN)
    
#     for i in range(num_classes):
#         res2[f'accuracy_{i}'] = class_accuracies[i]

#     return res, clas_rep, res2 




def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def transform_labels(y_train, y_val, y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # no validation split
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit
    #change
    y_train_val_test = np.concatenate((y_train, y_val, y_test), axis=0)
    # fit the encoder
    encoder.fit(y_train_val_test)
    # transform to min zero and continuous labels
    new_y_train_val_test = encoder.transform(y_train_val_test)
    # resplit the train and test
    new_y_train = new_y_train_val_test[:len(y_train)]
    new_y_val = new_y_train_val_test[len(y_train):len(y_train)+len(y_val)]
    new_y_test = new_y_train_val_test[len(y_train)+len(y_val):]

    return new_y_train, new_y_val, new_y_test


def generate_results_csv(output_file_name, root_dir, clfs):
    res = pd.DataFrame(data=np.zeros((0, 8), dtype=float), index=[],
                       columns=['classifier_name', 'archive_name', 'dataset_name', 'iteration',
                                'precision', 'accuracy', 'recall', 'duration'])
    for archive_name in ARCHIVE_NAMES:
        datasets_dict = read_all_datasets(root_dir, archive_name)
        for classifier_name in clfs:
            durr = 0.0

            curr_archive_name = archive_name
            for dataset_name in datasets_dict.keys():
                output_dir = root_dir + '/results/' + classifier_name + '/' \
                             + curr_archive_name + '/' + dataset_name + '/' + 'df_metrics.csv'
                print(output_dir)
                if not os.path.exists(output_dir):
                    continue
                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['archive_name'] = archive_name
                df_metrics['dataset_name'] = dataset_name
                df_metrics['iteration'] = 0
                res = pd.concat((res, df_metrics), axis=0, sort=False)
                durr += df_metrics['duration'][0]

    res.to_csv(root_dir + output_file_name, index=False)

    res = res.loc[res['classifier_name'].isin(clfs)]

    return res


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()






def save_logs(output_directory, hist, y_pred, y_true, duration,
              lr=True, plot_test_acc=True):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    #comment :calculate test metrics
    df_metrics, class_rep, df_metrics2 = calculate_metrics(y_true, y_pred, duration)

    #comment :save test metrics
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)


    df_metrics2.to_csv(output_directory + 'df_accuracies.csv', index=False) #change


    full_path = output_directory + 'classification_report.csv'

    with open(full_path, "w") as file:
        file.write(class_rep)



    #change loss to validation loss
    index_best_model = hist_df['val_loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]



    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])
    
   

    # if 'acc' not in row_best_model.columns:
    #     raise ValueError("Column 'acc' not found in the DataFrame.")

    


    df_best_model['best_model_train_loss'] = row_best_model['loss']
    if plot_test_acc:
        print('plot test acc true')
        df_best_model['best_model_val_loss'] = row_best_model['val_loss']

    #change
    # df_best_model['best_model_train_acc'] = row_best_model.get('acc', None)
    print('row_best_model' , row_best_model) #comment:prints in command the best models validation metrics

    df_best_model['best_model_train_acc'] = row_best_model['accuracy']

    if plot_test_acc:
        df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    if plot_test_acc:
        # plot losses
        plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics


def generate_array_of_colors(n):
    # https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    alpha = 1.0
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r / 255, g / 255, b / 255, alpha))
    return ret



def resample_dataset(x, rate):
    new_x = np.zeros(shape=(x.shape[0], rate))
    from scipy import signal
    for i in range(x.shape[0]):
        f = signal.resample(x[0], rate)
        new_x[i] = f
    return new_x
