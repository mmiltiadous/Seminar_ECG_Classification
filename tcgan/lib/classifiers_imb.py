import tensorflow as tf
from time import time
import numpy as np

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

from mlpy.lib.data.utils import one_hot_to_dense, dense_to_one_hot

from tensorflow.keras.callbacks import ModelCheckpoint


def imbalanced_metrics(y_true, y_pred, **kwargs):
    res = {
        'f1_macro': f1_score(y_true, y_pred, average='macro', **kwargs),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', **kwargs)
    }
    return res


def softmax(x_tr, y_tr, x_te, y_te, x_val, y_val, counts_te, n_classes, epochs=100, batch_size=16, verbose=0):
    t0 = time()

    batch_size = 32
    epochs = 150

    if y_tr.ndim == 1:
        y_tr = dense_to_one_hot(y_tr, n_classes)
        y_te = dense_to_one_hot(y_te, n_classes)
    model = tf.keras.Sequential(
        tf.keras.layers.Dense(n_classes,
                              input_shape=x_tr.shape[1:],
                              activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath='best_model.h5',
                             monitor='val_loss',
                             save_best_only=True,
                             save_weights_only=False,
                             mode='min',
                             verbose=0)

    history = model.fit(x_tr, y_tr,
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(x_val, y_val), callbacks=[checkpoint])
    t = time() - t0

    model = tf.keras.models.load_model('best_model.h5')

    #ALTERAR AQUI. Y-PRED JA NAO E ESTE
    y_pred = model.predict(x_te)
    true_y_pred = one_hot_to_dense(y_pred)
    all_y_true = one_hot_to_dense(y_te)
    res_imb = imbalanced_metrics(one_hot_to_dense(y_te), one_hot_to_dense(y_pred))

    unique_values_p, counts_p = np.unique(all_y_true, return_counts=True)
    value_counts_p = dict(zip(unique_values_p, counts_p))
    print('\n', value_counts_p, '\n')

    acc_tr = model.evaluate(x_tr, y_tr, verbose=verbose)[1]
    #acc_te = model.evaluate(x_te, y_te, verbose=verbose)[1]


    #predicted_calculated, real_y_true = segmented_test(all_y_true, true_y_pred, counts_te)
    predicted_calculated = true_y_pred
    real_y_true = all_y_true


    unique_values_pred, counts_pred = np.unique(predicted_calculated, return_counts=True)
    unique_values_real, counts_real = np.unique(real_y_true, return_counts=True)

    # Combine the results into a dictionary for better readability
    value_counts_pred = dict(zip(unique_values_pred, counts_pred))
    value_counts_real = dict(zip(unique_values_real, counts_real))

    # Print the result
    print('\n', value_counts_pred, '\n', value_counts_real, '\n')

    acc_te = accuracy_score(real_y_true, predicted_calculated)



    res = {
        'acc_tr': acc_tr,
        'acc_te': acc_te,
        'time': t
    }
    #res.update(res_imb)
    print(res)

    return res


def standard_clf(x_tr, y_tr, x_te, y_te, x_val, y_val,counts_te, n_classes, model_cls, **kwargs):
    t0 = time()

    if y_tr.ndim == 2:
        y_tr = one_hot_to_dense(y_tr)
        y_te = one_hot_to_dense(y_te)

    model = model_cls(**kwargs)
    model.fit(x_tr, y_tr)
    acc_tr = accuracy_score(y_tr, model.predict(x_tr))
    acc_te = accuracy_score(y_te, model.predict(x_te))
    t = time() - t0

    y_pred = model.predict(x_te)
    res_imb = imbalanced_metrics(y_te, y_pred)

    #true_y_pred = one_hot_to_dense(y_pred)
    #all_y_true = one_hot_to_dense(y_te)


    #predicted_calculated, real_y_true = segmented_test(y_te, y_pred, counts_te)
    predicted_calculated = y_pred
    real_y_true = y_te




    unique_values_pred, counts_pred = np.unique(predicted_calculated, return_counts=True)
    unique_values_real, counts_real = np.unique(real_y_true, return_counts=True)

    # Combine the results into a dictionary for better readability
    value_counts_pred = dict(zip(unique_values_pred, counts_pred))
    value_counts_real = dict(zip(unique_values_real, counts_real))

    # Print the result
    print('\n', value_counts_pred, '\n', value_counts_real, '\n')

    acc_te = accuracy_score(real_y_true, predicted_calculated)

    res = {
        'acc_tr': acc_tr,
        'acc_te': acc_te,
        'time': t
    }
    #res.update(res_imb)
    print(res)
    return res


def lr(x_tr, y_tr, x_te, y_te, x_val, y_val,counts_te, n_classes):
    kwargs = dict(max_iter=500)
    return standard_clf(x_tr, y_tr, x_te, y_te, x_val, y_val,counts_te, n_classes, LogisticRegression, **kwargs)


def lsvc(x_tr, y_tr, x_te, y_te, x_val, y_val,counts_te, n_classes):
    kwargs = dict(max_iter=500)
    return standard_clf(x_tr, y_tr, x_te, y_te, x_val, y_val,counts_te,n_classes, LinearSVC, **kwargs)


def svc(x_tr, y_tr, x_te, y_te, x_val, y_val,counts_te, n_classes):
    kwargs = dict(max_iter=500)
    return standard_clf(x_tr, y_tr, x_te, y_te, x_val, y_val,counts_te,n_classes, SVC, **kwargs)


def knn(x_tr, y_tr, x_te, y_te, x_val, y_val,counts_te, n_classes):
    kwargs = dict(n_neighbors=1)
    return standard_clf(x_tr, y_tr, x_te, y_te, x_val, y_val,counts_te,n_classes, KNeighborsClassifier, **kwargs)


def model_search(x_tr, y_tr, x_te, y_te,x_val, y_val, counts_te, n_classes, model_cls, cv=5, n_jobs=1):
    t_0 = time()

    if y_tr.ndim == 2:
        y_tr = one_hot_to_dense(y_tr)
        y_te = one_hot_to_dense(y_te)

    # select penalty
    C = np.inf
    max_iter = 500
    model = model_cls(C=C, max_iter=max_iter)
    if x_tr.shape[0] // n_classes < 5 or x_tr.shape[0] < 50:
        model.fit(x_tr, y_tr)
    else:
        param_grid = {
            'C': [
                0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                np.inf
            ],
            'max_iter': [max_iter],
        }
        gs = GridSearchCV(
            model, param_grid, cv=cv, n_jobs=n_jobs, error_score=0
        )
        gs.fit(x_tr, y_tr)

        model = gs.best_estimator_
        C = gs.best_params_['C']

    acc_tr = accuracy_score(y_tr, model.predict(x_tr))
    acc_te = accuracy_score(y_te, model.predict(x_te))
    t = time() - t_0

    res = {
        'acc_tr': acc_tr,
        'acc_te': acc_te,
        'time': t,
        'C': C
    }
    return res


def lr_search(x_tr, y_tr, x_te, y_te, x_val, y_val,counts_te, n_classes, cv=5, n_jobs=1):
    return model_search(x_tr, y_tr, x_te, y_te, n_classes, LogisticRegression, cv=cv, n_jobs=n_jobs)


def lsvc_search(x_tr, y_tr, x_te, y_te, x_val, y_val,counts_te, n_classes, cv=5, n_jobs=1):
    return model_search(x_tr, y_tr, x_te, y_te, n_classes, LinearSVC, cv=cv, n_jobs=n_jobs)


def svc_search(x_tr, y_tr, x_te, y_te, x_val, y_val,counts_te, n_classes, cv=5, n_jobs=1):
    return model_search(x_tr, y_tr, x_te, y_te, n_classes, SVC, cv=cv, n_jobs=n_jobs)



def segmented_test(all_y_true, true_y_pred, counts_te):
    amount_counts_te = []
    current = 0
    count = 0
    for i in range(counts_te):
        if i == current:
            count += 1
        else:
            current = i
            amount_counts_te.append(count)
            count = 0


    index = 0
    predicted_calculated = []
    real_y_true = []
    temp_predicted = []
    
    for i in amount_counts_te:
        real_y_true.append(all_y_true[index])
        for j in range(i):
            pred = true_y_pred[index]
            temp_predicted.append(pred)
            index += 1

        predicted_calculated.append(int(np.round(np.mean(np.array(temp_predicted)), 0)))

    return predicted_calculated, real_y_true

