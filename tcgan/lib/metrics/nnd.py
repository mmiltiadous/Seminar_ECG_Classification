import numpy as np
from sklearn import metrics

"""
     Basic operators of numpy
"""


def intX(X):
    return np.asarray(X, dtype=np.int32)


def floatX(X):
    return np.asarray(X, dtype=np.float32)  # Using float32 instead of theano's floatX


def sharedX(X, dtype=np.float32, name=None):
    return np.asarray(X, dtype=dtype)  # No need for shared variables in numpy


def shared0s(shape, dtype=np.float32, name=None):
    return np.zeros(shape, dtype=dtype)


def sharedNs(shape, n, dtype=np.float32, name=None):
    return np.ones(shape) * n


"""
    similarity metrics
"""


def l2normalize(x, axis=1, e=1e-8, keepdims=True):
    return x / l2norm(x, axis=axis, e=e, keepdims=keepdims)


def l2norm(x, axis=1, e=1e-8, keepdims=True):
    return np.sqrt(np.sum(np.square(x), axis=axis, keepdims=keepdims) + e)


def cosine(x, y):
    d = np.dot(x, y.T)
    d /= l2norm(x).reshape(x.shape[0], 1)
    d /= l2norm(y).reshape(1, y.shape[0])
    return d


def euclidean(x, y, e=1e-8):
    xx = np.square(np.sqrt((x * x).sum(axis=1) + e))
    yy = np.square(np.sqrt((y * y).sum(axis=1) + e))
    dist = np.dot(x, y.T)
    dist *= -2
    dist += xx.reshape(xx.shape[0], 1)
    dist += yy.reshape(1, yy.shape[0])
    dist = np.sqrt(dist)
    return dist


"""
    GPU calculations (not GPU-based anymore)
"""


def gpu_nnc_predict(trX, trY, teX, metric='cosine', batch_size=4096):
    if metric == 'cosine':
        metric_fn = cosine
    else:
        metric_fn = euclidean
    idxs = []
    for i in range(0, len(teX), batch_size):
        mb_dists = []
        mb_idxs = []
        for j in range(0, len(trX), batch_size):
            dist = metric_fn(floatX(teX[i:i + batch_size]), floatX(trX[j:j + batch_size]))
            if metric == 'cosine':
                mb_dists.append(np.max(dist, axis=1))
                mb_idxs.append(j + np.argmax(dist, axis=1))
            else:
                mb_dists.append(np.min(dist, axis=1))
                mb_idxs.append(j + np.argmin(dist, axis=1))
        mb_idxs = np.asarray(mb_idxs)
        mb_dists = np.asarray(mb_dists)
        if metric == 'cosine':
            i = mb_idxs[np.argmax(mb_dists, axis=0), np.arange(mb_idxs.shape[1])]
        else:
            i = mb_idxs[np.argmin(mb_dists, axis=0), np.arange(mb_idxs.shape[1])]
        idxs.append(i)
    idxs = np.concatenate(idxs, axis=0)
    nearest = trY[idxs]
    return nearest


def gpu_nnd_score(trX, teX, metric='cosine', batch_size=4096):
    if metric == 'cosine':
        metric_fn = cosine
    else:
        metric_fn = euclidean
    dists = []
    for i in range(0, len(teX), batch_size):
        mb_dists = []
        for j in range(0, len(trX), batch_size):
            dist = metric_fn(floatX(teX[i:i + batch_size]), floatX(trX[j:j + batch_size]))
            if metric == 'cosine':
                mb_dists.append(np.max(dist, axis=1))
            else:
                mb_dists.append(np.min(dist, axis=1))
        mb_dists = np.asarray(mb_dists)
        if metric == 'cosine':
            d = np.max(mb_dists, axis=0)
        else:
            d = np.min(mb_dists, axis=0)
        dists.append(d)
    dists = np.concatenate(dists, axis=0)
    return float(np.mean(dists))


"""
    Unified entrances.
    Note: the batch_size=4096 is too large for some datasets, for example the UCR/UEA datasets.
"""


def nnc_score(trX, trY, teX, teY, metric='euclidean', batch_size=64):
    pred = gpu_nnc_predict(trX, trY, teX, metric=metric, batch_size=batch_size)
    acc = metrics.accuracy_score(teY, pred)
    return acc * 100.


def nnd_score(trX, teX, metric='euclidean', batch_size=64):
    return gpu_nnd_score(trX, teX, metric=metric, batch_size=batch_size)
