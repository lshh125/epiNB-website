from ._models import NBScore

import multiprocessing as mp
import numpy as np
import pandas as pd

class NBMulti:
    def __init__(self, n_pan=10, n_spec=10, n_jobs=None, **kwargs):
        self.models = {}
        self.n_pan = n_pan
        self.n_spec = n_spec
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.unique_y = np.unique(y)
        
        multiproc_res = []
        
        pool = mp.Pool(self.n_jobs)
        async_res = {}
        for sub_y in self.unique_y:
            self.models[sub_y] = NBScore(self.n_pan, self.n_spec, **self.kwargs)
            sub_X = X[y == sub_y]
            async_res[sub_y] = pool.apply_async(self.models[sub_y].fit, (sub_X, ))
        pool.close()
        for sub_y in self.unique_y:
            self.models[sub_y] = async_res[sub_y].get()
        
    def predict_log_odds(self, X, return_df=False):
        res = {}
        async_res = {}
        pool = mp.Pool(self.n_jobs)
        for sub_y in self.unique_y:
            async_res[sub_y] = pool.apply_async(self.models[sub_y].predict_log_odds, (X, ))
        pool.close()
        for sub_y in self.unique_y:
            res[sub_y] = async_res[sub_y].get()
        if return_df:
            return pd.DataFrame(res, index=X)
        else:
            return res
    
    def predict_log_proba(self, X, return_df=False):
        res = {}
        async_res = {}
        pool = mp.Pool(self.n_jobs)
        for sub_y in self.unique_y:
            async_res[sub_y] = pool.apply_async(self.models[sub_y].predict_log_proba, (X, ))
        pool.close()
        for sub_y in self.unique_y:
            res[sub_y] = async_res[sub_y].get()
        if return_df:
            return pd.DataFrame(res, index=X)
        else:
            return res
    
    def predict(self, X, return_series=False):
        if return_series:
            return self.predict_log_odds(X, True).idxmax(axis=1)
        else:
            return self.predict_log_odds(X, True).idxmax(axis=1).tolist()


class NBSemiSupervised:
    def __init__(self, n_pan=10, n_spec=10, n_jobs=None, **kwargs):
        self.models = {}
        self.n_pan = n_pan
        self.n_spec = n_spec
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.unique_y = np.unique(y)

        multiproc_res = []

        pool = mp.Pool(self.n_jobs)
        async_res = {}
        for sub_y in self.unique_y:
            self.models[sub_y] = NBScore(self.n_pan, self.n_spec, **self.kwargs)
            sub_X = X[y == sub_y]
            async_res[sub_y] = pool.apply_async(self.models[sub_y].fit, (sub_X,))
        pool.close()
        for sub_y in self.unique_y:
            self.models[sub_y] = async_res[sub_y].get()

    def predict_log_odds(self, X, return_df=False):
        res = {}
        async_res = {}
        pool = mp.Pool(self.n_jobs)
        for sub_y in self.unique_y:
            async_res[sub_y] = pool.apply_async(self.models[sub_y].predict_log_odds, (X,))
        pool.close()
        for sub_y in self.unique_y:
            res[sub_y] = async_res[sub_y].get()
        if return_df:
            return pd.DataFrame(res, index=X)
        else:
            return res

    def predict_log_proba(self, X, return_df=False):
        res = {}
        async_res = {}
        pool = mp.Pool(self.n_jobs)
        for sub_y in self.unique_y:
            async_res[sub_y] = pool.apply_async(self.models[sub_y].predict_log_proba, (X,))
        pool.close()
        for sub_y in self.unique_y:
            res[sub_y] = async_res[sub_y].get()
        if return_df:
            return pd.DataFrame(res, index=X)
        else:
            return res

    def predict(self, X, return_series=False):
        if return_series:
            return self.predict_log_odds(X, True).idxmax(axis=1)
        else:
            return self.predict_log_odds(X, True).idxmax(axis=1).tolist()