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

    def fit(self, X, y, Z=None, unknown_allele=False):
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

        if Z is not None:
            pred = self.predict_log_odds(Z, True, False)
            alleles = pred.columns.to_numpy()
            argsort_res = (-pred.values).argsort()
            pred[['pred' + str(i) for i in range(len(alleles))]] = alleles[argsort_res]
            sorted_scores = pred[alleles].to_numpy()
            sorted_scores[:, ::-1].sort(axis=1)
            pred[['score' + str(i) for i in range(len(alleles))]] = sorted_scores

            if 'pred1' not in pred.columns:
                pred['pred1'] = pred['pred0']
                pred['score1'] = -float('inf')

            if unknown_allele:
                use_first = pred['score0'] > -50
                use_second = pred['score1'] > -50
            else:
                use_first = True
                use_second = True

            other = ~use_first & ~use_second
            if unknown_allele and other.sum() > 100:
                pred.loc[other, 'pred0'] = "OTHER"
                pred.loc[other, 'score0'] = 0
                use_first[other] = True
                alleles.append('OTHER')

            pool = mp.Pool(self.n_jobs)
            async_res = {}
            for sub_y in self.unique_y:
                self.models[sub_y] = NBScore(self.n_pan, self.n_spec, **self.kwargs)
                sub_X = (X[y == sub_y].tolist() +
                         pred.index[((pred['pred0'] == sub_y) & use_first) | ((pred['pred1'] == sub_y) & use_second)].tolist())
                async_res[sub_y] = pool.apply_async(self.models[sub_y].fit, (sub_X,))
            pool.close()
            for sub_y in self.unique_y:
                self.models[sub_y] = async_res[sub_y].get()

    def predict_log_odds(self, X, return_df=False, return_best=False):
        res = {}
        async_res = {}
        pool = mp.Pool(self.n_jobs)
        for sub_y in self.unique_y:
            async_res[sub_y] = pool.apply_async(self.models[sub_y].predict_log_odds, (X,))
        pool.close()
        for sub_y in self.unique_y:
            res[sub_y] = async_res[sub_y].get()
        if return_df:
            df = pd.DataFrame(res, index=X)
            if return_best:
                best_alleles = df.idxmax(axis=1).tolist()
                best_score = df.max(axis=1)
                df['best_allele'] = best_alleles
                df['best_score'] = best_score
            return df
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
