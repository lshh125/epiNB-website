import pandas as pd
import time
import numpy as np
import os
import scipy.stats
import warnings

from typing import List, Union, Iterable, Tuple, Optional

# from numba import jit

# Mapping from letters (for AAs) to numbers
aa2int_mapping = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4,
                  "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
                  "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
                  "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19}


def py2naming(x: Union[int, Iterable[int]]) -> str:
    """Mapping from python style subscripts to peptide location naming.
    Python:   0  1  2  3  4  ... -4 -3 -2 -1
    Location: 1  2  3  4  5  ...  7  8  9  0
    Aka:      1  2  3  4  5  ...           Ω

    :param x: subscripts to be converted.
    :return: converted location names
    """
    if isinstance(x, int):
        return str((x + 1) % 10)
    else:
        return "".join((py2naming(i) for i in x))


aa_prior = [0.0777, 0.0157, 0.0530, 0.0656, 0.0405,
            0.0691, 0.0227, 0.0591, 0.0595, 0.0960,
            0.0238, 0.0427, 0.0469, 0.0393, 0.0526,
            0.0694, 0.0550, 0.0667, 0.0118, 0.0311]

blosum62 = np.array([[4., 0., -2., -1., -2., 0., -2., -1., -1., -1., -1., -2., -1., -1., -1., 1., 0., 0., -3., -2.],
                     [0., 9., -3., -4., -2., -3., -3., -1., -3., -1., -1., -3., -3., -3., -3., -1., -1., -1., -2., -2.],
                     [-2., -3., 6., 2., -3., -1., -1., -3., -1., -4., -3., 1., -1., 0., -2., 0., -1., -3., -4., -3.],
                     [-1., -4., 2., 5., -3., -2., 0., -3., 1., -3., -2., 0., -1., 2., 0., 0., -1., -2., -3., -2.],
                     [-2., -2., -3., -3., 6., -3., -1., 0., -3., 0., 0., -3., -4., -3., -3., -2., -2., -1., 1., 3.],
                     [0., -3., -1., -2., -3., 6., -2., -4., -2., -4., -3., 0., -2., -2., -2., 0., -2., -3., -2., -3.],
                     [-2., -3., -1., 0., -1., -2., 8., -3., -1., -3., -2., 1., -2., 0., 0., -1., -2., -3., -2., 2.],
                     [-1., -1., -3., -3., 0., -4., -3., 4., -3., 2., 1., -3., -3., -3., -3., -2., -1., 3., -3., -1.],
                     [-1., -3., -1., 1., -3., -2., -1., -3., 5., -2., -1., 0., -1., 1., 2., 0., -1., -2., -3., -2.],
                     [-1., -1., -4., -3., 0., -4., -3., 2., -2., 4., 2., -3., -3., -2., -2., -2., -1., 1., -2., -1.],
                     [-1., -1., -3., -2., 0., -3., -2., 1., -1., 2., 5., -2., -2., 0., -1., -1., -1., 1., -1., -1.],
                     [-2., -3., 1., 0., -3., 0., 1., -3., 0., -3., -2., 6., -2., 0., 0., 1., 0., -3., -4., -2.],
                     [-1., -3., -1., -1., -4., -2., -2., -3., -1., -3., -2., -2., 7., -1., -2., -1., -1., -2., -4.,
                      -3.],
                     [-1., -3., 0., 2., -3., -2., 0., -3., 1., -2., 0., 0., -1., 5., 1., 0., -1., -2., -2., -1.],
                     [-1., -3., -2., 0., -3., -2., 0., -3., 2., -2., -1., 0., -2., 1., 5., -1., -1., -3., -3., -2.],
                     [1., -1., 0., 0., -2., 0., -1., -2., 0., -2., -1., 1., -1., 0., -1., 4., 1., -2., -3., -2.],
                     [0., -1., -1., -1., -2., -2., -2., -1., -1., -1., -1., 0., -1., -1., -1., 1., 5., 0., -2., -2.],
                     [0., -1., -3., -2., -1., -3., -3., 3., -2., 1., 1., -3., -2., -2., -3., -2., 0., 4., -3., -1.],
                     [-3., -2., -4., -3., 1., -2., -2., -3., -3., -2., -1., -4., -4., -2., -3., -3., -2., -3., 11., 2.],
                     [-2., -2., -3., -2., 3., -3., 2., -1., -2., -1., -1., -2., -3., -1., -2., -2., -2., -1., 2., 7.]])

blosum62_q = np.array([[0.0215, 0.0016, 0.0022, 0.003, 0.0016, 0.0058, 0.0011, 0.0032, 0.0033, 0.0044, 0.0013, 0.0019,
                        0.0022, 0.0019, 0.0023, 0.0063, 0.0037, 0.0051, 0.0004, 0.0013],
                       [0.0016, 0.0119, 0.0004, 0.0004, 0.0005, 0.0008, 0.0002, 0.0011, 0.0005, 0.0016, 0.0004, 0.0004,
                        0.0004, 0.0003, 0.0004, 0.001, 0.0009, 0.0014, 0.0001, 0.0003],
                       [0.0022, 0.0004, 0.0213, 0.0049, 0.0008, 0.0025, 0.001, 0.0012, 0.0024, 0.0015, 0.0005, 0.0037,
                        0.0012, 0.0016, 0.0016, 0.0028, 0.0019, 0.0013, 0.0002, 0.0006],
                       [0.003, 0.0004, 0.0049, 0.0161, 0.0009, 0.0019, 0.0014, 0.0012, 0.0041, 0.002, 0.0007, 0.0022,
                        0.0014, 0.0035, 0.0027, 0.003, 0.002, 0.0017, 0.0003, 0.0009],
                       [0.0016, 0.0005, 0.0008, 0.0009, 0.0183, 0.0012, 0.0008, 0.003, 0.0009, 0.0054, 0.0012, 0.0008,
                        0.0005, 0.0005, 0.0009, 0.0012, 0.0012, 0.0026, 0.0008, 0.0042],
                       [0.0058, 0.0008, 0.0025, 0.0019, 0.0012, 0.0378, 0.001, 0.0014, 0.0025, 0.0021, 0.0007, 0.0029,
                        0.0014, 0.0014, 0.0017, 0.0038, 0.0022, 0.0018, 0.0004, 0.0008],
                       [0.0011, 0.0002, 0.001, 0.0014, 0.0008, 0.001, 0.0093, 0.0006, 0.0012, 0.001, 0.0004, 0.0014,
                        0.0005, 0.001, 0.0012, 0.0011, 0.0007, 0.0006, 0.0002, 0.0015],
                       [0.0032, 0.0011, 0.0012, 0.0012, 0.003, 0.0014, 0.0006, 0.0184, 0.0016, 0.0114, 0.0025, 0.001,
                        0.001, 0.0009, 0.0012, 0.0017, 0.0027, 0.012, 0.0004, 0.0014],
                       [0.0033, 0.0005, 0.0024, 0.0041, 0.0009, 0.0025, 0.0012, 0.0016, 0.0161, 0.0025, 0.0009, 0.0024,
                        0.0016, 0.0031, 0.0062, 0.0031, 0.0023, 0.0019, 0.0003, 0.001],
                       [0.0044, 0.0016, 0.0015, 0.002, 0.0054, 0.0021, 0.001, 0.0114, 0.0025, 0.0371, 0.0049, 0.0014,
                        0.0014, 0.0016, 0.0024, 0.0024, 0.0033, 0.0095, 0.0007, 0.0022],
                       [0.0013, 0.0004, 0.0005, 0.0007, 0.0012, 0.0007, 0.0004, 0.0025, 0.0009, 0.0049, 0.004, 0.0005,
                        0.0004, 0.0007, 0.0008, 0.0009, 0.001, 0.0023, 0.0002, 0.0006],
                       [0.0019, 0.0004, 0.0037, 0.0022, 0.0008, 0.0029, 0.0014, 0.001, 0.0024, 0.0014, 0.0005, 0.0141,
                        0.0009, 0.0015, 0.002, 0.0031, 0.0022, 0.0012, 0.0002, 0.0007],
                       [0.0022, 0.0004, 0.0012, 0.0014, 0.0005, 0.0014, 0.0005, 0.001, 0.0016, 0.0014, 0.0004, 0.0009,
                        0.0191, 0.0008, 0.001, 0.0017, 0.0014, 0.0012, 0.0001, 0.0005],
                       [0.0019, 0.0003, 0.0016, 0.0035, 0.0005, 0.0014, 0.001, 0.0009, 0.0031, 0.0016, 0.0007, 0.0015,
                        0.0008, 0.0073, 0.0025, 0.0019, 0.0014, 0.0012, 0.0002, 0.0007],
                       [0.0023, 0.0004, 0.0016, 0.0027, 0.0009, 0.0017, 0.0012, 0.0012, 0.0062, 0.0024, 0.0008, 0.002,
                        0.001, 0.0025, 0.0178, 0.0023, 0.0018, 0.0016, 0.0003, 0.0009],
                       [0.0063, 0.001, 0.0028, 0.003, 0.0012, 0.0038, 0.0011, 0.0017, 0.0031, 0.0024, 0.0009, 0.0031,
                        0.0017, 0.0019, 0.0023, 0.0126, 0.0047, 0.0024, 0.0003, 0.001],
                       [0.0037, 0.0009, 0.0019, 0.002, 0.0012, 0.0022, 0.0007, 0.0027, 0.0023, 0.0033, 0.001, 0.0022,
                        0.0014, 0.0014, 0.0018, 0.0047, 0.0125, 0.0036, 0.0003, 0.0009],
                       [0.0051, 0.0014, 0.0013, 0.0017, 0.0026, 0.0018, 0.0006, 0.012, 0.0019, 0.0095, 0.0023, 0.0012,
                        0.0012, 0.0012, 0.0016, 0.0024, 0.0036, 0.0196, 0.0004, 0.0015],
                       [0.0004, 0.0001, 0.0002, 0.0003, 0.0008, 0.0004, 0.0002, 0.0004, 0.0003, 0.0007, 0.0002, 0.0002,
                        0.0001, 0.0002, 0.0003, 0.0003, 0.0003, 0.0004, 0.0065, 0.0009],
                       [0.0013, 0.0003, 0.0006, 0.0009, 0.0042, 0.0008, 0.0015, 0.0014, 0.001, 0.0022, 0.0006, 0.0007,
                        0.0005, 0.0007, 0.0009, 0.001, 0.0009, 0.0015, 0.0009, 0.0102]])

positions = [0, 1, 2, 3, 4, -4, -3, -2, -1]


class NBScore:
    def __init__(self, n_pan: int = 10, n_spec: int = 10,
                 pan_feature_candidates: List = None, *,
                 smoothing_strength_1: float = 0., smoothing_strength_2: float = 0.):
        """Create an epiNB model.

        :param n_pan: Number of pan-allelic 2nd-order motifs used in prediction. Default/recommended: 10.
        :param n_spec: Number of allele-specific 2nd-order motifs used in prediction. Default/recommended: 10.
        :param pan_feature_candidates: Customize pan-allelic 2nd-order motifs.
            Default/recommended: [(1, -1), (0, 1), (4, -4), (1, 2), (0, -1),
            (-2, -1), (2, 4), (-3, -1), (2, -1), (1, 3),
            (0, 2)]
        :param smoothing_strength_1: Smoothing used for 1st-order motifs using BLOSUM62. Default/recommended: 0.
        :param smoothing_strength_2: Smoothing used for 2nd-order motifs using BLOSUM62. Default/recommended: 0.
        """
        # Feature preparation
        self.n_pan = n_pan
        self.n_spec = n_spec
        self.positions = positions

        if pan_feature_candidates is None:
            # ['20', '12', '56', '23', '10', '80', '35', '70', '30', '24', '13']
            pan_feature_candidates = [(1, -1), (0, 1), (4, -4), (1, 2), (0, -1),
                                      (-2, -1), (2, 4), (-3, -1), (2, -1), (1, 3), (0, 2)]

        if n_pan < len(pan_feature_candidates):
            self.pan_features = pan_feature_candidates[:n_pan]
        else:
            raise ValueError(f"{n_pan = } must <= {len(pan_feature_candidates) = }.")

        # Empty slots for allele-specific features to be filled in `fit`
        self.spec_features = None
        self.both_features = None

        # Smoothing preparation
        # self.smoothing_matrix_1 = 2 ** (blosum62 / 2)
        self.smoothing_matrix_1 = blosum62_q
        self.smoothing_matrix_1 /= np.diag(self.smoothing_matrix_1)
        self.smoothing_matrix_1 = self.smoothing_matrix_1.T
        self.smoothing_matrix_2 = np.zeros([400, 400])
        temp = self.smoothing_matrix_1 * np.sqrt(smoothing_strength_2)
        np.fill_diagonal(temp, 1.)
        for a in range(20):
            for b in range(20):
                for i in range(20):
                    self.smoothing_matrix_2[a * 20 + b, i * 20:(i + 1) * 20] = temp[a, i] * temp[b, :]
        self.smoothing_matrix_1 *= smoothing_strength_1
        np.fill_diagonal(self.smoothing_matrix_1, 1.)

        # Prepare frequencies of "imaginary negative examples"
        self.neg_freq_1 = np.array(aa_prior)
        self.log_neg_prob_1 = np.log(self.neg_freq_1)

        self.neg_freq_2 = np.outer(aa_prior, aa_prior).reshape([-1])
        self.neg_freq_2 /= self.neg_freq_2.sum()
        self.log_neg_prob_2 = np.log(self.neg_freq_2)

        # Empty slots for frequency and log frequencies of positive examples. To be filled in by `fit`.
        self.pos_freq_1 = None
        self.pos_freq_2 = None
        self.log_pos_prob_1 = None
        self.log_pos_prob_2 = None

    def seq2matrix(self, peptides: Iterable[str], no_warning: bool = False, return_ind: bool = False,
                   min_len: int = 0, max_len: int = 100) -> Tuple[Optional[List[int]], np.ndarray]:
        """Helper function to convert sequences to a matrix

        :param peptides: Peptides to be processed.
        :param no_warning: If true, do not warn when ignoring a peptide for unknown AAs (e.g. X)
        :param return_ind: If true, return the indices of the kept peptides in the input.
            This helps to align the results with the input, even when some inputs are filtered out.
        :param min_len: minimum length of the peptide to be count in. Discard otherwise.
        :param max_len: maximum length of the peptide to be count in. Discard otherwise.
        :return: The matrix (in numpy) or the indices and the matrix when requested.
        """
        res = []
        index = []
        for i, peptide in enumerate(peptides):
            temp = []
            try:
                if min_len <= len(peptide) <= max_len:
                    for j in self.positions:
                        temp.append(aa2int_mapping[peptide[j]])
                    res.append(temp)
                    index.append(i)
            except KeyError:
                if not no_warning:
                    warnings.warn(f'{peptide} ignored.')
        if return_ind:
            return index, np.array(res)
        else:
            return np.array(res)

    def counter_1(self, X: np.ndarray) -> np.ndarray:
        """Counter for 1st order motifs.

        :param X: The peptide matrix
        :return: A matrix representing the frequency of each AA at each position
        """
        freq = np.zeros([len(self.positions), 20])
        for i in range(len(X)):
            for j in self.positions:
                freq[j, X[i][j]] += 1
        return freq

    def counter_2(self, X: np.ndarray, features: List[Tuple[int, int]]) -> np.ndarray:
        """Counter for 1st order motifs.

        :param X: The peptide matrix
        :param features: The requested 2nd order motifs
        :return: A matrix representing the frequency of each AA combination (400 in total) at each 2nd order motifs.
        """
        freq = np.zeros([len(features), 400])
        for i in range(len(X)):
            for index, (j, k) in enumerate(features):
                freq[index, X[i][j] * 20 + X[i][k]] += 1
        return freq

    def spec_feature_selection(self, X: np.ndarray, n_spec: int) -> list[tuple[int, int]]:
        """Select allele-specific 2nd order motifs

        :param X: The peptide matrix.
        :param n_spec: Number of motifs.
        :return: A list of motifs
        """
        freq1 = self.counter_1(X)

        entropy1 = {}
        entropy2 = {}
        mutual_info = {}

        # Calculate entropy of each position
        for i in self.positions:
            entropy1[i] = scipy.stats.entropy(freq1[i, :])

        # Generate a list of all combinations of positions and calculate their entropies
        candidate_features = [(self.positions[i], self.positions[j]) for i in range(len(self.positions))
                              for j in range(i + 1, len(self.positions))]
        freq2 = self.counter_2(X, candidate_features)
        for index, (j, k) in enumerate(candidate_features):
            entropy2[(j, k)] = scipy.stats.entropy(freq2[index, :])

        # Calculate mutual information from the two sets of entropies
        for i in candidate_features:
            mutual_info[i] = entropy1[i[0]] + entropy1[i[1]] - entropy2[i]

        mutual_info_keys = list(mutual_info.keys())
        mutual_info_values = list(mutual_info.values())

        self.features = self.pan_features.copy()

        spec_features = []
        for i in reversed(np.argsort(mutual_info_values)):
            if mutual_info_keys[i] not in self.pan_features:
                spec_features.append(mutual_info_keys[i])
                if len(spec_features) >= n_spec:
                    break
        self.entropy1 = entropy1
        self.mutual_info = mutual_info
        return spec_features

    def fit(self, X: Iterable[str], min_len: int = 8, max_len: int = 11):
        """Fit the model.

        :param X: Peptides
        :param min_len: minimum length of the peptide to be count in. Discard otherwise.
        :param max_len: maximum length of the peptide to be count in. Discard otherwise.
        :return: Fitted model (`self`)
        """
        X = self.seq2matrix(X, min_len=min_len, max_len=max_len)
        self.spec_features = self.spec_feature_selection(X, self.n_spec)
        self.both_features = self.pan_features + self.spec_features

        def fit_core(X, positions, features, smoothing_matrix_1, smoothing_matrix_2):
            """Calculate the frequency matrices"""
            pos_freq_1 = np.zeros((len(positions), 20))
            for i in range(len(X)):
                for j in positions:
                    pos_freq_1[j, :] += smoothing_matrix_1[X[i][j], :]

            pos_freq_2 = np.zeros([len(features), 400])
            for i in range(len(X)):
                for index, (j, k) in enumerate(features):
                    pos_freq_2[index, :] += smoothing_matrix_2[X[i][j] * 20 + X[i][k], :]

            return pos_freq_1, pos_freq_2

        self.pos_freq_1, self.pos_freq_2 = fit_core(X, self.positions, self.both_features,
                                                    self.smoothing_matrix_1, self.smoothing_matrix_2)

        # normalization (with Laplacian smoothing)
        self.log_pos_prob_1 = np.log(self.pos_freq_1 + 0.1) - np.log(self.pos_freq_1.sum(axis=1, keepdims=True) + 2)
        self.log_pos_prob_2 = np.log(self.pos_freq_2 + 0.1) - np.log(self.pos_freq_2.sum(axis=1, keepdims=True) + 40)

        # self.log_pos_prob_1 = np.log(self.pos_freq_1) - np.log(self.pos_freq_1.sum(axis=1, keepdims=True))
        # self.log_pos_prob_2 = np.log(self.pos_freq_2) - np.log(self.pos_freq_2.sum(axis=1, keepdims=True))
        return self

    def predict_log_odds(self, X: Iterable[str]):
        """Predict log odds for a list of peptides.
        This is the recommended measurement to rank peptides because it minimizes numerical issues.

        :param X: Input peptides
        :return: log odds
        """
        n_orig_test = len(X)
        ok_index, X = self.seq2matrix(X, return_ind=True)

        test_log_pos_prob = np.zeros(X.shape[0])
        test_log_neg_prob = np.zeros(X.shape[0])

        for i in self.positions:
            test_log_pos_prob += self.log_pos_prob_1[i, X[:, i]]
            test_log_neg_prob += self.log_neg_prob_1[X[:, i]]

        for index, (j, k) in enumerate(self.both_features):
            temp = X[:, j] * 20 + X[:, k]
            test_log_pos_prob += self.log_pos_prob_2[index, temp]
            test_log_neg_prob += self.log_neg_prob_2[temp]

        # return test_log_pos_prob, test_log_neg_prob

        res = test_log_pos_prob - test_log_neg_prob

        full_res = np.zeros(n_orig_test) - np.inf
        full_res[ok_index] = res

        return res


    def predict_log_proba(self, X: Iterable[str], *, log_prior: float = np.log(999)):
        """Predict log probability for a list of peptides.
        This is not the recommended measurement because numerical issues may make it hard to rank peptides.
        Peptides that are ranked high may have indistinguishable log probabilities. Please use log odds for ranking.

        :param X: input peptides.
        :param log_prior: log(Neg:Pos) as the prior for converting odds to probability.
        :return: log probabilities
        """
        n_orig_test = len(X)
        ok_index, X = self.seq2matrix(X, return_ind=True)

        test_log_pos_prob = np.zeros(X.shape[0])
        test_log_neg_prob = np.zeros(X.shape[0])

        for i in self.positions:
            test_log_pos_prob += self.log_pos_prob_1[i, X[:, i]]
            test_log_neg_prob += self.log_neg_prob_1[X[:, i]]

        for index, (j, k) in enumerate(self.both_features):
            temp = X[:, j] * 20 + X[:, k]
            test_log_pos_prob += self.log_pos_prob_2[index, temp]
            test_log_neg_prob += self.log_neg_prob_2[temp]

        # return test_log_pos_prob, test_log_neg_prob

        res = test_log_pos_prob - test_log_neg_prob

        res = -np.logaddexp(1, - res + log_prior)
        full_res = np.zeros(n_orig_test) - np.inf
        full_res[ok_index] = res

        return full_res

    def predict_proba(self, X: Iterable[str], *, prior: float = 999.) -> pd.DataFrame:
        """Predict (linear scale) probability for a list of peptides.
        Use of this measurement in ranking peptides is discouraged because many will have identical probabilities.
        Please use log odds for ranking.

        :param X: input peptides.
        :param prior: Neg:Pos ratio as the prior for converting odds to probability.
        :return: probabilities
        """
        warnings.warn("For ranking, please use predict_log_odds. "
                      "Use of this measurement in ranking peptides is discouraged for numerical issues.")
        return np.exp(self.predict_log_proba(X, log_prior=np.log(prior)))

    def predict_details(self, X: Iterable[str], *, log_prior=np.log(999)):
        """Show prediction details for a list of peptides.

        :param X: input peptides.
        :param log_prior: log(Neg/Pos) as the prior for converting odds to probability.
        :return: a data frame containing prediction details
        """
        all_X = X
        n_orig_test = len(X)
        ok_index, X = self.seq2matrix(X, return_ind=True)

        res = np.zeros([X.shape[0], 3 + len(self.positions) + len(self.both_features)]) - np.inf
        res[:, 0] = 0.
        i_col = 3
        columns = ['proba', 'log_proba', 'log_odds']

        for i in self.positions:
            columns.append(py2naming(i))
            res[:, i_col] = self.log_pos_prob_1[i, X[:, i]] - self.log_neg_prob_1[X[:, i]]
            i_col += 1

        for index, (j, k) in enumerate(self.both_features):
            columns.append(py2naming((j, k)))
            temp = X[:, j] * 20 + X[:, k]
            res[:, i_col] = self.log_pos_prob_2[index, temp] - self.log_neg_prob_2[temp]
            i_col += 1

        # weights = []
        # for i in self.positions:
        #    weights.append(3 - self.entropy1[i])

        # for index, (j, k) in enumerate(self.both_features):
        #    weights.append(self.mutual_info[(j, k)] * 10)

        # weights = np.array(weights)

        # print(weights)

        # return test_log_pos_prob, test_log_neg_prob
        res[:, 2] = (res[:, 3:]).sum(axis=1)
        res[:, 1] = -np.logaddexp(0, -res[:, 2] + log_prior)
        res[:, 0] = np.exp(res[:, 1])

        full_res = np.zeros((n_orig_test, res.shape[1]))
        full_res[1:, :] -= np.inf
        full_res[ok_index, :] = res

        return pd.DataFrame(full_res, index=all_X, columns=columns)

    def fit_details_1(self, what='freq'):
        """Show frequency of AAs (aka motifs, as input for a logo plot)

        :param what: "freq" for frequency, or "log_odds" for log odds
        :return: the request details
        """
        aas = list(aa2int_mapping.keys())
        if what == 'freq':
            details = pd.DataFrame(self.pos_freq_1 / self.pos_freq_1.sum(axis=1, keepdims=True), columns=aas,
                                   index=[py2naming(i) for i in positions])
        elif what == 'log_odds':
            details = pd.DataFrame(self.log_pos_prob_1 - np.log(aa_prior), columns=aas,
                                   index=[py2naming(i) for i in positions])

        return details.T

    def fit_details_2(self, what='freq', topk=None):
        """Show frequency of AAs (aka motifs, as input for a logo plot)

        :param what: "freq" for frequency, "log_odds" for log odds,
            or "surplus" for P(ab) - P(a)P(b).
        :param topk: If unspecified, return the matrix directly.
            If specified, sort the values, and return the AA combinations and the values in two data frames.
            Only the topk values will be returned. Thus, specify 400 if all values are wanted.
        :return: the request details
        """
        pos_freq_1 = self.pos_freq_1 / self.pos_freq_1.sum(axis=1, keepdims=True)
        pos_freq_2 = self.pos_freq_2 / self.pos_freq_2.sum(axis=1, keepdims=True)
        if what == 'freq':
            aas = [i + j for i in aa2int_mapping.keys() for j in aa2int_mapping.keys()]
            temp = pd.DataFrame(pos_freq_2, columns=aas, index=[py2naming(i) for i in self.both_features]).T

        elif what == 'log_odds':
            aas = [i + j for i in aa2int_mapping.keys() for j in aa2int_mapping.keys()]
            temp = pd.DataFrame(self.log_pos_prob_2.copy(),
                                columns=aas,
                                index=[py2naming(i) for i in self.both_features]).T
            for i, j in zip(self.both_features, temp.columns):
                temp[j] -= np.log(np.outer(aa_prior, aa_prior).flatten())

        elif what == 'surplus':
            aas = [i + j for i in aa2int_mapping.keys() for j in aa2int_mapping.keys()]
            temp = pd.DataFrame(pos_freq_2,
                                columns=aas,
                                index=[py2naming(i) for i in self.both_features]).T
            for i, j in zip(self.both_features, temp.columns):
                temp[j] -= np.outer(pos_freq_1[i[0],], pos_freq_1[i[1],]).flatten()

        if topk is None:
            return temp
        else:
            details_key = pd.DataFrame(index=range(topk), columns=temp.columns)
            details_val = pd.DataFrame(index=range(topk), columns=temp.columns)

            for i in temp.columns:
                temp_i = temp[i].sort_values(ascending=False)
                details_key[i] = temp_i.index[:topk]
                details_val[i] = temp_i.iloc[:topk].tolist()
            return details_key, details_val
