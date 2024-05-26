# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """
    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = pd.DataFrame.to_dict(
        pd.DataFrame({feat: [pd.to_numeric(CTG_features[feat], errors='coerce')] for feat in CTG_features if
                      feat != extra_feature}).dropna())
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas dataframe of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe c_cdf containing the "clean" features
    """

    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_cdf = pd.DataFrame({x: pd.to_numeric(CTG_features[x], errors='coerce') for x in CTG_features if
                          x != extra_feature})
    c_cdf_nonan = pd.DataFrame.copy(c_cdf).dropna()
    for feat in c_cdf:
        while any(c_cdf.loc[:, feat].isna()):
            c_cdf.loc[:, feat] = c_cdf.loc[:, feat].fillna(np.random.choice(c_cdf_nonan.loc[:, feat]), limit=1)
    # -------------------------------------------------------------------------
    return c_cdf


def get_stat(feature):
    """

    :param feature:  a feature of the data.
    :return: Dictionary containing summary statistics of a feature: Min, Q1, Median, Q3, Max.
    """
    statdict = {'Min': np.min(feature), 'Q1': np.percentile(feature, 25), 'Median': np.percentile(feature, 50),
                'Q3': np.percentile(feature, 75),
                'Max': np.max(feature)}
    return statdict


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_samp
    :return: Summary statistics as a dictionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = ({feat: get_stat(c_feat[feat]) for feat in c_feat})
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_samp
    :param d_summary: Output of sum_stat
    :return: Dataframe containing c_feat with outliers removed
    """
    c_no_outlier = c_feat.copy()
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for feat in c_no_outlier:
        q3 = d_summary[feat]['Q3']
        q1 = d_summary[feat]['Q1']
        iqr = q3 - q1
        for idx, x in enumerate(c_no_outlier[feat]):
            if x > q3 + (1.5 * iqr) or x < q1 - 1.5 * iqr:
                c_no_outlier[feat][idx] = np.nan
    # -------------------------------------------------------------------------
    return c_no_outlier


def phys_prior(c_samp, feature, thresh):
    """

    :param c_samp: Output of nan2num_samp
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = [x for x in c_samp[feature] if x <= thresh]
    # -------------------------------------------------------------------------
    return np.array(filt_feature)


class NSD:

    def __init__(self):
        self.max = np.nan
        self.min = np.nan
        self.mean = np.nan
        self.std = np.nan
        self.fit_called = False

    def fit(self, CTG_features):
        # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
        self.min = {feat: np.min(CTG_features[feat]) for feat in CTG_features}
        self.max = {feat: np.max(CTG_features[feat]) for feat in CTG_features}
        self.mean = {feat: np.mean(CTG_features[feat]) for feat in CTG_features}
        self.std = {feat: np.std(CTG_features[feat]) for feat in CTG_features}
        # -------------------------------------------------------------------------
        self.fit_called = True

    def transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        """
        Note: x_lbl should only be either: 'Original values [N.U]', 'Standardized values [N.U.]', 'Normalized values [N.U.]' or 'Mean normalized values [N.U.]'
        :param mode: A string determining the mode according to the notebook
        :param selected_feat: Two elements tuple of strings of the features for comparison
        :param flag: A boolean determining whether to plot a histogram
        :return: Dataframe of the normalized/standardized features called nsd_res
        """
        ctg_features = CTG_features.copy()
        if self.fit_called:
            if mode == 'none':
                nsd_res = ctg_features
                x_lbl = 'Original values [N.U]'
            # ------------------ IMPLEMENT YOUR CODE HERE (for the remaining 3 methods using elif):---------------------
            elif mode == 'MinMax':
                for feat in CTG_features:
                    min = self.min[feat]
                    max = self.max[feat]
                    ctg_features.loc[:, feat] = (ctg_features.loc[:, feat] - min) / (max - min)
                    nsd_res = ctg_features
                    x_lbl = 'Normalized values [N.U.]'
            elif mode == 'standard':
                for feat in CTG_features:
                    mean = self.mean[feat]
                    std = self.std[feat]
                    ctg_features.loc[:, feat] = (ctg_features.loc[:, feat] - mean) / std
                    nsd_res = ctg_features
                    x_lbl = 'Standardized values [N.U.]'
            elif mode == 'mean':
                for feat in CTG_features:
                    mean = self.mean[feat]
                    min = self.min[feat]
                    max = self.max[feat]
                    ctg_features.loc[:, feat] = (ctg_features.loc[:, feat] - mean) / (max - min)
                    nsd_res = ctg_features
                    x_lbl = 'Mean normalized values [N.U.]'
            # -------------------------------------------------------------------------
            if flag:
                self.plot_hist(nsd_res, mode, selected_feat, x_lbl)
            return nsd_res
        else:
            raise Exception('Object must be fitted first!')

    def fit_transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        self.fit(CTG_features)
        return self.transform(CTG_features, mode=mode, selected_feat=selected_feat, flag=flag)

    def plot_hist(self, nsd_res, mode, selected_feat, x_lbl):
        x, y = selected_feat
        if mode == 'none':
            bins = 50
        else:
            bins = 80
        # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
        plt.figure(figsize=(10, 5))
        plt.hist(nsd_res.loc[:, x], alpha=0.5, bins=bins, label=x)
        plt.hist(nsd_res.loc[:, y], alpha=0.5, bins=bins, label=y)
        plt.title(f'Mode: {mode}, Features: {x},{y}')
        plt.xlabel(x_lbl)
        plt.ylabel('Count')
        plt.legend(loc='upper right')
        plt.show()
        # -------------------------------------------------------------------------


# Debugging block!
if __name__ == '__main__':
    from pathlib import Path

    file = Path.cwd().joinpath(
        'messed_CTG.xls')  # concatenates messed_CTG.xls to the current folder that should be the extracted zip folder
    CTG_dataset = pd.read_excel(file, sheet_name='Raw Data')
    CTG_features = CTG_dataset[['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DR', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                                'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance',
                                'Tendency']]
    CTG_morph = CTG_dataset[['CLASS']]
    fetal_state = CTG_dataset[['NSP']]

    extra_feature = 'DR'
    c_ctg = rm_ext_and_nan(CTG_features, extra_feature)
