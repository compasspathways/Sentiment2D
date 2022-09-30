"""
Utilities useful for running the analyses in the accompaning notebook.
"""


import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TREAT_STR = {0: "1mg", 1: "10mg", 2: "25mg"}


class MyScaler(BaseEstimator, TransformerMixin):
    """
    Thin wrapper around StandardScaler that lets you specify a subset of columns to scale.
    """

    def __init__(self, scale_columns):
        self.scale_columns = scale_columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scale_columns = [c for c in self.scale_columns if c in X.columns]
        if self.scale_columns:
            self.scaler.fit(X[self.scale_columns])
        return self

    def transform(self, X, y=None):
        if not self.scale_columns:
            return X
        tmpdf = X.copy()
        tmpdf[self.scale_columns] = self.scaler.transform(tmpdf[self.scale_columns])
        return tmpdf


class MyPCA(BaseEstimator, TransformerMixin):
    """
    Does PCA on a subset of the columns specified by pca_columns in the init method. The first ncomponents of
    the PCA are added to X (named [pca_prefix][component_num]) and the pca_columns are removed from X in
    transform. This facilitates adding PCA to a CV pipeline when you don't want to do PCA across all columns.
    You could even chain several PCA stages to do separate PCAs on different subsets of columns.
    """

    def __init__(self, pca_columns, pca_prefix="pca_ebi", ncomponents=1):
        self.pca_columns = pca_columns
        self.pca_prefix = pca_prefix
        self.ncomponents = ncomponents

    def fit(self, X, y=None):
        self.pca_columns = [c for c in self.pca_columns if c in X.columns]
        if self.pca_columns:
            self.pca = PCA(n_components=self.ncomponents)
            self.pca.fit(X[self.pca_columns])
        return self

    def transform(self, X, y=None):
        if not self.pca_columns:
            return X
        tmpdf = X.copy()
        pca_vals = self.pca.transform(tmpdf[self.pca_columns])
        for idx in range(self.ncomponents):
            tmpdf[f"pca_ebi{idx}"] = pca_vals[:, idx]
        tmpdf.drop(self.pca_columns, axis=1, inplace=True)
        return tmpdf


class HidePrints:
    """
    Just what the name says-- hide prints from function calls that are overly verbose.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr.close()
        sys.stderr = self._original_stderr


def fit_logit_statsmodels(xdf, model):
    """
    :param xdf: Dataframe containing exogenous and endogenous variables.
    :param model: String specifying the model (patsy style).
    """
    md = smf.logit(formula=model, data=xdf)
    model_fit = md.fit()
    acc_sum, confmat, predicted = get_accuracy_logit(xdf, model, method="l1", thresh=0.5)
    return model_fit, acc_sum, confmat, predicted


def fit_logit_sklearn(xdf, model, scale_cols=None, ebi_items=None):
    """
    :param xdf: Dataframe containing exogenous and endogenous variables.
    :param model: String specifying the model (patsy style).
    :param scale_cols: Columns that should be scaled using the MyScaler.
    :param ebi_items: Columns containing the individual EBI items that should be run through PCA.
    """
    NORM = "l1"
    y, X = dmatrices(model, xdf, return_type="dataframe")
    endog = y.columns[0]
    logistic = LogisticRegression(penalty=NORM, solver="liblinear", max_iter=1000, C=1)
    model_fit = logistic.fit(X, y.values.ravel())
    scale_cols = scale_cols if scale_cols is not None else []

    if "pca" in model and ebi_items:
        pca = MyPCA(ebi_items)
        X = X.join(xdf[ebi_items])
    else:
        pca = "passthrough"
    if scale_cols:
        scaler = MyScaler(scale_cols)
    else:
        scaler = "passthrough"

    pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)])

    cv = KFold(n_splits=xdf.shape[0])
    y_pred = cross_val_predict(pipe, X, y.values.ravel(), cv=cv, method="predict_proba")

    y_pred_df = pd.DataFrame(y_pred, index=xdf.index, columns=["No_R", "Yes_R"])
    y_pred_df["predicted_responder"] = [int(x) for x in y_pred_df["Yes_R"] >= 0.5]
    y_pred_df[endog] = xdf[endog]
    y_pred_df["treatment"] = xdf["treatment"]

    cm_str, cm = confusion(y_pred_df[endog], y_pred_df.Yes_R)
    cmdfs = [pd.DataFrame(cm, index=pd.Index(["ALL"], name="Group"))]
    s = f"\nAll subjects\n-------------------------\n{cm_str}"
    for trt in pd.unique(xdf.treatment):
        idx = xdf.treatment == trt
        cm_str, cm = confusion(y_pred_df.loc[idx, endog], y_pred_df.loc[idx, "Yes_R"])
        s += f"\n{TREAT_STR[trt]}\n-------------------------\n{cm_str}"
        cmdfs.append(pd.DataFrame(cm, index=pd.Index([TREAT_STR[trt]], name="Group")))

    return s, model_fit, pd.concat(cmdfs), y_pred_df, endog


def confusion(true, predicted, thresh=0.5, nboot=0):
    """
    If nboot is > 0, the AUC 95%CIs are estimated with that many bootstrap iterations and the
    resulting outputs will then include the upper and lower CIs.
    """
    pred_bool = predicted > thresh
    tn, fp, fn, tp = confusion_matrix(true, pred_bool).ravel()
    n = tn + fp + fn + tp
    acc = (tp + tn) / n
    mcc = matthews_corrcoef(true, pred_bool)
    f1 = f1_score(true, pred_bool)
    auc = roc_auc_score(true, predicted)
    cm = {
        "Accuracy": acc,
        "MCC": mcc,
        "f1": f1,
        "AUC": auc,
        "N": n,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }
    auc_str = f"AUC: {auc:0.3f}\n"
    s = (
        f"accuracy: {(tp + tn)} / {n} ({acc * 100:0.1f}%)\n"
        f"MCC: {mcc:0.3f}\n"
        f"F1: {f1:0.3f}\n"
        f"{auc_str}\n"
        f"                 TRUTH\nPREDICTED   +respond  -respond\n"
        f"+respond       {tp:3}      {fp:3}\n"
        f"-respond       {fn:3}      {tn:3}\n"
    )
    return s, cm


def get_accuracy_logit(df, model, method=None, thresh=0.5):
    """
    Run leave-one-out cross-validation to estimate model accuracy.
    """
    loo = LeaveOneOut()
    loo.get_n_splits(df.values)

    tmpdf = df.copy()

    endog = model.split("~")[0].strip()

    pred = []
    for train_index, test_index in loo.split(tmpdf.values):
        print(".", end="", flush=True)
        dftrain = tmpdf.iloc[train_index, :].copy()
        dftest = tmpdf.iloc[test_index, :].copy()
        md = smf.logit(formula=model, data=dftrain)
        with HidePrints():
            if method:
                mdf = md.fit_regularized(method=method)
            else:
                mdf = md.fit()
        y_hat = mdf.predict(dftest)
        pred.append(y_hat.values[0])
    print("finished.")
    tmpdf["y_hat"] = pred
    tmpdf["predicted_responder"] = tmpdf.y_hat > thresh
    cm_str, cm = confusion(tmpdf[endog], tmpdf.y_hat, thresh)
    cmdfs = [pd.DataFrame(cm, index=pd.Index(["ALL"], name="Group"))]
    s = f"\nAll subjects\n-------------------------\n{cm_str}"
    for trt in pd.unique(tmpdf.treatment):
        idx = tmpdf.treatment == trt
        cm_str, cm = confusion(tmpdf.loc[idx, endog], tmpdf.loc[idx, "y_hat"], thresh)
        s += f"\n{TREAT_STR[trt]}\n-------------------------\n{cm_str}"
        cmdfs.append(pd.DataFrame(cm, index=pd.Index([TREAT_STR[trt]], name="Group")))

    return s, pd.concat(cmdfs), tmpdf.predicted_responder


def fdr_correct(pvalues):
    """
    Benjamini-Hochberg FDR
    """
    pvalues = np.array(pvalues)
    sample_size = pvalues.shape[0]
    qvalues = np.empty(sample_size)
    values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
    values.sort()
    values.reverse()
    new_values = []
    for i, vals in enumerate(values):
        rank = sample_size - i
        pvalue, index = vals
        new_values.append((sample_size / rank) * pvalue)
    for i in range(0, int(sample_size) - 1):
        if new_values[i] < new_values[i + 1]:
            new_values[i + 1] = new_values[i]
    for i, vals in enumerate(values):
        pvalue, index = vals
        qvalues[index] = new_values[i]
    return qvalues


def get_cols_from_model(model):
    """
    Parse a patsy model string to get all the variables specified (e.g., so you can dropna for just those)
    """
    cols = {s.strip(" I()") for s in re.split(r"-|\+|:|~|\*", model)} - {"1", ""}
    return list(cols)


def fold_prob(x, y):
    """
    Smoothing a probaility histogram (e.g., using KDE) results in values outside [0, 1].
    This folds the tails that are outside [0, 1] back inside, keeping x within [0, 1] while
    maintaining the sum total density estimate.
    """
    fold_l = y[x < 0]
    x = x[len(fold_l) :]
    y = y[len(fold_l) :]
    y[: len(fold_l)] += np.flip(fold_l)
    fold_r = y[x > 1]
    x = x[: -len(fold_r)]
    y = y[: -len(fold_r)]
    y[-len(fold_r) :] += np.flip(fold_r)
    return x, y
