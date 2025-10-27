from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

np.random.seed(42) # для воспроизводимости бутстрапов и других стохастических процедур
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import shap
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import t
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import SparsePCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
from sklearn.datasets import make_regression


def load_synthetic_data(
    n_samples=2000 # число образцов
    ) -> Tuple[
        pd.DataFrame, # матрица признаков;
        pd.Series # целевой вектор.
        ]:
    """ Генерация искусственного датасета с коррелированными признаками. """

    X_np, y_np, coef = make_regression(
        n_samples=n_samples, # число наблюдений
        n_features=5, # число признаков
        n_informative=3, # число информативных признаков (признаков, используемых для построения линейной модели, генерирующей выходные данные
        effective_rank=2, # ранк матрицы признаков (чем меньше, тем сильнее корреляция между признаками)
        shuffle=False, # не перемешиваем признаки
        bias=10.0, # смещение линейной модели
        tail_strength=0.5, # степень "хвостовости" сингулярного значения распределения
        noise=0.5, # стандартное отклонение гауссовского шума, применяемого к выходному сигналу
        coef=True, # возвращать коэффициенты линейной модели
        random_state=42 # для воспроизводимости
        ) # -> X_np: ndarray (n_samples, n_features), y_np: ndarray (n_samples,), coef: ndarray (n_features,)
    print()
    print("Матрица признаков:\n", X_np, "\nЦелевой вектор:\n", y_np, "\nКоэффициенты:\n", coef)
    # Генерация полиномиальных признаков
    X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])]) # -> DataFrame с признаками x0,.. x4
    # добавим квадраты и произведения производных признаков
    #X["x0_x1"] = X["x0"] * X["x1"]
    #X["x0²"] = X["x0"] ** 2
    #X["x1²"] = X["x1"] ** 2
    #X["x2_x3"] = X["x2"] * X["x3"]
    #X["x4²"] = X["x4"] ** 2
    y = pd.Series(y_np, name="y")
    #print("Признаки после расширения:\n", X, "\nЦелевой вектор:\n", y)
    #y_ = coef[0] * X["x0"] + coef[1] * X["x1"] + coef[2] * X["x2"] + 10.0
    #print("y-y_:\n", y-y_)

    return X, y

def load_synthetic_data_PCA():
    X_np, y = make_regression(n_samples=200, n_features=6, n_informative=4, noise=0.1, random_state=42)
    X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    return X, y
