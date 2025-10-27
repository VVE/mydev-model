import math
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


def sanitize_result_for_saving(results: dict) -> dict:
    """
    Преобразует значения в results к простым Python-типам,
    пригодным для записи в TSV/CSV и для добавления в список accepted_results.
    - np.float64 -> float
    - np.ndarray -> list
    - pd.Series -> list
    - np.nan -> None
    """
    out = {}
    for k, v in results.items():
        # None
        if v is None:
            out[k] = None
            continue

        # numpy scalar
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
            continue

        # floats / ints
        if isinstance(v, float) or isinstance(v, int):
            # guard against NaN/Inf
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                out[k] = None
            else:
                out[k] = v
            continue

        # numpy arrays, lists, pandas Series -> lists (convert elements)
        if isinstance(v, (np.ndarray, list, tuple)):
            try:
                lst = []
                for el in list(v):
                    if isinstance(el, (np.floating, np.integer)):
                        lst.append(float(el))
                    elif el is None or (isinstance(el, float) and (math.isnan(el) or math.isinf(el))):
                        lst.append(None)
                    else:
                        lst.append(el)
                out[k] = lst
            except Exception:
                out[k] = list(v) if not isinstance(v, str) else v
            continue

        # pandas types (Series)
        try:
            import pandas as _pd
            if isinstance(v, _pd.Series):
                out[k] = sanitize_result_for_saving(v.to_dict())
                continue
        except Exception:
            pass

        # default: try to cast numpy types, else keep as-is
        try:
            if hasattr(v, "tolist"):
                out[k] = v.tolist()
            else:
                out[k] = v
        except Exception:
            out[k] = v

    return out

THRESHOLDS: Dict[str, Any] = { # Пороговые значения (настраиваемые)
    "alpha": 0.05,              # уровень значимости для F и t
    "t_crit": 2.0,              # критическое значение t (можно заменить на t_{df,alpha/2})
    "coef_rel_sigma": 0.5,      # допустимый относительный robust std (MAD/|median|)
    "adj_r2_min": 0.0,          # минимально допустимый Adjusted R² (CV mean)
    "rmse_max": float('inf'),
    "mae_max": float('inf'),
    "cond_num_max": 50,
    "vif_max": 10,
    "resid_outliers_pct_max": 5.0
}

def make_model(
        model_name : str, # имя функции
        params # параметры функции
        ):
    """Модельная фабрика."""

    if model_name == "OLS":
        return LinearRegression(**params)
    elif model_name == "Ridge":
        return Ridge(**params)
    elif model_name == "Lasso":
        return Lasso(**params)
    else:
        raise ValueError(f"Модель {model_name} не поддерживается.")

def adjusted_r2(
    y_true, 
    y_pred, 
    n_features : int # число признаков
    ):
    """Adjusted R²."""

    n = len(y_true)
    if n - n_features - 1 <= 0:
        return float("-inf")
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

def corr_matrix(X, method="pearson"):
    """
    Возвращает корреляционную матрицу признаков.
    X – pandas DataFrame или ndarray.
    method – 'pearson', 'spearman', 'kendall'.
    """
    if isinstance(X, pd.DataFrame):
        return X.corr(method=method)
    else:
        # Если X – numpy array, конвертируем во временный DataFrame
        return pd.DataFrame(X).corr(method=method)

def compute_vif(
    X: pd.DataFrame
    ) -> pd.Series:
    """
    Расчёт VIF.
    - Ожидает DataFrame признаков БЕЗ константы.
    - Добавляет константу внутри (как требует формула VIF).
    - Возвращает pd.Series с индексами X.columns.
    - При ошибке — возвращает NaN для соответствующего признака.
    """
    if X.shape[1] == 0:
        return pd.Series([], dtype=float)

    try:
        X_const = add_constant(X, has_constant='add')
        vif_values = []
        for i in range(1, X_const.shape[1]):  # пропускаем константу (индекс 0)
            try:
                val = variance_inflation_factor(X_const.values, i)
            except Exception:
                val = np.nan
            vif_values.append(val)

        return pd.Series(vif_values, index=X.columns)

    except Exception as e:
        print(f"⚠ Ошибка compute_vif: {e}")
        return pd.Series([np.nan] * X.shape[1], index=X.columns)

"""Старая версия
def compute_vif(
    X: pd.DataFrame # матрица признаков
    ) -> pd.Series:
    Рассчитать VIF признаков.

    X_np = X.values
    vif_values = []
    for i in range(X_np.shape[1]):
        try:
            vif = variance_inflation_factor(X_np, i)
        except Exception:
            vif = np.nan
        vif_values.append(vif)
    return pd.Series(vif_values, index=X.columns)
"""

def cross_validate_model(
    X : pd.DataFrame, 
    y : pd.Series, 
    model, 
    k_folds : int=5
    ) -> Tuple[float, float, float]:
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    adj_r2_scores, mae_scores, rmse_scores = [], [], []
    coefs = []
    for train_idx, val_idx in kf.split(X):
        Xtr, Xval = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[train_idx], y.iloc[val_idx]
        mdl = deepcopy(model)
        mdl.fit(Xtr, ytr)
        yval_pred = mdl.predict(Xval)
        adj_r2_scores.append(adjusted_r2(yval.values, yval_pred, X.shape[1]))
        mae_scores.append(np.mean(np.abs(yval.values - yval_pred)))
        rmse_scores.append(np.sqrt(np.mean((yval.values - yval_pred) ** 2)))
        if hasattr(mdl, "coef_"):
            coefs.append(np.array(mdl.coef_, dtype=float))
    return {
        "adj_r2_mean": float(np.mean(adj_r2_scores)) if adj_r2_scores else np.nan,
        "adj_r2_std": float(np.std(adj_r2_scores)) if adj_r2_scores else np.nan,
        "mae_mean": float(np.mean(mae_scores)) if mae_scores else np.nan,
        "rmse_mean": float(np.mean(rmse_scores)) if rmse_scores else np.nan,
        "coefs": np.array(coefs) if coefs else np.array([]),
        "n_folds": k_folds
    }

def compute_coefficient_stats(
    model, 
    X: pd.DataFrame, 
    y: pd.Series, 
    robust : str ="HC3"
    ) -> Tuple[list, list]:
    """
    Вычисляет t-статистику и p-value коэффициентов.
    Поддержка только для моделей с атрибутом coef_.
    """
    try:
        X_sm = add_constant(X, has_constant='add')
        ols_res = OLS(y.values, X_sm.values).fit(cov_type=robust)
        params = ols_res.params           # includes const
        bse = ols_res.bse
        tvals = ols_res.tvalues
        pvals = ols_res.pvalues
        fstat = float(ols_res.fvalue) if hasattr(ols_res, 'fvalue') else None
        f_pvalue = float(ols_res.f_pvalue) if hasattr(ols_res, 'f_pvalue') else None

        # remove constant (index 0)
        return {
            "coef": params[1:].tolist(),
            "se": bse[1:].tolist(),
            "t": tvals[1:].tolist(),
            "p": pvals[1:].tolist(),
            "F_stat": fstat,
            "F_pvalue": f_pvalue,
            "df_model": int(ols_res.df_model),
            "df_resid": int(ols_res.df_resid)
        }
    except Exception as e:
        print(f"⚠ compute_coefficient_stats failed: {e}")
        return {"coef": None, "se": None, "t": None, "p": None, "F_stat": None, "F_pvalue": None}

"""Старая версия
def compute_vif(
    X : pd.DataFrame
    ) -> np.ndarray:
    
    Считает VIF для признаков.
    Возвращает массив .
    Возвращает numpy array (длины n_features) с VIF по порядку колонок X.columns.
    X ожидается не включающий константу. Если ошибка — возвращаем массив nan.

    
    try:
        X_const = add_constant(X, has_constant='add')
        n = X.shape[1]
        vif = []
        # variance_inflation_factor expects ndarray; index 0 is const
        for i in range(1, X_const.shape[1]):
            vif_val = variance_inflation_factor(X_const, i)
            vif.append(float(vif_val))
        return np.array(vif)
    except Exception as e:
        print(f"⚠ compute_vif failed: {e}")
        return np.full((X.shape[1],), np.nan)
"""

def compute_coef_stability(
    coefs_array: np.ndarray, 
    eps: float = 1e-8
    ) -> dict :
    """
    Stability via CV-coeffs: robust MAD and rel_mad = mad/ (|median|+eps).
    coefs_array: shape (n_folds, n_features)
    возвращает dict с median, mad, rel_mad (по столбцам)
    """
    if coefs_array is None or coefs_array.size == 0:
        return {"median": None, "mad": None, "rel_mad": None}
    median = np.median(coefs_array, axis=0)
    # use robust mad (numpy doesn't have mad in older versions): use scipy or np.median(|x-median|)
    mad = np.median(np.abs(coefs_array - median[None, :]), axis=0)
    rel_mad = mad / (np.abs(median) + eps)
    return {"median": median.tolist(), "mad": mad.tolist(), "rel_mad": rel_mad.tolist()}

def analyze_residuals(
    residuals
    ) -> dict :
    """
    analyze_residuals.
    σ остатков и % выбросов |res| > 3σ.
    """

    sigma = float(np.std(residuals))
    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = float(q3 - q1)
    iqr_over_sigma = float(iqr / sigma) if sigma > 0 else float('inf')
    outliers_pct = float(np.mean(np.abs(residuals) > 3 * sigma) * 100.0)
    return {"sigma": sigma, "IQR": iqr, "IQR_over_sigma": iqr_over_sigma, "outliers_pct": outliers_pct}