from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import f
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from utils import (THRESHOLDS, adjusted_r2, analyze_residuals,
                   compute_coef_stability, compute_coefficient_stats,
                   compute_vif, cross_validate_model)


def make_model(model_name: str, params: dict):
    """Возвращает sklearn-модель по имени и параметрам."""
    if model_name == "OLS":
        return LinearRegression(**params)
    elif model_name == "Ridge":
        return Ridge(**params)
    elif model_name == "Lasso":
        return Lasso(**params)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")


def adjusted_r2(y_true, y_pred, n_features):
    """Вычисляет Adjusted R² вручную."""
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


def cross_validate_model(X, y, model, k_folds):
    """Кросс-валидация по train-данным."""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    adj_r2_scores = []
    mae_scores = []
    rmse_scores = []
    coefs = []

    for train_idx, val_idx in kf.split(X):
        Xtr, Xval = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[train_idx], y.iloc[val_idx]

        # Клонируем модель, чтобы она не "запоминала" прошлое обучение
        mdl = deepcopy(model)

        #model = make_model(model_name, params)
        mdl.fit(Xtr, ytr)

        yval_pred = mdl.predict(Xval)

        adj_r2_scores.append(adjusted_r2(yval, yval_pred, X.shape[1]))
        mae_scores.append(mean_absolute_error(yval, yval_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(yval, yval_pred)))

        # Сохраняем коэффициенты, если доступны
        if hasattr(mdl, "coef_"):
            coefs.append(mdl.coef_)

    return {
        "adj_r2_mean": np.mean(adj_r2_scores),
        "adj_r2_std": np.std(adj_r2_scores),
        "mae_mean": np.mean(mae_scores),
        "rmse_mean": np.mean(rmse_scores),
        "coefs": np.array(coefs)
    }

def run_one_model_on_subset(
    X_train : pd.DataFrame, # обучающая матрица признаков
    y_train : pd.Series, # обучающий целевой вектор
    X_test : pd.DataFrame, # тестовая матрица признаков
    y_test : pd.Series, # тестовый целевой вектор
    subset_name : str, # имя набора признаков
    features, # набор признаков
    model_name : str, # имя модели
    params : dict, # параметры модели
    model
    ) -> dict:
    """
    Выполняет полную проверку для одной расширенной модели (только OLS сейчас).
    Возвращает плоский словарь с результатами (или None, если модель отвергнута).
    Печатает в терминал причины отклонения.
    """

    result = {
        "model_name": model_name,
        "params": params,
        "features": features
    }

    # 1) Кросс-валидация (на обучающей выборке)
    try:
        cv = cross_validate_model(X_train, y_train, model, k_folds=5)
    except Exception as e:
        print(f"[{subset_name}] ERROR during CV for model {model_name}, params={params}: {e}")
        return None

    result.update({
        "cv_adj_r2_mean": cv["adj_r2_mean"],
        "cv_adj_r2_std": cv["adj_r2_std"],
        "cv_rmse_mean": cv["rmse_mean"],
        "cv_mae_mean": cv["mae_mean"]
    })

    # 2) Обучение (с помощью библиотеки Scikit-Learn, на обучающей выборке)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"[{subset_name}] ERROR fitting model {model_name}, params={params}: {e}")
        return None

    # 3) Предсказание (на обеих выборках)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    result.update({
        "train_adj_r2": adjusted_r2(y_train.values, y_pred_train, len(features)),
        "test_adj_r2": adjusted_r2(y_test.values, y_pred_test, len(features)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
    })

    # 4) Поиск коэффициентов модели (с помощью библиотеки statsmodels.regression.linear_model.OLS, робастным методом (c параметром HC3))
    coef_stats = compute_coefficient_stats(model, X_train, y_train, robust="HC3")
    df_model = coef_stats.get("df_model")     # числитель
    df_resid = coef_stats.get("df_resid")     # знаменатель
    # Проверяем, что df корректны:
    if df_model is None or df_resid is None or df_model <= 0 or df_resid <= 0:
        print(f"набор признаков {subset_name}] отброшен из-за ошибки при расчете критического значения F-статистики")
        return None

    # 4) F-test (значимость модели)   
    if coef_stats["F_pvalue"] is None:
        print(f"набор признаков {subset_name} отброшен из-за ошибки при выполнении F-теста для модели {model_name} и параметров {params}.")
        return None
    
    alpha = THRESHOLDS["alpha"] # уровень значимости
    F_crit = f.ppf(1 - alpha, df_model, df_resid) # критическое значение F

    # Проверка F-теста: p-value и F_stat < F_crit
    if (coef_stats["F_pvalue"] >= alpha) or (coef_stats["F_stat"] < F_crit):
        print(f"набор признаков {subset_name}] отброшен по фильтру F-тест: F_stat={coef_stats['F_stat']:.4g} < F_crit={F_crit:.4g} "
          f"or F_pvalue={coef_stats['F_pvalue']:.4g} >= alpha={alpha}")
        return None

    result.update({ # набор признаков статистически значим -> прошел F-тест
        "F_stat": coef_stats["F_stat"],
        "F_pvalue": coef_stats["F_pvalue"],
        "F_crit": float(F_crit)
    })

    # 5) t-тест (в жестком варианте: принимать набор признаков, если все коэффициенты - значимы (|t| >= t_crit and p < alpha)
    t_vals = np.array(coef_stats["t"], dtype=float) if coef_stats["t"] is not None else None
    p_vals = np.array(coef_stats["p"], dtype=float) if coef_stats["p"] is not None else None

    if t_vals is None or p_vals is None:
        print(f"набор признаков {subset_name} отброшен из-за невозможности вычислить значения t-статистик или p-значения")
        return None

    tcrit = THRESHOLDS["t_crit"]
    alpha = THRESHOLDS["alpha"]

    t_mask = np.abs(t_vals) >= tcrit
    p_mask = p_vals < alpha
    all_pass = np.all(t_mask & p_mask)

    if not all_pass:
        failed_idx = np.where(~(t_mask & p_mask))[0].tolist()
        print(f"набор признаков {subset_name} отброшен по t-тесту. Индексы коэффициентов: {failed_idx}. t_vals={t_vals.tolist()}, p_vals={p_vals.tolist()}")
        return None

    result.update({ # набор признаков прошел t-тест
        "coef_t": t_vals.tolist(),
        "coef_p": p_vals.tolist()
    })

    # 6) Фильтр на стабильность коэффициентов, найденных при кросс-валидации (`cv['coefs']`)
    # Бутстрэп: повторно обучаем модель на бутстрэп-выборках, собираем коэффициенты; 
    # возвращаем median, mad, rel_mad = mad/ (|median| + eps), sign_consistency (доля бутстрэпов, где знак β совпадает с медианой). 
    # Это дает более робастную метрику, чем простая std/mean.
    coefs_arr = cv.get("coefs")
    stability = compute_coef_stability(coefs_arr)
    rel_mad = np.array(stability["rel_mad"]) if stability["rel_mad"] is not None else None
    if rel_mad is None:
        print(f"набор признаков {subset_name} отброшен из-за невозможности выполнения теста на стабильность коэффициентов")
        return None

    # If any relative MAD > threshold => reject
    rel_thr = THRESHOLDS["coef_rel_sigma"]
    if np.any(np.array(rel_mad) > rel_thr):
        idx_fail = np.where(np.array(rel_mad) > rel_thr)[0].tolist()
        print(f"набор признаков {subset_name}] отброшен по тесту на стабильность коэффициентов: rel_mad с индексами {idx_fail} > {rel_thr}. rel_mad={rel_mad.tolist()}")
        return None

    result.update({ # набор признаков прошел тест на стабильность коэффициентов
        "coef_median": stability["median"],
        "coef_mad": stability["mad"],
        "coef_rel_mad": stability["rel_mad"]
    })

    # 7) Фильтр на мультиколлинеарность (метрики VIF и cond) — compute and check (though you said these are applied earlier during feature generation,
    # we still compute and report them here)
    # исключен, так как проводится уже при создании наборов признаков
    """
    vif_vals = compute_vif(X_train)
    cond_num = float(np.linalg.cond(X_train.values))
    if vif_vals is None:
        max_vif = None
    else:
        max_vif = float(np.nanmax(vif_vals))

    result.update({
        "max_vif": max_vif,
        "cond_num": cond_num
    })
    """

    # 8) Фильтр по выбросам остатков (на обучающей выборке)
    residuals = y_train.values - y_pred_train
    resid_info = analyze_residuals(residuals)

    if resid_info["outliers_pct"] > THRESHOLDS["resid_outliers_pct_max"]:
        print(f"набор признаков {subset_name} отброшен (residuals outliers_pct {resid_info['outliers_pct']:.2f}% > {THRESHOLDS['resid_outliers_pct_max']}%)")
        return None
    if resid_info["IQR_over_sigma"] >  THRESHOLDS.get("IQR_over_sigma_threshold", 1.5):
        print(f"набор признаков {subset_name} отброшен (residual IQR/sigma {resid_info['IQR_over_sigma']:.3f} > threshold)")
        return None

    result.update({ # набор признаков прошел тест на выбросы остатков
        "res_sigma": resid_info["sigma"],
        "res_IQR": resid_info["IQR"],
        "res_IQR_over_sigma": resid_info["IQR_over_sigma"],
        "res_outliers_pct": resid_info["outliers_pct"]
    })

    # 9) Фильтр по метрикам качества предсказания (Adj R2, RMSE, MAE)
    if result["cv_adj_r2_mean"] < THRESHOLDS["adj_r2_min"]:
        print(f"[{subset_name}] REJECT (cv_adj_r2_mean {result['cv_adj_r2_mean']:.4f} < adj_r2_min {THRESHOLDS['adj_r2_min']})")
        return None
    if result["cv_rmse_mean"] > THRESHOLDS["rmse_max"]:
        print(f"[{subset_name}] REJECT (cv_rmse_mean {result['cv_rmse_mean']:.4f} > rmse_max {THRESHOLDS['rmse_max']})")
        return None
    if result["cv_mae_mean"] > THRESHOLDS["mae_max"]:
        print(f"[{subset_name}] REJECT (cv_mae_mean {result['cv_mae_mean']:.4f} > mae_max {THRESHOLDS['mae_max']})")
        return None

    # 10) Если дошли сюда — модель принята. Вернуть полный плоский словарь.
    print(f"набор признаков {subset_name} для модели {model_name} с параметрами {params} прошел фильтры, cv_adj_r2_mean={result['cv_adj_r2_mean']:.4f}")
    return result