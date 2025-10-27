import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

np.random.seed(42) # для воспроизводимости бутстрапов и других стохастических процедур
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
from sklearn.datasets import make_regression
from sklearn.decomposition import SparsePCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# импорт из других файлов проекта
from utils import compute_vif, corr_matrix

warnings.filterwarnings("ignore")

"""
def generate_feature_subsets(X, y):
    #Упрощенная версия.   
    # Минимально — вернём 2 подмножества
    return {
        "all_features": list(X.columns),
        "first_5": list(X.columns[:5])
    }
"""



# ---------------------------
# 1) Генерация подмножеств признаков
# ---------------------------
# ---------------------------
# Универсальные инструменты для добавления подмножеств
# ---------------------------

def format_feature_subset(features: List[str]) -> str:
    """
    Преобразует список признаков в строку вида:
    '{x0, x2, x3}'
    """
    return "{" + ", ".join(features) + "}"

"""
def add_subset(result_dict: Dict[str, List[str]],
               base_name: str,
               features: List[str],
               description: str = "") -> None:
    #
    #Унифицированное добавление подмножества в словарь.
    #- base_name — машинное имя (например, 'clust_k3_rep_maxcorr')
    #- features — список признаков ['x1','x5',...]
    #- description — человекочитаемая расшифровка
    #
    subset_name = f"'{format_feature_subset(features)}'"
    result_dict[subset_name] = features

    print(f"\nПодмножество создано: {subset_name}")
    print(f"Способ получения: {base_name}")
    if description:
        print(f"Описание: {description}")
"""

# ---------------------------
# Универсальные вспомогательные функции
# ---------------------------

def features_to_str_sorted(features: list) -> str:
    """Преобразует список признаков в строку вида '{x0, x2, x3}' с сортировкой."""
    return "{" + ", ".join(sorted(features)) + "}"

def add_subset_registry( 
    result_dict: dict, # дополняемый здесь словарь 
    X: pd.DataFrame, # исходная матрица признаков
    base_name: str, 
    features: list, # список признаков, который надо проверить на новизну 
    description: str, # описание основания формирования списка features
    cond_num_threshold: float = 1e3, # можно изменить потом в main
    max_vif_threshold: float = 10.0
    ) -> dict : # дополненный словарь result_dict
    """
    Добавляет подмножество признаков (features) в реестр result_dict:
    с проверкой на:
    - число обусловленности (cond_num);
    - максимальный VIF среди признаков.
    Если критерии превышены — не добавляет и выводит причину.

    - Не добавляет пустые множества.
    - Уникальность определяется по frozenset(features).
    - Если множество уже существует — просто добавляет новый метод (если его там нет).
    Печать только при первом добавлении множества.
    """
    
    # 1. Проверка пустого списка
    if not features or len(features) == 0:
        return

    # 2. Преобразуем в frozenset для проверки уникальности
    fset = frozenset(features)

    # 3. если уже есть — дописываем метод (no duplicate methods)
    if fset in result_dict:
        if base_name not in result_dict[fset]["methods"]:
            result_dict[fset]["methods"].append(base_name)
            print(f"Подмножество '{features_to_str_sorted(features)}' уже существует, добавлен метод: {base_name}")
        else:
            print(f"Подмножество '{features_to_str_sorted(features)}' уже существует, метод {base_name} уже зарегистрирован")
        return

    # 4. Фильтрация по cond_num и VIF
    X_sub = X[list(fset)]

    # Число обусловленности
    try:
        cond_num = np.linalg.cond(X_sub.values)
    except Exception as e:
        print(f"Отклонено подмножество {features}: ошибка вычисления cond_num ({e})")
        return

    if cond_num > cond_num_threshold:
        print(f"Отклонено подмножество {features}: cond_num={cond_num:.2f} > {cond_num_threshold}")
        return

    # VIF
    try:
        vif_vals = compute_vif(X_sub)
    except Exception as e:
        print(f"Отклонено подмножество {features}: ошибка вычисления VIF ({e})")
        return

    if np.any(np.isnan(vif_vals)):
        print(f"Отклонено подмножество {features}: VIF содержит NaN")
        return

    max_vif = float(np.max(vif_vals))
    if max_vif > max_vif_threshold:
        print(f"Отклонено подмножество {features}: max VIF={max_vif:.2f} > {max_vif_threshold}")
        return

    # 5. Всё прошло — добавляем в реестр
    result_dict[fset] = {
        "features": sorted(fset),
        "methods": [base_name]
    }

    # Печать (красивая строка множества)
    #pretty = "{" + ", ".join(sorted(fset)) + "}"
    #print(f"Новое подмножество признаков создано: '{pretty}'")
    print(f"Новое подмножество признаков создано: {sorted(fset)}")    
    print(f"Способ получения: {base_name}")
    print(f"Описание: {description}")

    return

def merge_registries(target: dict, src: dict) -> None:
    """
    Объединить src -> target in-place, где записи имеют вид:
      ключ = frozenset(features) или '_stability_freq'
      значение = {"features": [...], "methods": [...]}
    Правила:
      - для frozenset: если ключ новый — копируем запись; если уже есть — дополняем список methods без дубликатов.
      - для '_stability_freq': если в target нет — кладём; если есть — оставляем существующий (не перезаписываем).
        (Если нужно изменить поведение — можно сделать объединение/усреднение.)
    """
    if not isinstance(src, dict):
        return

    for k, v in src.items():
        if k == "_stability_freq":
            # если частоты ещё нет в target — добавим, иначе не трогаем (чтобы первичный результат stability сохранялся)
            if "_stability_freq" not in target:
                target["_stability_freq"] = v
            # если нужно — можно дописать стратегию объединения/усреднения
            continue

        # ключи ожидаются как frozenset
        if not isinstance(k, frozenset):
            # на всякий случай: если src использует строковые имена (старые версии), попытаться привести
            try:
                feats = list(v.get("features", [])) if isinstance(v, dict) else []
                k2 = frozenset(feats)
            except Exception:
                # пропускаем некорректный запись
                continue
            k = k2

        # значение — ожидаем словарь с keys "features" (list) и "methods" (list)
        if k in target:
            # объединяем методы, избегая дубликатов
            existing_methods = target[k].get("methods", [])
            new_methods = v.get("methods", []) if isinstance(v, dict) else []
            for m in new_methods:
                if m not in existing_methods:
                    existing_methods.append(m)
            target[k]["methods"] = existing_methods
            # ensure features sorted list present
            if "features" not in target[k] and isinstance(v, dict) and "features" in v:
                target[k]["features"] = sorted(v["features"])
        else:
            # добавляем новую запись — копируем features (sorted) и методы
            feats = sorted(v.get("features", list(k))) if isinstance(v, dict) else sorted(list(k))
            methods = list(v.get("methods", [])) if isinstance(v, dict) else []
            target[k] = {"features": feats, "methods": methods}

# ---------------------------
# Функции генерации подмножеств признаков
# ---------------------------

# ---------------------------
# 1) Clustering по |corr|
# ---------------------------
def generate_subsets_by_clustering(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    n_clusters_list: Iterable[int] = (2, 3, 4, 5),
    linkage_method: str = "average",
    representative: str = "max_corr_with_y"  # 'max_corr_with_y', 'min_vif', 'first'
) -> Dict[str, List[str]]:
    assert isinstance(X, pd.DataFrame)

    corr = corr_matrix(X)  # |corr|
    dist = 1 - corr
    condensed = squareform(dist.values, checks=False)

    results = {}

    print("\n=== Генерация подмножеств: агломеративная кластеризация по |corr| ===")

    for k in n_clusters_list:
        print(f"\nk: {k}")
        clusterer = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = clusterer.fit_predict(dist.values)

        clusters = {}
        for feat, lab in zip(X.columns, labels):
            clusters.setdefault(lab, []).append(feat)

        selected_features = []
        for lab, feats in clusters.items():
            if len(feats) == 1:
                selected_features.append(feats[0])
                continue

            if representative == "first":
                selected_features.append(feats[0])
            elif representative == "max_corr_with_y" and y is not None:
                corr_with_y = X[feats].corrwith(y).abs()
                selected_features.append(corr_with_y.idxmax())
            elif representative == "min_vif":
                try:
                    vif = compute_vif(X[feats])
                    selected_features.append(vif.idxmin())
                except Exception:
                    selected_features.append(feats[0])
            else:
                selected_features.append(feats[0])

        add_subset_registry(
            result_dict=results,
            X=X,
            base_name=f"Clustering | k={k}, representative='{representative}'",
            features=selected_features,
            description=(
                f"Агломеративная кластеризация признаков по модулю корреляции |corr|.\n"
                f"Число кластеров = {k}. Способ выбора представителя кластера = '{representative}'."
            )
        )

    return results

def generate_subsets_by_vif_thresholds(
    X: pd.DataFrame,
    thresholds: Iterable[float] = (5, 10, 20)
) -> Dict[str, List[str]]:
    results = {}
    print("\n=== Генерация подмножеств: VIF thresholds ===")
    for thr in thresholds:
        print(f"\nВерхний порог: {thr}")
        cols = list(X.columns)
        X_sub = X[cols].copy()
        while True:
            vif = compute_vif(X_sub)
            #if vif.isnull().any() or len(vif) <= 1: # ошибочная строка: у np.ndarray нет атрибута 'isnull'
            #    break
            max_vif = vif.max()
            if max_vif <= thr:
                break
            worst = vif.idxmax()
            cols.remove(worst)
            X_sub = X[cols]

        add_subset_registry(
            result_dict=results,
            X=X,
            base_name=f"VIF threshold = {thr}",
            features=cols,
            description=f"Итеративное удаление признаков с VIF > {thr}. Оставшиеся признаки образуют подмножество."
        )

    return results

def generate_subsets_by_shap_clustering(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    cluster_ns: Iterable[int] = (2, 3, 4),
    representative_metric: str = "mean_abs_shap"
) -> Dict[str, List[str]]:

    if model is None:
        model = RandomForestRegressor(n_estimators=50, random_state=42)

    mdl = clone(model)
    mdl.fit(X, y)

    try:
        explainer = shap.Explainer(mdl, X)
        shap_vals = explainer(X)
        shap_abs_mean = np.abs(shap_vals.values).mean(axis=0)
    except Exception:
        if hasattr(mdl, "coef_"):
            shap_abs_mean = np.abs(mdl.coef_)
        elif hasattr(mdl, "feature_importances_"):
            shap_abs_mean = np.abs(mdl.feature_importances_)
        else:
            raise RuntimeError("Cannot compute SHAP or importance for given model")

    shap_series = pd.Series(shap_abs_mean, index=X.columns)

    corr = corr_matrix(X)
    dist = 1 - corr
    results = {}

    print("\n=== Генерация подмножеств: SHAP-based Clustering ===")

    for k in cluster_ns:
        print(f"\nk: {k}")
        clusterer = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = clusterer.fit_predict(dist.values)
        clusters = {}
        for feat, lab in zip(X.columns, labels):
            clusters.setdefault(lab, []).append(feat)

        selected_features = []
        for lab, feats in clusters.items():
            if representative_metric in ["mean_abs_shap", "max_shap"]:
                chosen = shap_series[feats].idxmax()
            else:
                chosen = feats[0]
            selected_features.append(chosen)

        add_subset_registry(
            result_dict=results,
            X=X,
            base_name=f"SHAP clustering | k={k}, representative='{representative_metric}'",
            features=selected_features,
            description=(
                "SHAP-based выбор представителя кластера.\n"
                f"Число кластеров = {k}, метрика представителя = '{representative_metric}'."
            )
        )

    return results

def generate_subsets_by_stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    base_estimator=None,
    n_bootstraps: int = 100,
    sample_fraction: float = 0.75,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    selection_thresholds: Iterable[float] = (0.6, 0.7, 0.8),
    random_state: Optional[int] = 0
) -> Dict[str, list]:
    """
    Генерация подмножеств признаков методом Stability Selection.
    Возвращает dict: ключи = frozenset(features) -> {"features": [...], "methods": [...]}
    Кроме того, в results["_stability_freq"] кладётся pd.Series частот (index = X.columns).
    """
    print("\n=== Генерация подмножеств: Stability Selection ===")
    # базовый оцениватель
    if base_estimator is None:
        base_estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                    max_iter=2000, random_state=random_state)
    # частоты
    freq = pd.Series(0.0, index=X.columns, dtype=float)

    rng = np.random.RandomState(random_state)
    n, _ = X.shape
    for b in range(n_bootstraps):
        try:
            idx = rng.choice(n, size=int(sample_fraction * n), replace=False)
        except Exception:
            idx = rng.choice(n, size=max(1, int(sample_fraction * n)), replace=False)
        Xb = X.iloc[idx].reset_index(drop=True)
        yb = y.iloc[idx].reset_index(drop=True)

        est = clone(base_estimator)
        try:
            est.fit(Xb, yb)

            if hasattr(est, "coef_"):
                coefs = est.coef_
            elif hasattr(est, "feature_importances_"):
                coefs = est.feature_importances_
            else:
                # если оцениватель ничего не дал — пропускаем
                continue

            # очень малый порог для определения "ненулевого" коэффициента
            selected = [col for col, c in zip(X.columns, coefs) if abs(float(c)) > 1e-12]
            if selected:
                freq[selected] += 1.0
        except Exception:
            # пропускаем неудачные итерации
            continue

    # нормируем
    freq /= float(n_bootstraps)
    print("\n=== Частоты выбора признаков (Stability Frequencies) ===")
    print(freq.sort_values(ascending=False))

    results: Dict[object, dict] = {}
    any_added = False

    # для каждого порога - формируем набор
    for thr in selection_thresholds:
        print(f"\nПорог для частоты: {thr}")
        chosen = list(freq[freq >= thr].index)
        if len(chosen) > 0:
            add_subset_registry(
                result_dict=results,
                X=X,
                base_name=f"Stability Selection | threshold={thr}",
                features=chosen,
                description=f"Признаки, выбранные с частотой >= {thr}."
            )
            any_added = True
        else:
            print(f"Этому порогу не удовлетворяет ни одно подмножество признаков")

    if not any_added:
        print("⚠ Stability Selection не выделила признаков ни при одном пороге.")

    # сохраняем частоты для дальнейшего анализа
    # НЕ кладём freq в results как подмножество; вместо этого вернём его отдельно (кортежем)
    #return results, freq
    return results

def generate_subsets_by_sparse_pca(
    X: pd.DataFrame, # матрица признаков
    #n_components_list: Iterable[int] = (1, 2, 3), # числа компонент
    n_components_list: Iterable[int], # числа компонент
    alpha_list: Iterable[float] = (1.0,), # значения alpha
    random_state: Optional[int] = 42 # для воспроизводимости
) -> Dict[str, list]:
    """
    Генерация подмножеств по методу SparsePCA.
    Для каждого (n_comp, alpha) собираем признаки, у которых нагрузка по любой компоненте
    превышает threshold (здесь threshold = 0.1 * max|loading| по компоненте).
    Результат — добавляется через add_subset_registry.

    Returns
    -------
    results: Dict[str, list]
        Словарь подмножеств признаков.
    """

    print("\n=== Генерация подмножеств: SparsePCA ===")
    results: Dict[object, dict] = {}
    any_added = False

    if not n_components_list:
        n_components_list = range(1, X.shape[1]+1)

    for n_comp in n_components_list: # число компонентов
        for alpha in alpha_list: # параметр регуляризации
            spca = SparsePCA(n_components=n_comp, alpha=alpha, random_state=random_state, max_iter=3000)
            try:
                spca.fit(X.values)
            except Exception as e:
                print(f"⚠ SparsePCA не сошелся: n_comp={n_comp}, alpha={alpha}, ошибка: {e}")
                continue

            loads:np.ndarray = spca.components_  # shape (n_comp, n_features)
            chosen_total = set()
            for comp_idx in range(loads.shape[0]): # индекс компоненты
                comp:np.ndarray = loads[comp_idx, :] # нагрузки по компоненте (размер n_features)
                loadings_abs = np.abs(comp)
                print("\nСтрока матрицы нагрузок (|loading|) для этого компонента:", loadings_abs)
                maxabs = float(np.max(loadings_abs))
                thr = 0.1 * maxabs # нижний порог выбора признаков по этой компоненте (если thr == 0, то все с abs>0 попадут
                print(f"max |элемент строки матрицы нагрузок|: {maxabs}, нижний порог выбора признаков по этой компоненте: {thr}")
                comp_features:list[str] = [col for col, val in zip(X.columns, comp) if abs(float(val)) > thr] # выбранные признаки по этой компоненте
                #print(f"n_comp={n_comp}, alpha={alpha}, компонент {comp_idx}: выбрано {len(comp_features)} признаков -> {comp_features}")
                #print(f"n_comp={n_comp}, alpha={alpha}, компонент {comp}: выбрано {len(selected_features)} признаков -> {{{', '.join(selected_features)}}}")
                #print(f"n_comp={n_comp}, alpha={alpha}, компонент {comp_idx}: выбрано {len(comp_features)} признаков -> {{{', '.join(comp_features)}}}")
                #chosen_total.update(comp_features)
                print(f"\nЧисло компонентов: {n_comp}, α: {alpha}, индекс компонента: {comp_idx}, выбрано {len(comp_features)} признаков: {{{', '.join(comp_features)}}}")
                
                # Регистрируем каждую компоненту как отдельное подмножество
                if comp_features:
                    add_subset_registry(
                        result_dict=results,
                        X=X,
                        base_name=f"SparsePCA | n_comp={n_comp}, alpha={alpha}, comp={comp_idx}",
                        features=comp_features,
                        description=f"SparsePCA. индекс компонента: {comp_idx}, число компонентов: {n_comp}, α:{alpha}"
                    )
                else:
                    print(f"⚠ SparsePCA не выбрала ни одного признака для n_comp={n_comp}, alpha={alpha}, comp={comp_idx}")

                # В любом случае суммируем для информации
                chosen_total.update(comp_features)              

            if len(chosen_total) > 0:
                any_added = True
            else:
                print(f"⚠ SparsePCA не выбрала ни одного подмножества признаков для n_comp={n_comp}, alpha={alpha}")

    if not any_added:
        print("⚠ SparsePCA не выделила ни одного подмножества признаков")

    return results

def generate_feature_subsets(
        X : pd.DataFrame, 
        y : pd.Series) -> set:
    """
    Создание подмножеств признаков.
    Версия работающая из py5.

    Parameters
    ----------
    X : pd.DataFrame
        Матрица признаков;
    y : pd.Series
        Целевой вектор.

    Returns
    -------
    feature_subsets : set
        Множество подмножеств признаков 
    """

    subsets: dict = {}          # здесь вы собираете все feature subsets
    #subsets_metadata: dict = {} # здесь будут такие вещи как _stability_freq

    # 3) Stability Selection
    gens = [    
        (generate_subsets_by_clustering, # 1) Агломеративная кластеризация по |corr|
         dict(X=X, y=y, n_clusters_list=(2, 3), representative="max_corr_with_y")),
        (generate_subsets_by_vif_thresholds, # 2) VIF thresholds
         dict(X=X, thresholds=(5, 10))),
        (generate_subsets_by_stability_selection, # 3) Stability Selection
         dict(X=X, y=y, n_bootstraps=200, sample_fraction=0.65, alpha=0.001,
              l1_ratio=0.7, selection_thresholds=(0.005, 0.5, 0.9, 1.1,), random_state=42)),
        (generate_subsets_by_sparse_pca, # 4) SparsePCA
         #dict(X=X, n_components_list=(1, 2), alpha_list=(0.01, 0.1), random_state=42)),
         dict(X=X, n_components_list=[], alpha_list=(0.01, 0.1), random_state=42)),
        (generate_subsets_by_shap_clustering, # 5) CHAP
         dict(X=X, y=y, model=RandomForestRegressor(n_estimators=50, random_state=0), cluster_ns=(2, 3)))
    ]

    for func, kwargs in gens:
        try:
            res = func(**kwargs)
        except Exception as e:
            print(f"⚠ Ошибка при вызове {func.__name__}: {e}")
            continue

        # диагностическое сообщение
        if res is None:
            print(f"⚠ {func.__name__} вернул None — пропускаем.")
            continue

        # === Добавляем специальную обработку для Stability Selection ===
        # Унифицируем результат: ожидаем dict
        if not isinstance(res, dict):
            print(f"⚠ {func.__name__} вернул {type(res)}, ожидается dict — пропускаем.")
            continue

        # Если Stability Selection — отделяем частоты
        if func == generate_subsets_by_stability_selection:
            #freq = res.get("_stability_freq", None)
            #if freq is not None:
            #    subsets_metadata["stability_freq"] = freq
            # Удаляем служебный ключ, оставляем только подмножества:
            res = {k: v for k, v in res.items() if k != "_stability_freq"}

        # Приводим каждый элемент к единому формату
        for key, features in res.items():
            # features может быть list или dict — нормализуем:
            if isinstance(features, dict):
                feat_list = features.get("features", [])
                desc = features.get("description", "")
            else:
                feat_list = list(features)
                desc = ""

            f_key = frozenset(feat_list)

            # Если такое подмножество уже есть — дополняем методами
            if f_key in subsets:
                subsets[f_key]["methods"].append(func.__name__)
            else:
                subsets[f_key] = {
                    "features": feat_list,
                    "methods": [func.__name__],
                    "description": desc,
                    "params": kwargs  # сохраняем параметры вызова метода
                }

        print(f"\n-> После {func.__name__}: всего подмножеств = {len(subsets)}")

        if not isinstance(res, dict):
            print(f"⚠ {func.__name__} вернул объект типа {type(res)} (ожидался dict) — пропускаем.")
            continue

        # объединяем: просто обновляем словарь; не присваиваем результат update()
        subsets.update(res)

        # Если это Stability Selection — сохраняем freq отдельно
        #if func == generate_subsets_by_stability_selection:
        #    freq = res.get("_stability_freq", None)
        #    if freq is not None:
        #        subsets["_stability_freq"] = freq
        #        print("\n=== Частоты выбора признаков (Stability Frequencies) ===")
        #        print(freq.sort_values(ascending=False))

        #print(f"-> После {func.__name__}: всего подмножеств = {len(subsets)}")

    # ---------------------------
    # Вывод всех уникальных подмножеств
    # ---------------------------
    print("\n=== Все собранные подмножества признаков ===")
    # соберём ключи-результаты (фильтруем служебные)

    items = list(subsets.items())
    items_sorted = sorted(items, key=lambda kv: (-len(kv[1]["features"]), ", ".join(kv[1]["features"])))

    for _, info in items_sorted:
        features_list = info["features"]
        methods = ", ".join(info.get("methods", []))
        print(f"{{{', '.join(features_list)}}}  | methods: [{methods}]")

    # печатаем stability frequencies в конце (если есть)
    #if "_stability_freq" in subsets:
    #    print("\n_stability_freq:")
    #    print(subsets["_stability_freq"].sort_values(ascending=False))

    # после сборки subsets: защита от None / неверного типа
    if subsets is None:
        print("⚠ subsets is None — создаём пустой словарь")
        subsets = {}
    if not isinstance(subsets, dict):
        print(f"⚠ subsets имеет тип {type(subsets)} — преобразуем в пустой dict")
        subsets = {}

    # print("subsets:\n", subsets) # подготовленное к возврату из функции (dict), вот образец:
    # subsets = {
    #     frozenset({'x2', 'x1'}): {'features': ['x1', 'x2'], 'methods': ["SHAP clustering | k=2, representative='mean_abs_shap'"]}, 
    #     frozenset({'x2', 'x4', 'x1'}): {'features': ['x1', 'x2', 'x4'], 'methods': ["Clustering | k=3, representative='max_corr_with_y'"]},
    #     ...
    # }
    
    # Создаем feature_subsets:dict, элементы которого - подмножества признаков, собранные в `subsets`.
    # Так как множества не могут иметь элементами другие множества, но могут иметь элементы типа frozenset,
    # мы преобразуем списки не во множества, а в frozenset.
    # feature_subsets = {
    #     frozenset({'x1', 'x2'}), 
    #     frozenset({'x1', 'x2', 'x4'}),
    #     ...
    # }
    # Так как `subsets` - словарь, то порядок в `feature_subsets` будет другим, чем в выводе элементов `subsets`
    feature_subsets = {}
    for key, value in subsets.items():
        #new_feature_subset_name = key
        value_ = value['features']
        feature_subsets[key] = value_
        print(key, " -> ", value_)
    # print("feature_subsets:\n", feature_subsets) # возвращаемое из функции (dict)
    print("generate_feature_subsets: штатный выход")
    return feature_subsets