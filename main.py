import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from generate_feature_subsets import generate_feature_subsets
from load_dataset import load_synthetic_data
from postprocess import \
    reconstruct_coefficients_original_scale  # функция обратного преобразования
from save_results import save_results_with_details
from train_eval import run_one_model_on_subset
from utils import sanitize_result_for_saving


def select_best_model(results_list):
    """
    Выбираем лучшую модель по максимальному test_adj_r2 или другому критерию.
    Возвращает словарь (строку результата).
    """
    if not results_list:
        return None
    return max(results_list, key=lambda r: r.get("test_adj_r2", -np.inf))

def main():
    # 1. Загружаем исходные данные (еще НЕ масштабированные)
    X_raw, y_raw = load_synthetic_data() # X_raw — DataFrame, y_raw — Series/array

    # 2. Масштабируем X и y до [0, 1]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw.values.reshape(-1, 1)).ravel()

    # Восстанавливаем DataFrame с правильными именами колонок
    X = pd.DataFrame(X_scaled, columns=X_raw.columns)

    # 3. Генерация подмножеств признаков из уже масштабированных данных
    feature_subsets = generate_feature_subsets(X, y_scaled)

    # 4. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=0.2, random_state=42
    )

    # 5. Модели
    linear_models = {
        "OLS": {"params": [{}]},
        #"Ridge": {"params": [{"alpha": a} for a in [0.1, 1, 10]]},
        #"Lasso": {"params": [{"alpha": a} for a in [0.001, 0.01, 0.1]]}
    }

    accepted_results = []   # <--- собираем сюда все прошедшие модели

    for subset_name, features in feature_subsets.items():
        X_train_sub = X_train[features]
        X_test_sub = X_test[features]

        for model_name, cfg in linear_models.items():
            for params in cfg["params"]:
                if model_name == "OLS":
                    model = LinearRegression(**params)
                elif model_name == "Ridge":
                    model = Ridge(**params)
                elif model_name == "Lasso":
                    model = Lasso(**params)
                else:
                    raise ValueError(f"Неизвестная модель: {model_name}")

                # Запуск проверки
                results = run_one_model_on_subset(
                    X_train=X_train_sub, y_train=y_train, 
                    X_test=X_test_sub, y_test=y_test,
                    subset_name=subset_name,
                    features=features,
                    model_name=model_name,
                    params=params,
                    model=model
                )

                if results is not None:
                    # 1) Приводим результат к плоскому формату без вложенных структур
                    results_flat = sanitize_result_for_saving(results)

                    # 2) Добавляем в список принятых моделей
                    accepted_results.append(results_flat)

                    # 3) Сохраняем в TSV-файл (функция ожидает уже подготовленный словарь)
                    save_results_with_details(results_flat)

    # 6. Выбор лучшей модели
    if not accepted_results:
        print("Нет принятых моделей.")
        return

    best = select_best_model(accepted_results)
    print("\n=== Лучшая модель ===")
    print(best)

    # 7. Обратное преобразование коэффициентов
    # Повторим обучение лучшей модели, чтобы достать coef_ и intercept_
    best_model_name = best["model_name"]
    best_params = best["params"]
    best_features = best["features"]

    if best_model_name == "OLS":
        final_model = LinearRegression(**best_params)
    elif best_model_name == "Ridge":
        final_model = Ridge(**best_params)
    elif best_model_name == "Lasso":
        final_model = Lasso(**best_params)

    X_train_best = X_train[best_features]
    final_model.fit(X_train_best, y_train)

    # Берём коэффициенты и смещение в масштабе [0..1]
    coef_scaled = final_model.coef_
    intercept_scaled = final_model.intercept_

    # Преобразуем обратно
    coef_orig, intercept_orig = reconstruct_coefficients_original_scale(
        coef_scaled=coef_scaled,
        intercept_scaled=intercept_scaled,
        features=best_features,
        scaler_X=scaler_X,
        scaler_y=scaler_y
    )

    print("\nКоэффициенты в исходном масштабе:")
    for f_name, c in zip(best_features, coef_orig):
        print(f"{f_name}: {c:.6f}")
    print("Intercept:", intercept_orig)

if __name__ == "__main__":
    main()