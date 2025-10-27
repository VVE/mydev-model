import numpy as np


def reconstruct_coefficients_original_scale(
    coef_scaled, 
    intercept_scaled, 
    features, 
    scaler_X, 
    scaler_y
):
    """
    Преобразует коэффициенты линейной модели из масштабированных данных (MinMaxScaler)
    обратно в оригинальный масштаб.

    coef_scaled       — коэффициенты модели в масштабе [0..1]
    intercept_scaled  — свободный член модели в масштабе [0..1]
    features          — список признаков, соответствующий coef_scaled
    scaler_X, scaler_y — обученные MinMaxScaler для X и y
    """

    # Преобразование коэффициентов:
    # y = a0 + Σ(ai * x_i_scaled)
    # где x_i_scaled = (x_i - min_i)/(max_i - min_i)
    # Тогда в оригинальном масштабе:
    # a_i_orig = ai * (y_max - y_min) / (x_max_i - x_min_i)
    # intercept_orig = y_min + (a0 * (y_max - y_min)
    #                          - Σ(a_i_orig * x_min_i))
    y_min = scaler_y.data_min_[0]
    y_max = scaler_y.data_max_[0]
    y_range = y_max - y_min

    X_min = scaler_X.data_min_
    X_max = scaler_X.data_max_
    X_range = X_max - X_min

    coef_orig = []
    for i, f in enumerate(features):
        xi_idx = list(scaler_X.feature_names_in_).index(f)
        xi_scale = X_range[xi_idx]
        if xi_scale == 0:
            coef_orig.append(0.0)
        else:
            coef_orig.append(coef_scaled[i] * y_range / xi_scale)

    coef_orig = np.array(coef_orig)

    # Свободный член:
    intercept_orig = (
        y_min 
        + intercept_scaled * y_range
        - np.sum(coef_orig * X_min[[list(scaler_X.feature_names_in_).index(f) for f in features]])
    )

    return coef_orig, intercept_orig