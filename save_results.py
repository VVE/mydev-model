import os

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

RESULTS_FILE = "model_results.tsv"

def save_results_with_details(
    results: dict # результаты для одного сочетания модель х параметры модели x набор признаков
    ) -> None:
    """
    Принимает уже подготовленный (sanitize_result_for_saving) словарь и дозаписывает строку в TSV.
    Не изменяет входной словарь.
    """

    #print("results:", results)
    df = pd.DataFrame([results])

    # Если файла еще нет — создаём (с заголовками)
    if not os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, sep="\t", index=False, mode='w')
    else:
        df.to_csv(RESULTS_FILE, sep="\t", index=False, mode='a', header=False)

    # Печатать компактно — одна строка
    print(df)
    print(f"{results.get('model_name')} params={results.get('params')} on {results.get('features')} | R2={results.get('cv_adj_r2_mean'):.4f}")
    return