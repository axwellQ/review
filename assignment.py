"""
Задание 3: Анализ датасета Diabetes
Цель: Анализ данных диабета - регрессионная задача
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data() -> pd.DataFrame:
    """
    Загрузить датасет Diabetes и конвертировать в DataFrame.

    Возвращает:
        pd.DataFrame: DataFrame с 10 признаками и целевой переменной 'target'.
        Признаки: age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
        Целевая переменная: target (прогрессия заболевания диабетом)

    Returns:
        pd.DataFrame: DataFrame с признаками и целевой переменной 'target'.
    """
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df = df.assign(target=diabetes.target)
    return df


def target_analysis(df: pd.DataFrame) -> None:
    """
    Анализ целевой переменной (прогрессия болезни).

    Вычисляет и выводит:
    - Среднее, медиана, стандартное отклонение
    - Минимум, максимум, размах
    - Квартили (25%, 50%, 75%)

    Args:
        df (pd.DataFrame): DataFrame с колонкой 'target'.
    """
    target = df['target']
    print("\nАнализ целевой переменной:")
    print(f"Среднее: {target.mean():.2f}")
    print(f"Медиана: {target.median():.2f}")
    print(f"Стандартное отклонение: {target.std():.2f}")
    print(f"Минимум: {target.min():.2f}")
    print(f"Максимум: {target.max():.2f}")
    print(f"Размах: {target.max() - target.min():.2f}")
    print(f"25% квантиль: {target.quantile(0.25):.2f}")
    print(f"50% квантиль (медиана): {target.quantile(0.50):.2f}")
    print(f"75% квантиль: {target.quantile(0.75):.2f}")


def feature_statistics(df: pd.DataFrame) -> None:
    """
    Вычислить статистику по признакам.

    Для каждого числового признака вычисляет:
    - Среднее, стандартное отклонение
    - Минимум, максимум

    Args:
        df (pd.DataFrame): DataFrame с признаками и целевой переменной.
    """
    print("\nСтатистика по признакам:")
    feature_cols = [col for col in df.columns if col != 'target']
    features_df = df[feature_cols]

    stats = features_df.describe()
    relevant_stats = stats.loc[['mean', 'std', 'min', 'max']]
    print(relevant_stats.round(3))


def visualize_target(df: pd.DataFrame) -> None:
    """
    Визуализировать распределение целевой переменной.

    Создает:
    - Гистограмму распределения с линией среднего
    - KDE график

    Сохраняет как: 03_diabetes_target_distribution.png

    Args:
        df (pd.DataFrame): DataFrame с колонкой 'target'.
    """
    target = df['target']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Гистограмма
    axes[0].hist(target, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(target.mean(), color='red', linestyle='--', label=f'Среднее: {target.mean():.2f}')
    axes[0].set_title('Распределение целевой переменной')
    axes[0].set_xlabel('Значение прогрессии болезни')
    axes[0].set_ylabel('Плотность')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # KDE график
    sns.kdeplot(target, ax=axes[1], color='orange')
    axes[1].set_title('KDE график целевой переменной')
    axes[1].set_xlabel('Значение прогрессии болезни')
    axes[1].set_ylabel('Плотность')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('03_diabetes_target_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_features(df: pd.DataFrame) -> None:
    """
    Визуализировать распределение признаков.

    Создает гистограммы для всех 10 признаков в сетке 4x3.

    Сохраняет как: 03_diabetes_features_distribution.png

    Args:
        df (pd.DataFrame): DataFrame с признаками и целевой переменной.
    """
    feature_cols = [col for col in df.columns if col != 'target']
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = 4  # 10 признаков, сетка 4x3 даст 12 ячеек

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 16))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].hist(df[col], bins=20, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[i].set_title(f'Распределение {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Плотность')
        axes[i].grid(axis='y', alpha=0.3)

    # Скрываем лишние subplot'ы (11 и 12)
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('03_diabetes_features_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def scatter_features_vs_target(df: pd.DataFrame) -> None:
    """
    Scatter plots признаков относительно целевой переменной.

    Создает scatter plot для каждого признака против целевой переменной.

    Сохраняет как: 03_diabetes_features_vs_target.png

    Args:
        df (pd.DataFrame): DataFrame с признаками и целевой переменной.
    """
    feature_cols = [col for col in df.columns if col != 'target']
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 16))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].scatter(df[col], df['target'], alpha=0.6, color='coral', s=20)
        axes[i].set_title(f'{col} vs Target')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Target (Прогрессия болезни)')
        axes[i].grid(axis='y', alpha=0.3)

    # Скрываем лишние subplot'ы
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('03_diabetes_features_vs_target.png', dpi=150, bbox_inches='tight')
    plt.show()


def correlation_analysis(df: pd.DataFrame) -> None:
    """
    Анализ корреляций между признаками и целевой переменной.

    Вычисляет корреляцию Пирсона каждого признака с целевой переменной,
    сортирует по абсолютной величине и создает горизонтальную столбчатую диаграмму.

    Сохраняет как: 03_diabetes_correlation_bars.png

    Args:
        df (pd.DataFrame): DataFrame с признаками и целевой переменной.
    """
    feature_cols = [col for col in df.columns if col != 'target']
    correlations = {}
    for col in feature_cols:
        corr_val = df[col].corr(df['target'])
        correlations[col] = corr_val

    corr_series = pd.Series(correlations)
    sorted_corr = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)

    print("\nКорреляции признаков с целевой переменной (отсортированы по абсолютной величине):")
    for feature, corr in sorted_corr.items():
        print(f"{feature}: {corr:.3f}")

    # Визуализация корреляций
    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0 else 'blue' for x in sorted_corr.values]
    sorted_corr.plot(kind='barh', color=colors, alpha=0.7)
    plt.title('Корреляция признаков с целевой переменной', fontsize=14, fontweight='bold')
    plt.xlabel('Коэффициент корреляции Пирсона', fontsize=12)
    plt.ylabel('Признаки', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('03_diabetes_correlation_bars.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Главная функция"""
    print("=" * 60)
    print("ЗАДАНИЕ 3: EXPLORATORY DATA ANALYSIS - DIABETES DATASET")
    print("=" * 60)

    df = load_data()
    print(f"\nДатасет загружен. Размер: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head().round(3))

    target_analysis(df)
    feature_statistics(df)
    visualize_target(df)
    visualize_features(df)
    scatter_features_vs_target(df)
    correlation_analysis(df)

    print("\n" + "=" * 60)
    print("Анализ завершен!")
    print("=" * 60)


if __name__ == "__main__":
    main()