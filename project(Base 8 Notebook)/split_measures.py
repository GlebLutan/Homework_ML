import numpy as np


def evaluate_measures(sample):
    """Вычислить меры качества разбиения (каждый узел отдельно).

    Используйте натуральный логарифм (например, np.log) для оценки значения меры энтропии.

    Параметры
    ----------
    sample : список целых чисел. Размер списка равен числу объектов в текущем узле. Целые
    значения соответствуют меткам классов объектов в узле.

    Возвращает
    -------
    measures - словарь, содержащий три значения качества разбиения.
    Пример результата:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """

    if(sample == None or sample == []):
        return {'gini': float(0), 'entropy': float(0), 'error': float(0)}

    sample = np.array(sample)
    print(sample)
    other = np.bincount(sample) / len(sample)
    gini = np.sum((1 - other)*other)

    entropy = -np.sum(other * np.log(other))

    error = 1 - np.max(other)

    measures = {'gini': float(gini), 'entropy': float(entropy), 'error': float(error)}

    return measures