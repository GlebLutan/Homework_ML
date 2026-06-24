import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Разделить [0, 1, ..., num_objects - 1] на равные num_folds фолдов
       (последний фолд может быть длиннее) и вернуть num_folds пар train-val
       индексов.

    Параметры:
    num_objects: количество объектов в обучающем наборе
    num_folds: количество фолдов для разделения кросс-валидации

    Возвращает:
    список длины num_folds, где i-й элемент списка
    содержит кортеж из 2 numpy массивов, первый numpy массив
    содержит все индексы без i-го фолда, а второй
    содержит i-й фолд
    """
    Num_indexes_over_flod_without_last = num_objects // num_folds
    Num_indexes_in_last_flod = Num_indexes_over_flod_without_last + num_objects % num_folds
    X = np.arange(num_objects)
    folds = []
    for i in range(num_folds):
        if i == num_folds - 1:
            folds.append((X[:i * Num_indexes_over_flod_without_last], X[i * Num_indexes_over_flod_without_last:]))
        else:
            folds.append((np.concatenate((X[:i * Num_indexes_over_flod_without_last],
                                         X[(i + 1) * Num_indexes_over_flod_without_last:])),
                          X[i * Num_indexes_over_flod_without_last:(i + 1) * Num_indexes_over_flod_without_last]))
    return folds


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Принимает обучающие данные, вычисляет оценку кросс-валидации по
    сетке параметров (все возможные комбинации параметров)

    Параметры:
    X: обучающий набор
    y: обучающие метки
    parameters: словарь с ключами из
        {n_neighbors, metrics, weights, normalizers}, значения типа list,
        parameters['normalizers'] содержит кортежи (normalizer, normalizer_name)
        см. пример параметров в вашем jupyter notebook

    score_function: функция с входом (y_true, y_predict)
        которая выводит метрику оценки
    folds: вывод kfold_split
    knn_class: класс модели knn для обучения

    Возвращает:
    dict: ключ - кортеж (normalizer_name, n_neighbors, metric, weight),
    значение - средняя оценка по всем фолдам
    """

    otv = []

    for i in range(len(parameters['n_neighbors'])):
        for j in range(len(parameters['metrics'])):
            for k in range(len(parameters['weights'])):
                for m in range(len(parameters['normalizers'])):
                    score = 0
                    for train_index, val_index in folds:
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y[train_index], y[val_index]
                        normalizer, normalizer_name = parameters['normalizers'][m]
                        if normalizer is not None:
                            normalizer.fit(X_train)
                            X_train = normalizer.transform(X_train)
                            X_val = normalizer.transform(X_val)
                            del normalizer
                        model = knn_class(n_neighbors=parameters['n_neighbors'][i],
                                          metric=parameters['metrics'][j],
                                          weights=parameters['weights'][k])
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score += score_function(y_val, y_pred)
                    score /= len(folds)
                    otv.append((score, (parameters['normalizers'][m][1],
                                        parameters['n_neighbors'][i],
                                        parameters['metrics'][j],
                                        parameters['weights'][k])))
    return {param: score for score, param in otv}
