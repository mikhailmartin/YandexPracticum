# Спринт 12 «Сборный Проект - 2»


# Проект «Восстановление золота из руды»


## Описание проекта

Необходимо подготоить прототип модели машинного обучения для «Цифры». Компания разрабатывает решения для эффективной
работы промышленных предприятий.

Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды. Будем использовать данные с
параметрами добычи и очистки.

Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.


## Используемые инструменты

- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`


## Общий вывод

В результате выполнения проекта мы получили две модели, которые предсказывают коэффициент восстановления золота из
золотосодержащей руды после этапа флотации и финальной очистки.

1. Для этого мы проверили, правильно ли вычисляется коэффициент в размеченных данных.
2. Предобработали данные, верно выбрав признаки для обучения моделей и заполнив пропуски.
3. Исследовали как меняются концентрации металлов на разных этапах очистки.
4. Проверили нет ли сдвига в целевом признаке между train и test.
5. Выбрали лучшие модели с использованием кросс-валидации. И наконец оценили модель на тестовой выборке.
