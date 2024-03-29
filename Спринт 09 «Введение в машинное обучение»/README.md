# Спринт 9 «Введение в машинное обучение»


# Проект «Рекомендация тарифов»


## Описание проекта

Оператор мобильной связи «Мегалайн» выяснил: многие клиенты пользуются архивными тарифами. Они хотят построить систему,
способную проанализировать поведение клиентов и предложить пользователям новый тариф: «Смарт» или «Ультра».

В нашем распоряжении данные о поведении клиентов, которые уже перешли на эти тарифы. Нужно построить модель для задачи
классификации, которая выберет подходящий тариф.


## Используемые инструменты

- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`


## Общий вывод
Построили модель для предложения тарифа клиенту на RandomForest c перфомансом на тестовой выборке `ROC-AUC = 0.83`.

Можно ещё попробовать построить модель на GBDT (`CatBoost`/`LightGBM`) с оптимизацией гиперпараметров через `optuna`.
Этот подход должен увеличить перфоманс модели.
