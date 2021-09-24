#!/usr/bin/env python
# coding: utf-8

# # Задание 1
# Импортируйте библиотеки pandas и numpy.
# Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn. Создайте датафреймы X и y из этих данных.
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test) с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 30% от всех данных, при этом аргумент random_state должен быть равен 42.
# Создайте модель линейной регрессии под названием lr с помощью класса LinearRegression из модуля sklearn.linear_model.
# Обучите модель на тренировочных данных (используйте все признаки) и сделайте предсказание на тестовых.
# Вычислите R2 полученных предказаний с помощью r2_score из модуля sklearn.metrics.

# In[25]:


import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
boston = load_boston()
data = boston["data"]

boston = load_boston()
boston.keys()


# In[26]:


feature_names = boston["feature_names"]

X = pd.DataFrame(data, columns=feature_names)
X.info()


# In[27]:


target = boston["target"]

y = pd.DataFrame(target, columns=['price'])
# y.tail(192)
y.info()


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[29]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[30]:


lr.fit(X_train, y_train)


# In[32]:


y_pred = lr.predict(X_test)

len(y_pred)


# In[33]:


check_test = pd.DataFrame({
    "y_test": y_test["price"],
    "y_pred": y_pred.flatten(),
})

check_test["error"] = check_test["y_pred"] - check_test["y_test"]
check_test


# In[34]:


from sklearn.metrics import r2_score

r2_score_lr = r2_score(check_test["y_pred"], check_test["y_test"])
r2_score_lr


# In[35]:


from sklearn.metrics import mean_squared_error
mean_squared_error = mean_squared_error(check_test["y_pred"], check_test["y_test"])

from sklearn.metrics import mean_absolute_error
mean_absolute_error = mean_absolute_error(check_test["y_pred"], check_test["y_test"])

mean_squared_error, mean_absolute_error


# # Задание 2
# Создайте модель под названием model с помощью RandomForestRegressor из модуля sklearn.ensemble.
# Сделайте агрумент n_estimators равным 1000,
# max_depth должен быть равен 12 и random_state сделайте равным 42.
# Обучите модель на тренировочных данных аналогично тому, как вы обучали модель LinearRegression,
# но при этом в метод fit вместо датафрейма y_train поставьте y_train.values[:, 0],
# чтобы получить из датафрейма одномерный массив Numpy,
# так как для класса RandomForestRegressor в данном методе для аргумента y предпочтительно применение массивов вместо датафрейма.
# Сделайте предсказание на тестовых данных и посчитайте R2. Сравните с результатом из предыдущего задания.
# Напишите в комментариях к коду, какая модель в данном случае работает лучше.

# In[8]:


import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
boston = load_boston()
data = boston["data"]

feature_names = boston["feature_names"]
X = pd.DataFrame(data, columns=feature_names)

target = boston["target"]
y = pd.DataFrame(target, columns=['price'])


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[13]:


from sklearn.ensemble import RandomForestRegressor


# In[14]:


model_rfr = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)


# In[15]:


model_rfr.fit(X_train, y_train.values[:, 0])


# In[16]:


y_pred_rfr = model_rfr.predict(X_test)

check_test_rfr = pd.DataFrame({
    "y_test": y_test["price"], 
    "y_pred_rfr": y_pred_rfr.flatten()})

check_test_rfr.head()


# In[23]:


from sklearn.metrics import mean_squared_error
mean_squared_error_rfr = mean_squared_error(check_test_rfr["y_pred_rfr"], check_test_rfr["y_test"])

from sklearn.metrics import mean_absolute_error
mean_absolute_error_rfr = mean_absolute_error(check_test_rfr["y_pred_rfr"], check_test_rfr["y_test"])

mean_squared_error_rfr, mean_absolute_error_rfr


# In[38]:


from sklearn.metrics import r2_score

r2_score_rfr = r2_score(check_test_rfr["y_pred_rfr"], check_test_rfr["y_test"])


# In[43]:


print(r2_score_lr,'\t', r2_score_rfr)


# Здесь видно, что и по абсолютной и среднеквадратичной ошибке (чем меньше, тем лучше), и по r2_score (чем больше, тем лучше), Случайный лес уделывает линейную регрессию. 
# Наверно, можно еще поэкспериментировать с параметрами модели вроде max_features, но точность 0.848 это уже хорошо.

# # *Задание 3
# Вызовите документацию для класса RandomForestRegressor,
# найдите информацию об атрибуте feature_importances_.
# С помощью этого атрибута найдите сумму всех показателей важности,
# установите, какие два признака показывают наибольшую важность.

# In[45]:


model_rfr.feature_importances_


# In[51]:


feature_importance = pd.DataFrame({'name':X.columns, 
                                   'feature_importance':model_rfr.feature_importances_}, 
                                  columns=['feature_importance', 'name'])

feature_importance.nlargest(2, 'feature_importance')


# 2 наиболее важных признака - RM и LSTAT.

# # *Задание 4
# В этом задании мы будем работать с датасетом, с которым мы уже знакомы по домашнему заданию
# по библиотеке Matplotlib, это датасет Credit Card Fraud Detection.Для этого датасета мы будем решать
# задачу классификации - будем определять,какие из транзакций по кредитной карте являются
# мошенническими.Данный датасет сильно несбалансирован (так как случаи мошенничества
# относительно редки),так что применение метрики accuracy не принесет пользы и не поможет выбрать
# лучшую модель.Мы будем вычислять AUC, то есть площадь под кривой ROC.
# Импортируйте из соответствующих модулей RandomForestClassifier, GridSearchCV и train_test_split.
# Загрузите датасет creditcard.csv и создайте датафрейм df.
# С помощью метода value_counts с аргументом normalize=True убедитесь в том, что выборка
# несбалансирована. Используя метод info, проверьте, все ли столбцы содержат числовые данные и нет
# ли в них пропусков.Примените следующую настройку, чтобы можно было просматривать все столбцы
# датафрейма:
# pd.options.display.max_columns = 100.
# Просмотрите первые 10 строк датафрейма df.
# Создайте датафрейм X из датафрейма df, исключив столбец Class.
# Создайте объект Series под названием y из столбца Class.
# Разбейте X и y на тренировочный и тестовый наборы данных при помощи функции train_test_split, используя аргументы: test_size=0.3, random_state=100, stratify=y.
# У вас должны получиться объекты X_train, X_test, y_train и y_test.
# Просмотрите информацию о их форме.
# Для поиска по сетке параметров задайте такие параметры:
# parameters = [{'n_estimators': [10, 15],
# 'max_features': np.arange(3, 5),
# 'max_depth': np.arange(4, 7)}]
# Создайте модель GridSearchCV со следующими аргументами:
# estimator=RandomForestClassifier(random_state=100),
# param_grid=parameters,
# scoring='roc_auc',
# cv=3..
# Обучите модель на тренировочном наборе данных (может занять несколько минут).
# Просмотрите параметры лучшей модели с помощью атрибута best_params_.
# Предскажите вероятности классов с помощью полученнной модели и метода predict_proba.
# Из полученного результата (массив Numpy) выберите столбец с индексом 1 (вероятность класса 1) и
# запишите в массив y_pred_proba. Из модуля sklearn.metrics импортируйте метрику roc_auc_score.
# Вычислите AUC на тестовых данных и сравните с результатом,полученным на тренировочных данных,
# используя в качестве аргументов массивы y_test и y_pred_proba.

# In[57]:


df = pd.read_csv('creditcard.csv')


# In[58]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[59]:


df['Class'].value_counts(normalize=True)


# In[60]:


df.info()


# In[62]:


pd.options.display.max_columns=100


# In[63]:


df.head(10)


# In[68]:


X = df.drop('Class', axis=1)
X.head(3)


# In[69]:


y = df['Class']
y.head(3)


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)


# In[72]:


X_train.shape, X_test.shape


# In[74]:


y_train.shape, y_test.shape


# In[75]:


parameters = [{'n_estimators': [10, 15],
'max_features': np.arange(3, 5),
'max_depth': np.arange(4, 7)}]


# In[76]:


model_gscv = GridSearchCV(estimator=RandomForestClassifier(random_state=100),
param_grid=parameters,
scoring='roc_auc',
cv=3
)


# In[78]:


model_gscv.fit(X_train, y_train)


# In[79]:


model_gscv.best_params_


# In[82]:


y_pred_gscv = model_gscv.predict_proba(X_test)


# In[86]:


y_pred_proba = y_pred[:, 1]
y_pred_proba


# In[90]:


from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_pred_proba)

