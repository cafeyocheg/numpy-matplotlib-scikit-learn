#!/usr/bin/env python
# coding: utf-8

# # Тема “Визуализация данных в Matplotlib”

# # Задание 1.
# Загрузите модуль pyplot библиотеки matplotlib с псевдонимом plt, а также библиотеку numpy с псевдонимом np.
# Примените магическую функцию %matplotlib inline для отображения графиков в Jupyter Notebook и настройки конфигурации ноутбука со значением 'svg' для более четкого отображения графиков.
# Создайте список под названием x с числами 1, 2, 3, 4, 5, 6, 7 и список y с числами 3.5, 3.8, 4.2, 4.5, 5, 5.5, 7.
# С помощью функции plot постройте график, соединяющий линиями точки с горизонтальными координатами из списка x и вертикальными - из списка y.
# Затем в следующей ячейке постройте диаграмму рассеяния (другие названия - диаграмма разброса, scatter plot).

# In[20]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

x = [1, 2, 3, 4, 5, 6, 7]
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]


# In[18]:


plt.plot(x, y)
plt.title("Линейный график функции", color="#111111", family="serif", loc='right')
plt.xlabel("Переменная x")
plt.ylabel("Функция f(x)")
plt.show()


# In[19]:


plt.scatter(x, y)
plt.title('Диаграмма разброса', color="#222222", family="serif", loc='right')
plt.xlabel("Переменная x")
plt.ylabel("Функция f(x)")
plt.show()


# # Задание 2.
# С помощью функции linspace из библиотеки Numpy создайте массив t из 51 числа от 0 до 10 включительно.
# Создайте массив Numpy под названием f, содержащий косинусы элементов массива t.
# Постройте линейную диаграмму, используя массив t для координат по горизонтали,а массив f - для координат по вертикали. Линия графика должна быть зеленого цвета.
# Выведите название диаграммы - 'График f(t)'. Также добавьте названия для горизонтальной оси - 'Значения t' и для вертикальной - 'Значения f'.
# Ограничьте график по оси x значениями 0.5 и 9.5, а по оси y - значениями -2.5 и 2.5.

# In[22]:


t = np.linspace(0, 10, 51)
f = np.cos(t)
print(t)


# In[33]:


plt.plot(t, f, color='red', linestyle="dashed")
plt.title('График f(t)')
plt.xlabel("Значения t")
plt.ylabel("Значения f")
plt.grid(axis="y", color="lightgrey")
plt.axis([0.5, 9.5, -2.5, 2.5])

plt.show()


# # *Задание 3
# С помощью функции linspace библиотеки Numpy создайте массив x из 51 числа от -3 до 3 включительно.
# Создайте массивы y1, y2, y3, y4 по следующим формулам:
# y1 = x**2
# y2 = 2 * x + 0.5
# y3 = -3 * x - 1.5
# y4 = sin(x)
# Используя функцию subplots модуля matplotlib.pyplot, создайте объект matplotlib.figure.Figure с названием fig и массив объектов Axes под названием ax,причем так, чтобы у вас было 4 отдельных графика в сетке, состоящей из двух строк и двух столбцов. В каждом графике массив x используется для координат по горизонтали.В левом верхнем графике для координат по вертикали используйте y1,в правом верхнем - y2, в левом нижнем - y3, в правом нижнем - y4.Дайте название графикам: 'График y1', 'График y2' и т.д.
# Для графика в левом верхнем углу установите границы по оси x от -5 до 5.
# Установите размеры фигуры 8 дюймов по горизонтали и 6 дюймов по вертикали.
# Вертикальные и горизонтальные зазоры между графиками должны составлять 0.3.

# In[54]:


x = np.linspace(-3, 3, 51)
y1 = x ** 2
y2 = 2 * x + 0.5
y3 = -3 * x - 1.5
y4 = np.sin(x)

fig, ax = plt.subplots(nrows=2, ncols=2)
ax1, ax2, ax3, ax4 = ax.flatten()

ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y3)
ax4.plot(x, y4)
ax1.set_title('График y1')
ax1.set_xlim([-5, 5])
ax2.set_title('График y2')
ax3.set_title('График y3')
ax4.set_title('График y4')

fig.set_size_inches(8, 6)
plt.subplots_adjust(wspace=0.3, hspace=0.3)


# # *Задание 4
# В этом задании мы будем работать с датасетом, в котором приведены данные по мошенничеству с кредитными данными: Credit Card Fraud Detection (информация об авторах: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015).
# Ознакомьтесь с описанием и скачайте датасет creditcard.csv с сайта Kaggle.com по ссылке:
# Credit Card Fraud Detection
# Данный датасет является примером несбалансированных данных, так как мошеннические операции с картами встречаются реже обычных.
# Импортируйте библиотеку Pandas, а также используйте для графиков стиль “fivethirtyeight”.
# Посчитайте с помощью метода value_counts количество наблюдений для каждого значения целевой переменной Class и примените к полученным данным метод plot, чтобы построить столбчатую диаграмму. Затем постройте такую же диаграмму, используя логарифмический масштаб.
# На следующем графике постройте две гистограммы по значениям признака V1 - одну для мошеннических транзакций (Class равен 1) и другую - для обычных (Class равен 0). Подберите значение аргумента density так, чтобы по вертикали графика было расположено не число наблюдений, а плотность распределения. Число бинов должно равняться 20 для обеих гистограмм, а коэффициент alpha сделайте равным 0.5, чтобы гистограммы были полупрозрачными и не загораживали друг друга. Создайте легенду с двумя значениями: “Class 0” и “Class 1”. Гистограмма обычных транзакций должна быть серого цвета, а мошеннических - красного. Горизонтальной оси дайте название “V1”.

# In[14]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')


# In[23]:


creditcard = pd.read_csv('creditcard.csv')
creditcard.tail(5)


# In[20]:


class_list = creditcard['Class'].value_counts()
print(class_list)


# In[27]:


class_list.plot(kind='bar')
plt.show()


# In[28]:


class_list.plot(kind='barh', logx=True)
plt.show()


# In[35]:


class0 = creditcard.loc[creditcard['Class'] == 0, ['V1']]
class1 = creditcard.loc[creditcard['Class'] == 1, ['V1']]

plt.hist(class0['V1'], bins=20, density=True, alpha=0.5, label='Class 0', color='grey')
plt.hist(class1['V1'], bins=20, density=True, alpha=0.5, label='Class 1', color='red')
plt.legend()
plt.show()

