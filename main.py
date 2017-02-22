from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import random

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RUN PREFERENCES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
show_plot_init = False  # to show initial data plot
print_x_and_y = False   # print all x's and y's
show_final_plot = True  # show final plot
show_convergence = True     # 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DATA PREPARATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
iris = datasets.load_iris()

# print(iris.DESCR)

x = []
for i in iris.data:
    x.append(i[0])
y = []
for i in iris.data:
    y.append(i[2])


if print_x_and_y:
    print("{0:>4} | {1:>2}\n--------------".format("X", "Y"))
    for i in range(len(x)):
        print("{0:>4.1f} | {1:4.1f}".format(x[i], y[i]))

    if show_plot_init:
        plt.plot(x, y, "o")
        plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GDM METHOD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

gamma = 0.01
m = random.random()
b = random.random()
print(m, b)
error = 1


def calc_df_dm():
    df_dm = 0
    for j in range(len(x)):
        df_dm += x[j]*(y[j] - (m*x[j] + b))
    df_dm /= len(x)
    df_dm *= -2
    return df_dm


def calc_df_db():
    df_db = 0
    for j in range(len(x)):
        df_db += (y[j] - (m*x[j] + b))
    df_db /= len(x)
    df_db *= -2
    return df_db

n = 0
while n < 10000:
    n += 1
    m -= gamma*calc_df_dm()
    b -= gamma*calc_df_db()
    if show_final_plot:
        print(m, b)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FINAL GRAPH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

lin_y = []
for i in range(len(x)):
    lin_y.append(m*x[i]+b)

slope, intercept, corr, p_value, std_error = stats.linregress(x, y)
lsq = [intercept + slope * i for i in x]

if show_final_plot:
    plt.plot(x, lin_y)
    plt.plot(x, y, "o")
    plt.show()
    plt.plot(x, lsq, 'r')

print("Final m, b: {0}  {1}".format(m, b))
print("Expected m, b: {0}  {1}".format(slope, intercept))