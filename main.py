from sklearn import datasets
import matplotlib.pyplot as plt
from scipy import stats
import random

# Creator: Robert Costales
# Date: 2/20/2017
# Purpose: Find best fit line for data using GDM

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RUN PREFERENCES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

show_plot_init = False  # to show initial data plot
print_x_and_y = False   # print all x's and y's
show_final_plot = True  # show final plot
show_convergence = True     # show the updated values for m and b

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DATA PREPARATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

iris = datasets.load_iris()

# print(iris.DESCR)

x = []
for i in iris.data:     # creates x data
    x.append(i[0])
y = []
for i in iris.data:     # creates y data
    y.append(i[2])


if print_x_and_y:
    print("{0:>4} | {1:>2}\n--------------".format("X", "Y"))   # prints x and y values
    for i in range(len(x)):
        print("{0:>4.1f} | {1:4.1f}".format(x[i], y[i]))

if show_plot_init:              # shows initial plot of x and y
    plt.plot(x, y, "o")
    plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GDM METHOD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

gamma = 0.01            # initialized value of gamma
m = random.random()     # set m and b equal to random values between 0 and 1
b = random.random()
print(m, b)


def calc_df_dm():           # Calculates df/dm for each update
    df_dm = 0
    for j in range(len(x)):
        df_dm += x[j]*(y[j] - (m*x[j] + b))
    df_dm /= len(x)
    df_dm *= -2
    return df_dm


def calc_df_db():           # Calculates df/db for each update
    df_db = 0
    for j in range(len(x)):
        df_db += (y[j] - (m*x[j] + b))
    df_db /= len(x)
    df_db *= -2
    return df_db

n = 0
while n < 15000:            # Can update many times because data set is small, and doesn't take too long
                            # Could easily make it check for error each time though
    n += 1                      # iteration number
    m -= gamma*calc_df_dm()
    b -= gamma*calc_df_db()
    if show_final_plot:
        print("{0} | {1:.4f}, {2:.4f}".format(n, m, b))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FINAL GRAPH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

lin_y = []                          # Creates plot of linear function we found
for i in range(len(x)):
    lin_y.append(m*x[i]+b)
slope, intercept, corr, p_value, std_error = stats.linregress(x, y)
lsq = [intercept + slope * i for i in x]

if show_final_plot:             # Shows final plot of linear function we found and original data
    plt.plot(x, lin_y)
    plt.plot(x, y, "o")
    plt.show()
    # plt.plot(x, lsq, 'r')       # We can even include the least squares line, which is usually right on top of ours

print("Final m, b: {0:.4f}  {1:.4f}".format(m, b))
print("Expected m, b: {0:.4f}  {1:.4f}".format(slope, intercept))
