import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets

# load the dataset
iris = datasets.load_iris()
# access the data
data = iris.data
# set the columns (variables) we care about to be x and y
x = data[:,0]; y = data[:,2]

# set up the graph
plt.plot(x,y,'o')
plt.title("Does Sepal Length Determine Petal Length?")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
# find the least-squares solution and place line on graph
slope, intercept, corr, p_value, std_error = stats.linregress(x,y)
lsq = [intercept + slope*i for i in x]
plt.plot(x,lsq, 'r')

n = len(x)
# pick a sufficient number of iterations
# OR check the error to make sure it is small enough
# be careful though... if you want it to be too small
# your method may never converge
# I suggest checking that it is smaller than 1e-2
maxiter = 

# initialize your slope and intercept
m = np.random.randn(1); bhat = np.random.randn(1)

# choose your learning rate... this can also highly influence
# your method... if you choose too big, your values can blow up!
# if you choose too small, you may need more iterations
gam = 

# set aside the original values so you can compare how far you've come
m0 = m; b0 = bhat;

# create a place to save errors in case you'd like to create an error plot
error = [0 for i in range(maxiter)]
# creates a list of ones
ones = [1 for i in range(n)]

# gradient descent method
for i in range(maxiter): 

    #what is the formula for the difference between y and lsq line
    cost = 
    # what is the partial of F(m,b) wrt m
    nabla_m = 
    # what is the partial of F(m,b) wrt b
    nabla_b = 
  
    # what is the gradient descent formula to find m
    m = 
    # what is the gradient descent formula to find b
    bhat = 
  
    # In case you'd like to plot the error values, you can save each value after an iteration
    error[i] = np.linalg.norm(y - (m*x + bhat*ones), ord = 2)

# calculate your estimated values and plot them
# use the code at the top to help you figure this out.
estimate = 
plt.plot(x,estimate, 'g')
plt.show()