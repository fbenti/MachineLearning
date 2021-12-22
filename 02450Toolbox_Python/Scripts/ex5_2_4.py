# exercise 5.2.4
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm

# requires wine data from exercise 5.1.5
from ex5_1_5 import *

# Split dataset into features and target vector
alcohol_idx = attributeNames.index('Alcohol')
y = X[:,alcohol_idx]

X_cols = list(range(0,alcohol_idx)) + list(range(alcohol_idx+1,len(attributeNames)))
X = X[:,X_cols]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

print("w0 = ",model.intercept_)
print("\n",attributeNames)
print("\n",model.coef_)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot

# In the plot, the first graph shows the estimation of the alcohol content 
# (thanks to our model) against the true content (the one found in the actual 
# dataset). You can compare them and see that the perfect regression would lead 
# to a y = x curve. The second plot shows how the residuals are distributed. 
# The residuals are defined as the difference between what we estimated and the 
# true value of y.
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Alcohol content (true)'); ylabel('Alcohol content (estimated)');
subplot(2,1,2)
hist(residual,40)

show()

print('Ran Exercise 5.2.4')