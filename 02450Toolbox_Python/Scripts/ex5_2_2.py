# exercise 5.2.2

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show
import sklearn.linear_model as lm
import numpy as np

# Use dataset as in the previous exercise
N = 100
X = np.array(range(N)).reshape(-1,1)
eps_mean, eps_std = 0, 0.05
eps = np.array(eps_std*np.random.randn(N) + eps_mean).reshape(-1,1)
w0 = -0.5
w1 = 0.01
y = w0 + w1*X + eps
y_true = y - eps

# Fit ordinary least squares regression model
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(X,y) # fit -> estimate parameters
# Compute model output:
y_est = model.predict(X) # predict -> to predict values for new data points
# Or equivalently:
#y_est = model.intercept_ + X @ model.coef_

print("Score = ",model.score(X,y))
print("Estimated parameters:")
print("\tw0 = {:.5}\n\tw1 = {:.5}".format(model.intercept_[0], model.coef_[0][0]))

# Plot original data and the model output
f = figure()

plot(X,y,'.')
plot(X,y_true,'-')
plot(X,y_est,'-')
xlabel('X'); ylabel('y')
legend(['Training data', 'Data generator', 'Regression fit (model)'])

show()

print('Ran Exercise 5.2.2')