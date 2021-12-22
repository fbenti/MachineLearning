import numpy as np
from sklearn import model_selection
import source.Data as Da
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid, figure, plot, xlabel, ylabel, legend, ylim, show)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import mcnemar
import pandas as pd


def logistic_regression(data: Da.Data, k_folds: int):
    X = data.x2[:, range(4, 12)]
    y = data.y.squeeze()
    classNames = data.class_dict
    N = data.N
    M = data.M
    C = len(classNames)

    K = k_folds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.9, stratify=y)

    mu = np.mean(X, 0)
    sigma = np.std(X, 0)

    X_train = (X_train - mu)
    X_train = X_train / sigma
    X_test = (X_test - mu) / sigma

    lambda_interval = np.logspace(0, 3, 50)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    for k in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty='l2', C=1 / lambda_interval[k])

        mdl.fit(X_train, y_train)

        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T

        train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

        w_est = mdl.coef_[0]
        coefficient_norm[k] = np.sqrt(np.sum(w_est ** 2))

    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]

    print(opt_lambda)

    plt.figure(figsize=(8, 8))
    plt.semilogx(lambda_interval, train_error_rate * 100)
    plt.semilogx(lambda_interval, test_error_rate * 100)
    plt.semilogx(opt_lambda, min_error * 100, 'o')
    plt.text(1e-3, 3, "Minimum test error: " + str(np.round(min_error * 100, 2)) + ' % at 1e' + str(
        np.round(np.log10(opt_lambda), 2)))
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.ylabel('Error rate (%)')
    plt.title('Classification error')
    plt.legend(['Training error', 'Test error', 'Test minimum'], loc='upper right')
    plt.ylim([0, 60])
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.semilogx(lambda_interval, coefficient_norm, 'k')
    plt.ylabel('L2 Norm')
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.title('Parameter vector L2 norm')
    plt.grid()
    plt.show()


def k_nearest_neighbours(data: Da.Data, k_folds, max_neighbours: int):
    X = data.x2[:, range(4, 12)]
    y = data.y

    CV = model_selection.KFold(k_folds, shuffle=True)
    errors = np.zeros((k_folds, max_neighbours))
    i = 0
    test_size = 0
    for train_index, test_index in CV.split(X, y):
        print('Crossvalidation fold: {0}/{1}'.format(i + 1, k_folds))
        test_size += test_index.size

        # extract training and test set for current CV fold
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]

        for l in range(1, max_neighbours + 1):
            knclassifier = KNeighborsClassifier(n_neighbors=l)
            knclassifier.fit(X_train, y_train)
            y_est = knclassifier.predict(X_test)
            errors[i, l - 1] = np.sum(y_est[:] != y_test[:])

        i += 1

    # Plot the classification error rate
    figure()
    plt.plot(100*sum(errors, 0)/test_size)
    xlabel('Number of neighbors')
    ylabel('Classification error rate (%)')
    show()


def test(data: Da.Data):
    y = data.y

    c0 = np.sum(y[:] == 0)
    c1 = np.sum(y[:] == 1)

    print('class[0]: {0} - class[1]: {1}'.format(c0, c1))
    print('class[0]%: {0}'.format(100*c0/y.size))


def baseline(data: Da.Data, k_folds: int):
    X = data.x2[:, range(4, 12)]
    y = data.y
    N = data.N

    CV = model_selection.KFold(k_folds, shuffle=True)
    errors = []

    i = 0
    for train_index, test_index in CV.split(X, y):
        print('Crossvalidation fold: {0}/{1}'.format(i + 1, k_folds))
        # extract training and test set for current CV fold
        y_train = y[train_index]
        y_test = y[test_index]

        # fit classifier
        c0 = 0
        c1 = 0
        for c in y_train:
            if c == 0:
                c0 += 1
            else:
                c1 += 1

        if c0 > c1:
            c = 0
        else:
            c = 1

        #test classifier
        err = 0
        for yh in y_test:
            if c != yh:
                err += 1

        errors.append(100 * err / y_test.size)

        i += 1

    print(errors)


def two_layer_cross_validation(data: Da.Data, k_outerfold: int, k_innerfold: int, regularize: bool = False):
    X = data.x2[:, range(4, 12)]
    y = data.y.squeeze()
    N = data.N

    if regularize:
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)
        X = (X - mu) / sigma

    # log_reg complexity parameter (lambda)
    lambda_interval = np.logspace(1, 3, 50)

    # k_nearest complexity parameter (max neighbours)
    max_neighbours = 20

    # statistical evaluation
    yhat = []
    y_true = []

    CV_outer = model_selection.KFold(k_outerfold, shuffle=True)
    CV_inner = model_selection.KFold(k_innerfold, shuffle=True)
    errors = np.zeros((k_outerfold, 3))
    gen_errors = np.zeros((k_outerfold, 3))
    params = np.zeros((k_outerfold, 2))
    outer_test_size = 0
    i = 0
    for outer_train_index, outer_test_index in CV_outer.split(X, y):
        print('outer_fold: {0}/{1}'.format(i + 1, k_outerfold))

        outer_X_train = X[outer_train_index, :]
        outer_y_train = y[outer_train_index]
        outer_X_test = X[outer_test_index, :]
        outer_y_test = y[outer_test_index]

        outer_test_size += outer_test_index.size
        test_size = 0

        # k_nearest need
        k_errors = np.zeros((k_innerfold, max_neighbours))

        # log_reg need
        log_reg_errors = np.zeros((k_innerfold, len(lambda_interval)))

        # statistical evaluation
        dy = []

        inner_i = 0
        for inner_train_index, inner_test_index in CV_inner.split(outer_X_train, outer_y_train):
            inner_X_train = outer_X_train[inner_train_index, :]
            inner_y_train = outer_y_train[inner_train_index]
            inner_X_test = outer_X_train[inner_test_index, :]
            inner_y_test = outer_y_train[inner_test_index]

            test_size += inner_test_index.size

            # region log_reg
            # reg_X_train = inner_X_train[:, range(0, 11)]
            # reg_X_test = inner_X_test[:, range(0, 11)]
            for k in range(0, len(lambda_interval)):
                mdl = LogisticRegression(penalty='l2', C=1 / lambda_interval[k])
                mdl.fit(inner_X_train, inner_y_train)

                y_test_est = mdl.predict(inner_X_test).T
                log_reg_errors[inner_i, k] = np.sum(y_test_est[:] != inner_y_test[:])
            # endregion

            # region k_nearest
            for l in range(1, max_neighbours + 1):
                knclassifier = KNeighborsClassifier(n_neighbors=l)
                knclassifier.fit(inner_X_train, inner_y_train)
                y_est = knclassifier.predict(inner_X_test)
                k_errors[inner_i, l - 1] = np.sum(y_est[:] != inner_y_test[:])
            # endregion

        log_reg_percentages = 100 * sum(log_reg_errors, 0) / test_size
        k_percentages = 100 * sum(k_errors, 0) / test_size

        # region log_reg
        low = log_reg_percentages[0]
        low_index = 0
        inner_i = 0
        for r in log_reg_percentages:
            if r < low:
                low = r
                low_index = inner_i
            inner_i += 1

        # train best model again
        # reg_X_train = outer_X_train[:, range(0, 11)]
        # reg_X_test = outer_X_test[:, range(0, 11)]
        mdl = LogisticRegression(penalty='l2', C=1 / lambda_interval[low_index])
        mdl.fit(outer_X_train, outer_y_train)

        y_test_est = mdl.predict(outer_X_test).T
        dy.append(y_test_est)
        errors[i, 0] = 100 * np.sum(y_test_est[:] != outer_y_test[:]) / outer_y_test.size
        gen_errors[i, 0] = errors[i, 0] * outer_y_test.size
        params[i, 0] = lambda_interval[low_index]
        # endregion

        # region k_nearest
        low = k_percentages[0]
        low_index = 1
        inner_i = 1
        for r in k_percentages:
            if r < low:
                low = r
                low_index = inner_i
            inner_i += 1

        # train best model again
        knclassifier = KNeighborsClassifier(n_neighbors=low_index)
        knclassifier.fit(outer_X_train, outer_y_train)

        y_est = knclassifier.predict(outer_X_test)
        dy.append(y_est)
        errors[i, 1] = 100 * np.sum(y_est[:] != outer_y_test[:]) / outer_y_test.size
        gen_errors[i, 1] = errors[i, 1] * outer_y_test.size
        params[i, 1] = low_index
        # endregion

        # region baseline
        # fit classifier
        c0 = 0
        c1 = 0
        for c in outer_y_train:
            if c == 0:
                c0 += 1
            else:
                c1 += 1

        if c0 > c1:
            c = 0
        else:
            c = 1

        # test classifier
        y_est = []
        err = 0
        for yh in outer_y_test:
            if c != yh:
                err += 1

        for x in outer_y_test:
            y_est.append(c)

        dy.append(y_est)

        errors[i, 2] = 100 * err / outer_y_test.size
        gen_errors[i, 2] = errors[i, 2] * outer_y_test.size
        # endregion

        dy = np.stack(dy, axis=1)
        yhat.append(dy)
        y_true.append(outer_y_test)

        i += 1

    # print results
    i = 0
    for fold in errors:
        print("Fold: {0}, Log_comp: {1}, Log_err: {2}, K_near_comp: {3}, K_near_err: {4}, baseline_err: {5}".format(i, params[i, 0], fold[0], params[i, 1], fold[1], fold[2]))

        i += 1

    log_reg_general = np.sum(gen_errors[:, 0] / outer_test_size)
    k_near_general = np.sum(gen_errors[:, 1] / outer_test_size)
    baseline_general = np.sum(gen_errors[:, 2] / outer_test_size)
    print("Log_gen_err: {0}, k_near_gen_err: {1}, baseline_gen_err: {2}".format(log_reg_general, k_near_general, baseline_general))

    yhat = np.concatenate(yhat)
    y_true = np.concatenate(y_true)
    alpha = 0.05
    print("theta_K_near - theta_Log")
    [thetahat, CI, p] = mcnemar(y_true, yhat[:, 1], yhat[:, 0], alpha=alpha)
    print("theta ", thetahat)

    print("theta_Baseline - theta_Log")
    [thetahat, CI, p] = mcnemar(y_true, yhat[:, 2], yhat[:, 0], alpha=alpha)
    print("theta ", thetahat)

    print("theta_Baseline - theta_K_near")
    [thetahat, CI, p] = mcnemar(y_true, yhat[:, 2], yhat[:, 1], alpha=alpha)
    print("theta ", thetahat)


def mcnemera(data: Da.Data, k_fold: int, regularize: bool = False):
    X = data.x2[:, range(4, 12)]
    y = data.y.squeeze()
    N = data.N

    if regularize:
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)
        X = (X - mu) / sigma

    # log_reg complexity parameter (lambda)
    lambda_interval = np.logspace(-3, 5, 50)

    # k_nearest complexity parameter (max neighbours)
    max_neighbours = 20

    CV = model_selection.KFold(k_fold, shuffle=True)
    yhat = []
    y_true = []
    i = 0
    for train_index, test_index in CV.split(X, y):
        print('outer_fold: {0}/{1}'.format(i + 1, k_fold))

        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]

        dy = []

        # region log_reg
        reg_X_train = X_train[:, range(0, 11)]
        reg_X_test = X_test[:, range(0, 11)]
        mdl = LogisticRegression(penalty='l2', C=1 / 12.067926406393289)
        mdl.fit(reg_X_train, y_train)

        y_test_est = mdl.predict(reg_X_test).T
        dy.append(y_test_est)
        # endregion

        # region k_near
        knclassifier = KNeighborsClassifier(n_neighbors=10)
        knclassifier.fit(X_train, y_train)

        y_est = knclassifier.predict(X_test)
        dy.append(y_est)
        # endregion

        # region baseline
        c0 = 0
        c1 = 0
        for c in y_train:
            if c == 0:
                c0 += 1
            else:
                c1 += 1

        if c0 > c1:
            c = 0
        else:
            c = 1

        y_est = []
        for x in X_test:
            y_est.append(c)

        dy.append(y_est)
        # endregion

        dy = np.stack(dy, axis=1)
        yhat.append(dy)
        y_true.append(y_test)

        i += 1

    yhat = np.concatenate(yhat)
    y_true = np.concatenate(y_true)

    alpha = 0.05
    [thetahat, CI, p] = mcnemar(y_true, yhat[:, 1], yhat[:, 0], alpha=alpha)

    print("theta = theta_K_near - theta_Log  point estimate", thetahat, " CI: ", CI, "p-value", p)

    [thetahat, CI, p] = mcnemar(y_true, yhat[:, 2], yhat[:, 0], alpha=alpha)

    print("theta = theta_Baseline - theta_Log point estimate", thetahat, " CI: ", CI, "p-value", p)

    [thetahat, CI, p] = mcnemar(y_true, yhat[:, 2], yhat[:, 1], alpha=alpha)

    print("theta = theta_Baseline - theta_K_near point estimate", thetahat, " CI: ", CI, "p-value", p)


def train_log_model(data: Da.Data, lambda_val: float, regularize: bool = False):
    X = data.x2[:, range(4, 12)]
    y = data.y.squeeze()

    if regularize:
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)
        X = (X - mu) / sigma

    mdl = LogisticRegression(penalty='l2', C=1 / lambda_val)
    mdl.fit(X, y)

    print(np.round(mdl.coef_, 2))

    Weights = pd.DataFrame(np.round(mdl.coef_, 2))
    #Weights['Features'] = pd.Series(['X', 'Y', 'Month', 'Day', 'FFMC', 'DMC', 'DC', 'ISI',
    #                                 'temp', 'RH', 'wind', 'rain'])
    Weights_opt = Weights.stack()

    #Weights_opt.drop(index=0, inplace=True)
    ax = Weights_opt.plot(kind='bar', figsize=(15, 10), legend=False, fontsize=25)
    ax.set_xticklabels(['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain'])
    plt.xticks(rotation=-30)
    plt.ylabel('Weights', fontsize=25)
    plt.show()
