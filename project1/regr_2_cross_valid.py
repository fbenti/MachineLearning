#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========================================================================================================================================
# Importing packages
# ========================================================================================================================================
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
import seaborn as sns
import cufflinks as cf
import plotly as py
import torch
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
py.offline.init_notebook_mode(connected = True)
cf.go_offline()
sns.set()
import sklearn
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from toolbox_02450 import rlr_validate
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary, correlated_ttest, rlr_validate
from datetime import datetime

import source.Data as Da
# ========================================================================================================================================
# Part 1B
# ========================================================================================================================================

# ========================================================================================================================================
# Separating the data in X and y format. Also applying K-fold cross validation
# ========================================================================================================================================
data = Da.Data()
X = data.df
X = X.drop('area',axis=1)
X = np.array(X)
y = data.df['area']
yTrans = np.log(y+1)
y = yTrans    
y = [np.array(y)]
y = np.transpose(y)
N, M = X.shape


# random_seed = 1
X_tilda = X - np.ones((N,1))*X.mean(axis=0)
X_tilda = X_tilda*(1/np.std(X_tilda,0))
X = X_tilda


print_cv_inner_loop_text = True
print_cv_outer_loop_text = True

apply_baseline = True
apply_ANN      = True 
apply_linear   = True

# apply_setup_ii = True

min_n_hidden_units = 1 # The minimum number of hidden units
max_n_hidden_units = 1 # The maximum number of hidden units

lambda_interval = range(164,168)

# ANN options
loss_fn   = torch.nn.MSELoss()
max_iter  = 10000
n_rep_ann = 1

# Set K-folded CV options 
random_seed = 3
K_1   = 10 # Number of outer loops
K_2   = 10 # Number of inner loops
CV_1 = model_selection.KFold(n_splits=K_1 , shuffle=True, random_state=random_seed)
CV_2 = model_selection.KFold(n_splits=K_2, shuffle=True, random_state=random_seed)
# CV_setup_ii = sklearn.model_selection.KFold(n_splits=K_1,shuffle=True, random_state = random_seed + 1) 

## Define holders for outer CV results
test_error_outer_baseline                = [] 
test_error_outer_linear                  = [] 
test_errors_outer_ANN                    = []
data_outer_test_length                   = []
optimal_regularization_param_baseline    = []
optimal_regularization_param_linear      = []
optimal_regularization_param_ANN         = []

# Outer loop
k_outer = 0
for train_outer_index, test_outer_index in CV_1.split(X):
    if(print_cv_outer_loop_text):
        print('Computing CV outer fold: {0}/{1}..'.format(k_outer+1,K_1))
        
    X_train_outer, y_train_outer = X[train_outer_index,:], y[train_outer_index]
    X_test_outer, y_test_outer = X[test_outer_index,:], y[test_outer_index]
    
    if (apply_ANN):
        X_train_outer_tensor = torch.tensor(X[train_outer_index,:], dtype=torch.float)
        y_train_outer_tensor = torch.tensor(y[train_outer_index], dtype=torch.float)
        X_test_outer_tensor  = torch.tensor(X[test_outer_index,:], dtype=torch.float)
        y_test_outer_tensor  = torch.tensor(y[test_outer_index], dtype=torch.uint8)
    
    # Save length of outer train and test data
    data_outer_train_length    = float(len(y_train_outer))
    data_outer_test_length_tmp = float(len(y_test_outer))
    
    # Define holders for inner CV results
    best_inner_model_baseline      = []
    error_inner_baseline           = [] 
    data_validation_length         = [] 
    
    validation_errors_inner_ANN_matrix         = np.array(np.ones(max_n_hidden_units - min_n_hidden_units + 1)) 
    validation_errors_inner_linear_matrix      = np.array(np.ones(len(lambda_interval))) 
    hidden_units_matrix                        = np.array(np.ones(max_n_hidden_units - min_n_hidden_units + 1))  
    regularization_param_linear_matrix         = np.array(np.ones(len(lambda_interval)))
    
    # Inner loop
    k_inner=0
    for train_inner_index, test_inner_index in CV_2.split(X_train_outer):
        if(print_cv_inner_loop_text):
            print('Computing CV inner fold: {0}/{1}..'.format(k_inner+1,K_2))
    
        # Extract training and test set for current CV fold
        X_train_inner, y_train_inner = X[train_inner_index,:], y[train_inner_index]
        X_test_inner, y_test_inner = X[test_inner_index,:], y[test_inner_index]        
                  
        if (apply_ANN):
            X_train_inner_tensor = torch.tensor(X[train_inner_index,:], dtype=torch.float)
            y_train_inner_tensor = torch.tensor(y[train_inner_index], dtype=torch.float)
            X_test_inner_tensor = torch.tensor(X[test_inner_index,:], dtype=torch.float)
            y_test_inner_tensor = torch.tensor(y[test_inner_index], dtype=torch.uint8)
        
        
        """INNER - BASELINE"""
        # 'Fit' baseline model (simply the unconditional mean value of y)
        mean_y                        = np.mean(y_train_inner)
        y_est_test_inner_baseline     = mean_y
        # Calculate validation error over inner test data
        validation_errors_inner_baseline = np.sum((y_est_test_inner_baseline - y_test_inner)**2) / float(len(y_test_inner)) 
        ## Store accuracy of CV-loop
        error_inner_baseline.append(validation_errors_inner_baseline)
        
        ## Store data validation length
        data_validation_length.append(float(len(y_test_inner)))

        if (apply_linear):
            validation_errors_inner_linear  = []
            regularization_param_linear     = []
            
             
            for lambda_val in lambda_interval:
                model = sklearn.linear_model.Ridge(alpha=lambda_val)
                model = model.fit(X_train_inner,y_train_inner)
                y_est_test_inner_linear = model.predict(X_test_inner)
                 
                error      = (y_est_test_inner_linear - y_test_inner)**2
                error_rate =  np.sum(error) / len(y_test_inner)
                validation_errors_inner_linear.append(error_rate)
                regularization_param_linear.append(lambda_val)
                
            validation_errors_inner_linear        = np.array(validation_errors_inner_linear)
            validation_errors_inner_linear_matrix = np.vstack((validation_errors_inner_linear_matrix,validation_errors_inner_linear))
            regularization_param_linear_matrix    = np.vstack((regularization_param_linear_matrix,regularization_param_linear))     
                
        # Estimate ANN if apply_ANN is true
        if (apply_ANN):
            validation_errors_inner_ANN  = []
            hidden_unit_applied          = []
            for n_hidden_units in range(min_n_hidden_units,max_n_hidden_units + 1):
                model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), 
                    torch.nn.Tanh(),   
                    torch.nn.Linear(n_hidden_units, 1), 
                    )
                """
                Returns:
                    A list of three elements:
                        best_net:       A trained torch.nn.Sequential that had the lowest 
                                        loss of the trained replicates
                        final_loss:     An float specifying the loss of best performing net
                        learning_curve: A list containing the learning curve of the best net.
                """
                net, final_loss, learning_curve = train_neural_net(model,
                                                    loss_fn,
                                                    X=X_train_inner_tensor,
                                                    y=y_train_inner_tensor,
                                                    n_replicates=n_rep_ann,
                                                    max_iter=max_iter)
                
                # Determine estimated class labels for test set
                y_est_inner   = net(X_test_inner_tensor) 
                
                # Determine errors and error rate
                e = (y_est_inner.float()-y_test_inner_tensor.float())**2
                error_rate = (sum(e).type(torch.float)/len(y_test_inner_tensor)).data.numpy()[0]
                validation_errors_inner_ANN.append(error_rate)
                
                ## Add applied hidden units to array
                hidden_unit_applied.append(n_hidden_units)
                
            validation_errors_inner_ANN        = np.array(validation_errors_inner_ANN)
            validation_errors_inner_ANN_matrix = np.vstack((validation_errors_inner_ANN_matrix,validation_errors_inner_ANN))
            hidden_units_matrix                = np.vstack((hidden_units_matrix,hidden_unit_applied))

        k_inner+=1
    
    """Outer-Baseline"""
    # 'Fit' baseline model 
    mean_y                        = np.mean(y_train_outer)
    y_est_test_outer_baseline     = mean_y             
    # Estimate the test error (best model from inner fitted on the outer data)
    test_error_outer_baseline_tmp = np.sum((y_est_test_outer_baseline - y_test_outer)**2) / float(len(y_test_outer))
    test_error_outer_baseline.append(test_error_outer_baseline_tmp)
    # Calculate validation error over inner test data
    validation_errors_inner_baseline = np.sum((y_est_test_inner_baseline - y_test_inner)**2) / float(len(y_test_inner)) 

    # Add length of outer test data
    data_outer_test_length.append(data_outer_test_length_tmp)
    
    # Find optimal model of ANN (if apply_ANN is true)
    if (apply_ANN):        
        validation_errors_inner_ANN_matrix = np.delete(validation_errors_inner_ANN_matrix,0,0)
        hidden_units_matrix                = np.delete(hidden_units_matrix,0,0)
        validation_errors_inner_ANN_matrix = np.transpose(validation_errors_inner_ANN_matrix)  
        estimated_inner_test_error_ANN_models = []
        for s in range(0,len(validation_errors_inner_ANN_matrix)):
            # tmp_inner_test_error = np.sum(np.multiply(data_validation_length,validation_errors_inner_ANN_matrix[s])) / data_outer_train_length
            tmp_inner_test_error = np.sum(np.multiply(data_validation_length,validation_errors_inner_ANN_matrix[s])) / data_outer_train_length

            estimated_inner_test_error_ANN_models.append(tmp_inner_test_error)
        
        # Saves the regularization parameter for the best performing ANN model
        lowest_est_inner_error_ANN_models = min(estimated_inner_test_error_ANN_models)
        index_tmp                         = (list(estimated_inner_test_error_ANN_models).index(lowest_est_inner_error_ANN_models))        
        optimal_regularization_param_ANN.append(hidden_units_matrix[k_outer][index_tmp])
        
        # Estimates the test error on outer test data
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, optimal_regularization_param_ANN[k_outer].astype(int)), 
            torch.nn.Tanh(),   
            torch.nn.Linear(optimal_regularization_param_ANN[k_outer].astype(int), 1), 
            )
        # Run optimization
        net, final_loss, learning_curve = train_neural_net(model,
                                            loss_fn,
                                            X=X_train_outer_tensor,
                                            y=y_train_outer_tensor,
                                            n_replicates=n_rep_ann,
                                            max_iter=max_iter)
        y_est_outer_ANN = net(X_test_outer_tensor)
        
        # Determine errors and error rate
        e = (y_est_outer_ANN.float()-y_test_outer_tensor.float())**2
        error_rate = (sum(e).type(torch.float)/len(y_test_outer_tensor)).data.numpy()[0]
        test_errors_outer_ANN.append(error_rate)


    if (apply_linear):
        validation_errors_inner_linear_matrix = np.delete(validation_errors_inner_linear_matrix,0,0) 
        validation_errors_inner_linear_matrix   = np.transpose(validation_errors_inner_linear_matrix)
        estimated_inner_test_error_linear_models = []
        for s in range(0,len(validation_errors_inner_linear_matrix)):
            tmp_inner_test_error = np.sum(np.multiply(data_validation_length,validation_errors_inner_linear_matrix[s])) / data_outer_train_length
            estimated_inner_test_error_linear_models.append(tmp_inner_test_error)
  
        lowest_est_inner_error_linear_models = min(estimated_inner_test_error_linear_models)
        index_lambda = list(estimated_inner_test_error_linear_models).index(lowest_est_inner_error_linear_models)
        optimal_regularization_param_linear.append(lambda_interval[index_lambda])         
  
        # Standardize outer fold based on training set
        mu = np.mean(X_train_outer[:, 1:], 0)
        sigma = np.std(X_train_outer[:, 1:], 0)
        X_train_outer[:, 1:] = (X_train_outer[:, 1:] - mu) / sigma
        X_test_outer[:, 1:] = (X_test_outer[:, 1:] - mu) / sigma
        
        model = sklearn.linear_model.Ridge(alpha=optimal_regularization_param_linear[k_outer])
        model = model.fit(X_train_outer,y_train_outer)
        y_est_test_outer_linear = model.predict(X_test_outer)
       
        error      = (y_est_test_outer_linear - y_test_outer)**2
        error_rate =  np.sum(error) / len(y_test_outer)
        test_error_outer_linear.append(error_rate)  

    k_outer+=1
    
generalization_error_baseline_model = np.sum(np.multiply(test_error_outer_baseline,data_outer_test_length)) * (1/N) 
print('est gen error of baseline model: ' +str(round(generalization_error_baseline_model, ndigits=3)))  
if (apply_ANN):
    generalization_error_ANN_model = np.sum(np.multiply(test_errors_outer_ANN,data_outer_test_length)) * (1/N)
    print('est gen error of ANN model: ' +str(round(generalization_error_ANN_model, ndigits=3)))    

if (apply_linear):
    generalization_error_linear_model = np.sum(np.multiply(test_error_outer_linear,data_outer_test_length)) * (1/N)
    print('est gen error of linear model: ' +str(round(generalization_error_linear_model, ndigits=3)))



## Create output table as dataframe
n_of_cols                  = sum([apply_ANN,apply_linear])*2 + 2   
n_of_index                 = K_1 + 1 
df_output_table            = pd.DataFrame(np.ones((n_of_index,n_of_cols)),index=range(1,n_of_index + 1))
df_output_table.index.name = "Outer fold"
   
    
if(apply_ANN):
    df_output_table.columns                = ['test_data_size','n_hidden_units','ANN_test_error','lambda','Linear_test_error','baseline_test_error']
    optimal_regularization_param_ANN.append('')
    optimal_regularization_param_linear.append('')
    data_outer_test_length.append('')
    col_2                                  = list(np.array(test_errors_outer_ANN).round(3)*100)
    col_2.append(round(generalization_error_ANN_model*100,ndigits=1))
    col_4                                  = list(np.array(test_error_outer_linear).round(3)*100)
    col_4.append(round(generalization_error_linear_model*100,ndigits=1))    
    col_5                                  = list(np.array(test_error_outer_baseline).round(3)*100)
    col_5.append(round(generalization_error_baseline_model*100,ndigits=1))       
        
    ## Add values to columns in output table    
    df_output_table['test_data_size']      = data_outer_test_length
    df_output_table['n_hidden_units']      = optimal_regularization_param_ANN
    df_output_table['ANN_test_error']      = col_2
    df_output_table['lambda']              = optimal_regularization_param_linear
    df_output_table['Linear_test_error']   = col_4
    df_output_table['baseline_test_error'] = col_5
## Export as csv
df_output_table.to_csv('final_cross.csv')


   
# Stats evaluation
from toolbox_02450 import ttest_twomodels
    
# Should use the same D_par train set and D_test for stats
CV_setup_I = CV_1

# Loss function for the stats    
loss_in_r_function = 2 

alpha_t_test            = 0.05
rho_t_test              = 1/K_1

y_true = []
yhat_ANN = []
yhat_baseline = []
yhat_linear = []

k = 0
most_common_lambda = stats.mode(optimal_regularization_param_linear).mode[0].astype('float64')
for train_index,test_index in CV_setup_I.split(X):
    print('Computing setup I CV K-fold: {0}/{1}..'.format(k+1,K_1))
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]
    
    X_train_tensor = torch.tensor(X[train_index,:], dtype=torch.float)
    y_train_tensor = torch.tensor(y[train_index], dtype=torch.float)
    X_test_tensor = torch.tensor(X[test_index,:], dtype=torch.float)
    y_test_tensor = torch.tensor(y[test_index], dtype=torch.uint8)
    
    # Train the 3 models on the same TRAIN data set
    
    # Train baseline
    model_baseline = np.mean(y_train)      
    # Train linear-regression
    model_linear = sklearn.linear_model.Ridge(alpha=most_common_lambda).fit(X_train,y_train.squeeze())
    # Train ANN
    most_common_regu_ANN  = 1
    model_second = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, most_common_regu_ANN), #M features to H hidden units
                            # 1st transfer function, either Tanh or ReLU:
                            #torch.nn.ReLU(), 
                            torch.nn.Tanh(),   
                            torch.nn.Linear(most_common_regu_ANN, 1), # H hidden units to 1 output neuron
                            # no final tranfer function, i.e. "linear output"
                            )
    net, final_loss, learning_curve = train_neural_net(model_second,
                                       loss_fn,
                                       X=X_train_tensor,
                                       y=y_train_tensor,
                                       n_replicates=n_rep_ann,
                                       max_iter=max_iter)
    
    
    
    # Train the 3 models on the same TEST data set
    
    # Test baseline
    base_pred = np.ones((y_test.shape[0],1))*model_baseline
    for el in base_pred:
        yhat_baseline.append(el[0])        
    # Test linear
    lin_pred = model_linear.predict(X_test).reshape(-1,1)
    for el in lin_pred:
        yhat_linear.append(el[0])
    # Test ANN
    yhat_ = net(X_test_tensor)
    for el in yhat_:
        yhat_ANN.append(el.detach().numpy()[0])
    for el in y_test:
        y_true.append(el[0])
    
    k += 1
    
    
        
y_true = np.asarray(y_true)
yhat_baseline = np.asarray(yhat_baseline)
yhat_linear = np.asarray(yhat_linear)
yhat_ANN = np.asarray(yhat_ANN)

mean_baseline_vs_linear = - np.mean(np.abs(y_true - yhat_baseline)**2 - np.abs(y_true - yhat_linear) ** 2)
mean_baseline_vs_ANN = - np.mean(np.abs(y_true - yhat_baseline)**2 - np.abs(y_true - yhat_ANN) ** 2)
mean_ANN_vs_linear = - np.mean(np.abs(y_true - yhat_ANN)**2 - np.abs(y_true - yhat_linear) ** 2)

## Baseline vs logistic regression    
mean_setupI_base_vs_linear, CI_setupI_base_vs_linear, p_setupI_base_vs_linear = ttest_twomodels(np.asarray(y_true),yhat_baseline,yhat_linear, loss_norm_p = 2)
## Baseline vs 2nd model    
mean_setupI_base_vs_ANN, CI_setupI_base_vs_ANN, p_setupI_base_vs_ANN = ttest_twomodels(y_true,yhat_baseline,yhat_ANN, loss_norm_p = 2)
## Logistic regression vs 2nd model    
mean_setupI_ANN_vs_linear, CI_setupI_ANN_vs_linear,  p_setupI_ANN_vs_linear = ttest_twomodels(y_true,yhat_ANN,yhat_linear, loss_norm_p = 2)

## Create output table for statistic tests
df_output_table_statistics = pd.DataFrame(np.ones((3,6)), columns = ['H_0','meanZ','p_value','CI_lower','CI_upper','conclusion'])
df_output_table_statistics['H_0'] = ['err_baseline - err_linear','err_baseline - err_ANN=0','err_ANN-err_linear=0']
df_output_table_statistics['meanZ'] = [- mean_setupI_base_vs_linear, - mean_setupI_base_vs_ANN, - mean_setupI_ANN_vs_linear]
df_output_table_statistics['p_value']         = [p_setupI_base_vs_linear,p_setupI_base_vs_ANN,p_setupI_ANN_vs_linear]
df_output_table_statistics['CI_lower']        = [CI_setupI_base_vs_linear[0],CI_setupI_base_vs_ANN[0],CI_setupI_ANN_vs_linear[0]]
df_output_table_statistics['CI_upper']        = [CI_setupI_base_vs_linear[1],CI_setupI_base_vs_ANN[1],CI_setupI_ANN_vs_linear[1]]
rejected_null                                   = (df_output_table_statistics.loc[:,'p_value']<alpha_t_test)
df_output_table_statistics.loc[rejected_null,'conclusion']   = 'H_0 rejected'
df_output_table_statistics.loc[~rejected_null,'conclusion']  = 'H_0 not rejected'
df_output_table_statistics                      = df_output_table_statistics.set_index('H_0')

## Export df as csv
df_output_table_statistics.to_csv('final_stat.csv',encoding='UTF-8')

