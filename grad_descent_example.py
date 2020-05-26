#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:33:47 2020

@author: morleycoulson
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression


def compute_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_log_loss(x, y):
    assert len(x) == len(y)
    N = len(x)
    loss = (1 / N) * np.sum((-y * np.log(x) - (1 - y)*np.log(1 - x)))
    return loss
    

def fast_auc(actual, predicted, approx = False):
    if approx: r = np.argsort(predicted)
    else: r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(r[actual==1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)


class gradient_descent_optimiser():
    def __init__(self, X, y, loss_fun = None, eta = 0.01, tol = 1e-5):
        self.X = X
        self.y = y
        self.N = len(y)
        self.eta = eta
        self.loss_fun = loss_fun
        self.history = []
        self.params = np.random.rand(X.shape[1])
        self.tol = tol
        self.max_iter = 1000
        self._X_theta = None
        self._Px = None
        
    def X_param_prod(self):
        return np.dot(self.X, self.params)
            
    def compute_loss(self):
        if self.loss_fun:
            loss = self.loss_fun(self.X, self.y, self.params)
        else:
            X_sigmoid = compute_sigmoid(self._X_theta)
            loss = compute_log_loss(X_sigmoid, self.y)
        return loss
    
    def compute_gradient(self):
        return (1/self.N) * np.dot(self.X.T, self._Px - y)
    
    def update_params(self):
        return self.params - self.eta * self.compute_gradient()
    
    def gradient_descent(self):
        i = 0
        diff = 1
        prev_loss = 0
        while i < self.max_iter and diff > self.tol:
            #step the parameters
            self._X_theta = self.X_param_prod()
            self._Px = compute_sigmoid(self._X_theta)
            
            self.params = self.update_params()
            current_loss = self.compute_loss()
            print('iteration : {}; loss : {}'.format(i, round(current_loss, 4)))
            self.history.append(current_loss)
            #update the iterators
            i+=1
            diff = np.abs(current_loss - prev_loss)
            prev_loss = current_loss
        if i < self.max_iter:
            print('solution converged within max iterations')
        else:
            print('solution did not converge in max iterations')
            
    def predict(self, X):
        exponent = np.matmul(X, self.params)
        return compute_sigmoid(exponent)
            
    
def fit_sklearn_log_reg(X,y):
    model = LogisticRegression(
        penalty = 'none',
        fit_intercept = False,
    )
    model.fit(X, y)
    return model
    
    
if __name__ == '__main__':
    #process the data
    df = pd.read_csv('fake_data.csv')
    X = df.iloc[:, 0:11].to_numpy()
    y = df.iloc[:, 11].to_numpy()
    
    #run logistic regression
    log_reg_optimiser = gradient_descent_optimiser(X = X, y = y)
    log_reg_optimiser.gradient_descent()
    
    log_reg_sklearn = fit_sklearn_log_reg(X, y)
    
    #evaluate performance
    print(log_reg_optimiser.params)
    
    preds = log_reg_optimiser.predict(X)
    print(fast_auc(actual = y, predicted = preds))
    
    preds_sklearn = log_reg_sklearn.predict(X)
    print(fast_auc(actual = y, predicted = preds_sklearn))
    
