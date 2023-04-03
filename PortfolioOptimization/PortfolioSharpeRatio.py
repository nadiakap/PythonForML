import pandas as pd
#import pandas_datareader as web

import numpy as np


def min_func_sharpe(weights):
    return -statistics(weights)[2]

def min_func_variance(weights):
    return statistics(weights)[1] ** 2

def statistics(weights):
    '''
    Returns portfolio statistics.
    Parameters
    ==========
    weights : array-like
    weights for different securities in portfolio
    Returns
    =======
    pret : float
    expected portfolio return
    pvol : float
    expected portfolio volatility
    pret / pvol : float
    Sharpe ratio for rf=0
    '''
    filename='stockdata.csv'
    data=pd.read_csv(filename)
    df1 = data[['AAPL', 'MSFT', 'IBM', 'AABA', 'GLDI']]

    #rets = np.log(df1 / df1.shift(1))
    rets = ((df1 - df1.shift(1))/ df1)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])

if __name__ =="__main__":
    weights=np.array([0.1,0.2,0.2,0.3,0.3])
    print(min_func_sharpe(weights))

