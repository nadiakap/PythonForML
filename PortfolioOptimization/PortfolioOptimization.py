import pandas as pd
#import pandas_datareader as web

import numpy as np
import scipy.optimize as sco

def min_func_drawdown(weights):
    return statistics(weights)[3]

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
    Portfolio drawdown over entire period
    '''
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 52
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 52, weights)))

    pvalue = np.sum(df1 * weights,axis=1)
    max_value=np.max(pvalue)
    ind_max = pvalue.values.argmax()
    min_value = np.min(pvalue[ind_max:])
    pddr = 1-min_value/max_value  
    return np.array([pret, pvol, pret / pvol,pddr])

symbols = ['SPY','BTC','AAPL', 'MSFT', 'IBM',  'GLDI', 'V']
filename='myportfolio.csv'
noa = len(symbols)

data=pd.read_csv(filename)
df1 = data[['SPY','BTC','AAPL','MSFT', 'IBM', 'GLDI', 'V']]

#rets = np.log(df1 / df1.shift(1))
rets = ((df1 - df1.shift(1))/ df1)
#print(rets)
weights = np.random.random(noa)
weights /= np.sum(weights)

#constraint - all parametrs add up to 1
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 0.5) for x in range(noa))
opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP',
   bounds=bnds, constraints=cons)
print('Portfolio Instruments')
print('SPY','BTC','AAPL','MSFT', 'IBM', 'GLDI', 'V')
print('***************Maximum Sharpe Ratio portfolio*******')
print('weights =',opts['x'].round(3))
print('return',(statistics(opts['x'])[0]).round(3))
print('variance ',(statistics(opts['x'])[1]**2).round(3))
print('Sharpe ratio = ',-opts['fun'])
print('max drawdown ',statistics(opts['x'])[3].round(3))
#minimize portfoliovariance
optv = sco.minimize(min_func_variance, noa * [1. / noa,],
  method='SLSQP', bounds=bnds,
  constraints=cons)
#print(optv)
#absolute minimum variance poerfolio weights

print('*************Absolute Minimum Variance portfolio*********')
print('weights = ',optv['x'].round(3))
print('return',(statistics(optv['x'])[0]).round(3))
print('variance = ',optv['fun'])
print('Sharpe ratio ',statistics(optv['x'])[2].round(3))
print('max drawdown ',statistics(optv['x'])[3].round(3))
#minimize portfoliodrawdown
bnds1 = tuple((0, 0.5) for x in range(noa))
optd = sco.minimize(min_func_drawdown, noa * [1. / noa,], method='SLSQP',
   bounds=bnds1, constraints=cons)

print('******Minimized Max Drawdown portfolio****************')
print('weights = ',optd['x'].round(3))
print('return',(statistics(optd['x'])[0]).round(3))
print('variance ',(statistics(optd['x'])[1]**2).round(3))
print('Sharpe ratio ',statistics(optd['x'])[2].round(3))
print('max drawdown = ',optd['fun'])