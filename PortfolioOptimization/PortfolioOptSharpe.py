import pandas as pd
#import pandas_datareader as web

import numpy as np
import scipy.optimize as sco

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
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])

symbols = ['AAPL', 'MSFT', 'IBM',  'GLDI', 'V']
filename='DowJonesdata.csv'
noa = len(symbols)

data=pd.read_csv(filename)
df1 = data[['AAPL', 'MSFT', 'IBM', 'GLDI', 'V']]

#rets = np.log(df1 / df1.shift(1))
rets = ((df1 - df1.shift(1))/ df1)
#print(rets)
weights = np.random.random(noa)
weights /= np.sum(weights)

#constraint - all parametrs add up to 1
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP',
   bounds=bnds, constraints=cons)

print('optimized sharpe ratio = ',-opts['fun'])
print('and weights of the optimized portfolio = ',opts['x'].round(3))
#expected return,volatility andSarpe ratioprint(statistics(opts['x']).round(3))
#minimize portfoliovariance
optv = sco.minimize(min_func_variance, noa * [1. / noa,],
  method='SLSQP', bounds=bnds,
  constraints=cons)
#print(optv)
#absolute minimum variance poerfolio weights
print('absolute minimum variance  ',optv['fun'])
print('absolute minimum variance porfolio weights ',optv['x'].round(3))

#print(statistics(optv['x']).round(3))

