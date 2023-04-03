import pandas as pd
import numpy as np
import itertools as it
import matplotlib.pyplot as pp

def drawdown(w):
    w = np.array(w)
    pvalue = np.sum(prices * w,axis=1)
    max_value=np.max(pvalue)
    ind_max = pvalue.values.argmax()
    min_value = np.min(pvalue[0:ind_max+1])
    return 1-min_value/max_value  

def portfolio_value(w):
    w = np.array(w)
    return  np.sum(prices * w,axis=1)
   
#read stock quotes from excel file
sdata = pd.read_csv('myportfolio.csv',parse_dates=True, index_col=0)
sdata = sdata.sort_values('Date',ascending=False)
prices = sdata[['AAPL', 'MSFT', 'IBM', 'GLDI', 'V','BTC','SPY']]
prices=prices[0:52]
noa = prices.shape[1]
res=[]
mesh=[0.1,0.2,0.3,0.4,0.05,0.25]
rng=it.product(mesh,repeat=len(mesh))
for elem in list(rng):  
    wght = list(elem)
    wght.append(1-sum(elem))  
    wght = np.array(wght)
    if round(sum(wght),2)==1.00 and wght[-1]>0.01:
        d=drawdown(wght)
        pvalue=portfolio_value(wght)
        #pret = portfolio_return(wght)
        res.append((d,wght,pvalue))
res = sorted(res)   
  

print('portfolio value as of May 12, 2020:',res[0][2][0])
print('portfolio maximum drawdown and weights of components')
print('drawdown:', res[0][0]*100,'%')
print('AAPL:',res[0][1][0]*100,'%') 
print('MSFT:',res[0][1][1]*100,'%')
print('IBM:',res[0][1][2]*100,'%') 
print('GLDI:',res[0][1][3]*100,'%') 
print('V:',res[0][1][4]*100,'%')
print('BIC:',res[0][1][5]*100,'%') 
print('SPY:',res[0][1][6]*100,'%') 

#plot max drawdown for individual instruments of the portfolio
sdatapart = sdata[0:52]
roll_max_V = sdatapart['V'].rolling(252, min_periods=2, center=False, win_type=None, on=None, axis=0, closed=None).max()
daily_drawdown_V = sdatapart['V']/roll_max_V - 1.0
max_daily_drawdown_V = daily_drawdown_V.rolling(52, min_periods=1).min()
#daily_drawdown_V.plot()
max_daily_drawdown_V.plot()
pp.title('V: maximum drawdown')
pp.legend()
pp.show()
roll_max_A = sdatapart['AAPL'].rolling(52, min_periods=2, center=False, win_type=None, on=None, axis=0, closed=None).max()
daily_drawdown_A = sdatapart['AAPL']/roll_max_A - 1.0
max_daily_drawdown_A = daily_drawdown_A.rolling(52, min_periods=1).min()

#daily_drawdown_A.plot()
max_daily_drawdown_A.plot()
pp.title('AAPL:maximum drawdown')
pp.legend()
pp.show()
roll_max_M = sdatapart['MSFT'].rolling(52, min_periods=2, center=False, win_type=None, on=None, axis=0, closed=None).max()
daily_drawdown_M = sdatapart['MSFT']/roll_max_M - 1.0
max_daily_drawdown_M = daily_drawdown_M.rolling(52, min_periods=1).min()
#daily_drawdown_M.plot()
max_daily_drawdown_M.plot()
pp.title('MSFT:maximum drawdown')
pp.legend()
pp.show()
roll_max_I = sdatapart['IBM'].rolling(52, min_periods=2, center=False, win_type=None, on=None, axis=0, closed=None).max()
daily_drawdown_I = sdatapart['IBM']/roll_max_I - 1.0
max_daily_drawdown_I = daily_drawdown_I.rolling(52, min_periods=1).min()
#daily_drawdown_I.plot()
max_daily_drawdown_I.plot()
pp.title('IBM:maximum drawdown')
pp.legend()
pp.show()
roll_max_B = sdatapart['BTC'].rolling(52, min_periods=2, center=False, win_type=None, on=None, axis=0, closed=None).max()
daily_drawdown_B = sdatapart['BTC']/roll_max_B - 1.0
max_daily_drawdown_B = daily_drawdown_B.rolling(52, min_periods=1).min()
#daily_drawdown_B.plot()
max_daily_drawdown_B.plot()
pp.title('BTC:maximum drawdown over')
pp.legend()
pp.show()
roll_max_S = sdatapart['SPY'].rolling(52, min_periods=2, center=False, win_type=None, on=None, axis=0, closed=None).max()
daily_drawdown_S = sdatapart['SPY']/roll_max_S - 1.0
max_daily_drawdown_S = daily_drawdown_S.rolling(52, min_periods=1).min()
#daily_drawdown_S.plot()
max_daily_drawdown_S.plot()
pp.title('SPY:maximum drawdown')
pp.legend()
pp.show()
roll_max_G = sdatapart['GLDI'].rolling(52, min_periods=2, center=False, win_type=None, on=None, axis=0, closed=None).max()
daily_drawdown_G = sdatapart['GLDI']/roll_max_G - 1.0
max_daily_drawdown_G = daily_drawdown_G.rolling(52, min_periods=1).min()
#daily_drawdown_S.plot()
max_daily_drawdown_G.plot()
pp.title('GLDI:maximum drawdown')
pp.legend()
pp.show()
