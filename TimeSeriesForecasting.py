import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from pandas import datetime

data = pd.read_csv("C:/Users/matth/Downloads/Users Data.csv")



def dateparse(dates):
    return pd.to_datetime(dates, format='%Y%m%d')


data = pd.read_csv('C:/Users/matth/Downloads/Users Data.csv',
                   parse_dates=['DATE'], index_col='DATE', date_parser=dateparse, usecols=['DATE', 'REVENUE'])
print('\n Parsed Data: ')
print(data.head())


"""plt.plot(data)
plt.show()"""
#not the best plot bc of daily variation but we can see that there's a seasonal pattern evident



"""
Rolling Average and DF Tests  
"""


def rolling_avg(timeseries):
    rolmean = timeseries.rolling(10).mean()
    rolstd = timeseries.rolling(10).std()

    orig = plt.plot(timeseries, color= 'blue', label='Original')
    mean = plt.plot(rolmean, color='red',label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)


def DF_test(timeseries):
    print('Results of Dickey-Fuller Test: ')
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print(dfoutput)



"""
                                       Removing Trends and Seasonality 
"""
#aggregate into daily totals
tsOG = data.resample('D').sum()
print(tsOG)
ts = tsOG['REVENUE']
DF_test(ts)
#rolling_avg(ts)
#Looks like it's not stationary so we'll do some remodelling

ts_log = np.log(ts)

""" 
                                            --- Moving Average ---                                    
"""

moving_avg = ts_log.rolling(7).mean() #weekly rolling avg
"""plt.plot(ts_log)
plt.plot(moving_avg, color ='red')"""


ts_log_moving_avg_diff = ts_log - moving_avg

ts_log_moving_avg_diff.dropna(inplace=True)
#this is good now

"""
                                    --- Exponentially Weighted Average ---
"""

#doing a exponentially weighted average to spice things up...
ts_log_DF = ts_log.to_frame()

expweighted_avg = ts_log_DF.ewm(halflife=12).mean()
#plt.plot(ts_log_DF)
#plt.plot(expweighted_avg, color='blue')

ts_log_ewma_DIFF = ts_log_DF - expweighted_avg

DF_test(ts_log_ewma_DIFF)
#rolling_avg(ts_log_ewma_DIFF)
#plt.show()
#I like the way this looks and stats look very good as well, but the first looks better and had better stats


"""
                                            ---- Differencing ---
"""
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
DF_test(ts_log_diff)
#rolling_avg(ts_log_diff)

"""
Forecasting: acf/pacf
"""

#ACF PACF plots for q AND p variables
lag_acf = acf(ts_log_ewma_DIFF, nlags=15)
lag_pacf = pacf(ts_log_ewma_DIFF, nlags=15, method='ols')

"""#ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_ewma_DIFF)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_ewma_DIFF)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')


#PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_ewma_DIFF)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_ewma_DIFF)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()"""
#from here we get p,q=1

model = ARIMA(ts_log_DF, order=(1,1,0))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_ewma_DIFF)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.show()
#MA model worked best so went with that

"""
                                         --- Predictions --- 
"""
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()


predictions_ARIMA_log = pd.Series(ts_log_DF.iloc[:, 0], index=ts_log_DF.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
print(predictions_ARIMA_log.head())


predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA, color='red')
plt.show()


"""final = pd.concat([tsOG, predictions_ARIMA], axis=1, sort= False)
final.to_csv('predictions.csv')"""