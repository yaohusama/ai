import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
df=pd.read_csv("1.csv")
df['new_date']=df['tr_date']+' '+df['tr_time']
res=[]
df['new_date']=df['new_date'].replace('-','/')
for i in df['new_date']:
    if i[0]=='0':
        res.append('2'+i)
    else:
        res.append(i)
df['newer_date'] = pd.to_datetime(res,format='%Y/%m/%d %H:%M:%S')  # 将字符串索引转换成时间索引
df.index = pd.date_range(start = '2018/3/1 00:00:00',periods=826, freq='H')
print(df.index)
ts1 = df['upgb']
ts2 = df['downgb']  # 生成pd.Series对象
ts2.head()
import statsmodels.tsa.stattools as ts


def draw_trend(timeseries, size):
    f = plt.figure(facecolor='white')
    rol_mean = timeseries.rolling(window=size).mean()
    rol_std = timeseries.rolling(window=size).std()
    timeseries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_std.plot(color='black', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
def draw_ts(timeseries):
    f = plt.figure(facecolor='white')
    timeseries.plot(color='blue')
    plt.show()
def teststationarity(ts):
    dftest = adfuller(ts)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput
draw_trend(ts1, 12)
from statsmodels.tsa.stattools import *
teststationarity(ts1)
ts1_log = np.log(ts1)
def draw_moving(timeSeries, size):
    f = plt.figure(facecolor='white')
    rol_mean = timeSeries.rolling(window=size).mean()
    rol_weighted_mean = pd.DataFrame.ewm(timeSeries, span=size)
    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
draw_moving(ts1_log, 12)
diff_12 = ts1_log.diff(12)
diff_12.dropna(inplace=True)
diff_12_1 = diff_12.diff(1)
diff_12_1.dropna(inplace=True)
teststationarity(diff_12_1)
print(ts1.index)
from statsmodels.tsa.seasonal import seasonal_decompose


def decompose(timeseries):
    decomposition = seasonal_decompose(timeseries)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return trend, seasonal, residual
trend, seasonal, residual = decompose(ts1_log)
residual.dropna(inplace=True)
draw_trend(residual, 12)
teststationarity(residual)
rol_mean = ts1_log.rolling(window=12).mean()
rol_mean.dropna(inplace=True)
ts_diff_1 = rol_mean.diff(1)
ts_diff_1.dropna(inplace=True)
teststationarity(ts_diff_1)
ts_diff_2 = ts_diff_1.diff(1)
ts_diff_2.dropna(inplace=True)
teststationarity(ts_diff_2)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def draw_acf_pacf(ts,lags):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts,ax=ax1,lags=lags)
    ax2 = f.add_subplot(212)
    plot_pacf(ts,ax=ax2,lags=lags)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
draw_acf_pacf(ts_diff_2,30)
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_diff_1, order=(1,1,1))
result_arima = model.fit( disp=-1, method='css')
predict_ts = result_arima.predict(len(ts1)-5,len(ts1))
diff_shift_ts = ts_diff_1.shift(1)
diff_recover_1 = predict_ts.add(diff_shift_ts)
rol_shift_ts = rol_mean.shift(1)
diff_recover = diff_recover_1.add(rol_shift_ts)
rol_sum = ts_log.rolling(window=11).sum()
rol_recover = diff_recover*12 - rol_sum.shift(1)
log_recover = np.exp(rol_recover)
ts1 = ts1[log_recover.index]
print(('RMSE: %.4f'% np.sqrt(sum((log_recover-ts1)**2)/ts1.size)))