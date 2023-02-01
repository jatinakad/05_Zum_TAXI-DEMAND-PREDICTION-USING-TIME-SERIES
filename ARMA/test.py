import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # Dickey–Fuller test:
    result = adfuller(timeseries['id'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


main_df = pd.read_csv("../../train.csv",index_col="pickup_datetime")
main_df.index =  pd.to_datetime(main_df.index)
concat_df = pd.DataFrame(main_df.groupby(main_df.index.strftime('%Y-%m-%d %H')).count()['id'])
concat_df = concat_df.dropna()
node_values = []
for i in concat_df.index:
    a = "".join("".join(i.split("-")).split(" "))
    node_values.append([a,concat_df.id[i]])


concat_df = concat_df[:200]

rolling_mean = concat_df.rolling(window = 12).mean()
rolling_std = concat_df.rolling(window = 12).std()
#plt.plot(concat_df, color = 'blue', label = 'Original')
#plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
#plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
#plt.legend(loc = 'best')
#plt.title('Rolling Mean & Rolling Standard Deviation')
#plt.show()

result = adfuller(concat_df['id'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

df_log = np.log(concat_df)
concat_df.index =  pd.to_datetime(concat_df.index)

decomposition = seasonal_decompose(concat_df) 
model = ARIMA(concat_df, order=(38,1,14))
results = model.fit()
plt.plot(concat_df[:int(len(concat_df)*.9)])
plt.plot(concat_df[int(len(concat_df)*.9):])
plt.plot(results.fittedvalues, color='green',alpha=.5)
plt.show()

