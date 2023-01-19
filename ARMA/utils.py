import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
pd.options.mode.chained_assignment = None  # default='warn'

from model import AR,MA,adf_check

### getting preprocessed data from (add notebook location)

df = pd.read_csv("augmented.csv",parse_dates=True, index_col="datetime")

### Making the data stationary by differencing 

df_testing = pd.DataFrame(np.log(df.Value).diff().diff(12))
adf_check(df_testing.Value.dropna())

### Plot for Autocorrelation (notebook)

### fitting the AR model

best_RMSE=100000000000
best_p = -1

for i in range(1,21):
  [df_train,df_test,theta,intercept,RMSE] = AR(i,pd.DataFrame(df_testing.Value))
  if RMSE < best_RMSE:
      best_RMSE = RMSE
      best_p = i

[df_train,df_test,theta,intercept,RMSE] = AR(best_p,pd.DataFrame(df_testing.Value))

df_c = pd.concat([df_train,df_test])
df_c[['Value','Predicted_Values']].plot()

res = pd.DataFrame()
res['Residuals'] = df_c.Value - df_c.Predicted_Values

### this shows the mean and variance of the AR model
res.plot(kind='kde')

### fitting the MA model
best_RMSE=100000000000
best_q = -1

for i in range(1,13):
  [res_train,res_test,theta,intercept,RMSE] = MA(i,pd.DataFrame(res.Residuals))
  if RMSE < best_RMSE:
      best_RMSE = RMSE
      best_p = i

### getting the best value for MA
[res_train,res_test,theta,intercept,RMSE] = MA(best_q,pd.DataFrame(res.Residuals))
print(theta)
print(intercept)

### Concatenating the results and making it ARIMA
res_c = pd.concat([res_train,res_test])
df_c.Predicted_Values += res_c.Predicted_Values
df_c[['Value','Predicted_Values']].plot()

### getting the original data back

df_c.Value += np.log(df).shift(1).Value
df_c.Value += np.log(df).diff().shift(12).Value
df_c.Predicted_Values += np.log(df).shift(1).Value 
df_c.Predicted_Values += np.log(df).diff().shift(12).Value
df_c.Value = np.exp(df_c.Value)
df_c.Predicted_Values = np.exp(df_c.Predicted_Values)

print(df_c)
