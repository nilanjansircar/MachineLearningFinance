#Program to forcast Adj. Close of n days after.
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("WIKI-GOOGL.csv") # Reading the data to Pandas data frame, can be done directly using quandl

df=df.iloc[::-1] #Reverse the data ordering, as the given data is in reverse chronological order

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']] # Shortening the data frame to relevant ones
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change']=(df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
n=10 # no. of days in future where it will predict

df['label'] = df[forecast_col].shift(-n)
df=df[:-n]

N_tt = int(math.ceil(0.1 * len(df))) # We will train and test our classifier on 90% of the data


X = np.array(df.drop(['label'], 1)) #defining the features
X = preprocessing.scale(X)
X_lately=X[-N_tt:] #10% of the data for which classfier will predict
X=X[:-N_tt] #defining the features for 90% of the data


y = np.array(df['label']) #defining the features
y_known = y[-N_tt:]#10% of the labels
y = y[:-N_tt] #defining the label for 90% of the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##########Training and testing the Various Classifier on 90% of data
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)
######################################################
##########Training and testing the Linear Classifier on 90% of data
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print "Linear Regression ", accuracy
########################################################

forecast_set = clf.predict(X_lately) # Predicting using Linear Regression
diff = forecast_set - y_known # difference between known and prediction

diff = diff[:-1] # removing the last case where there was no known value

avg_diff=np.average(diff)

print "Prediction and known averaged difference", avg_diff
sigma = math.sqrt(np.average((diff-avg_diff)**2))
print "Standard deviation of Prediction and known difference", sigma
skewness = np.average(((diff-avg_diff)/sigma)**3)
print "Skewness of Prediction and known difference", skewness
kurt=np.average(((diff-avg_diff)/sigma)**4)
print "Kurtosis of Prediction and known difference", kurt

x=range(len(diff))
plt.figure(1)
forecastplot=plt.scatter(x,forecast_set[:-1],c='r')
knownplot=plt.scatter(x,y_known[:-1], c='b')
plt.legend([forecastplot,knownplot],['forecasted value', 'known value'])
plt.xlabel('days')
plt.ylabel('Adj.Close')

plt.figure(2)  
plt.scatter(x,diff)
plt.xlabel('days')
plt.ylabel('(forcated-known) of Adj.Close')
plt.figure(3)  
plt.hist(diff, bins=20)
plt.title('Histogram of (forcated-known) values of Adj.Close')
plt.show()
