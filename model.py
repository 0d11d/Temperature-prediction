import os.path
import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import tensorflow as tf
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json, load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_absolute_error,accuracy_score,mean_squared_error 

from matplotlib import pyplot as plt
import math

train = pd.read_csv('train.csv')
validation = pd.read_csv('validation.csv')


# Process data for model

total_population = 7900000000
total_area = 196939900 
total_gdp = 847100000000000

def process(df):
    df = df.drop(['Year', 'Month', 'country', 'gdp', 'Annual_CO2_emissions', 'Five-year', 'Ten-year'], axis=1)
    df = df.dropna()
    
    df['population'] = df['population']/total_population
    df['Area (sq. mi.)'] = df['Area (sq. mi.)']/total_area
    #df['gdp'] = df['gdp']/total_gdp
    #df['Annual_CO2_emissions']= df['Annual_CO2_emissions']/(df['Annual_CO2_emissions'].max())
    
    df['sin(lat)'] = df['latitude'].transform(lambda x: math.sin(x))
    df['sin(lon)'] = df['longitude'].transform(lambda x: math.sin(x))
    df['cos(lat)'] = df['latitude'].transform(lambda x: math.cos(x))
    df['cos(lon)'] = df['longitude'].transform(lambda x: math.cos(x))
    df['x'] = df['cos(lat)']* df['cos(lon)']
    df['y'] = df['cos(lat)']* df['sin(lon)']
    df['z'] = df['sin(lat)']
    df['x'] = df['x'].transform(lambda x: (x+1)/2)
    df['y'] = df['y'].transform(lambda x: (x+1)/2)
    df['z'] = df['z'].transform(lambda x: (x+1)/2)

    
    df = df.drop(['sin(lat)', 'sin(lon)', 'cos(lat)', 'cos(lon)','latitude','longitude'], axis=1)                             
    df['countryid'] = df['countryid']/234
    
    # assume Tmax=40, Tmin=-30   
    df['Monthly'] = (df['Monthly'] +30)/70
    df['Annual'] = (df['Annual'] +30)/70
    
    df = df.set_index('time')
    df = df[['countryid','population','Area (sq. mi.)','x', 'y', 'z','Annual', 'Monthly' ]]

    return df



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



def split(df):
    X = df.drop(['var8(t)'], axis=1)
    y = df['var8(t)']
    return X,y

df_train=process(train)
df_train=series_to_supervised(df_train,1, 1)
# drop the non-realted columns
df_train=df_train.drop(["var1(t)","var2(t)","var3(t)","var4(t)","var5(t)","var6(t)","var7(t)" ], axis=1)

df_val=process(validation)
df_val=series_to_supervised(df_val,1, 1)
# drop the non-realted columns
df_val=df_val.drop(["var1(t)","var2(t)","var3(t)","var4(t)","var5(t)","var6(t)","var7(t)"], axis=1)



# prepare train and validation data

lstm_train = df_train.to_numpy()
X_train, y_train = lstm_train[:, :-1], lstm_train[:, -1]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

# validation data
lstm_val = df_val.to_numpy()
X_val, y_val = lstm_val[:, :-1], lstm_val[:, -1]
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

print(lstm_train.shape,X_train.shape, y_train.shape)
print(lstm_val.shape,X_val.shape, y_val.shape)

 # Build model
from datetime import datetime as dt
start = dt.now()

warnings.filterwarnings("ignore")

# design network
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(1)])

model.compile(loss='mae', optimizer='adam')
model.summary()

early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')

history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_val, y_val), verbose=2, callbacks=[early_stopings], shuffle=False)

running_secs = (dt.now() - start).seconds




# Saving the model for Future Inferences

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("model.h5")


# plot the result
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 0.05])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

plot_loss(history)


# evaluate model
from datetime import datetime as dt
start1 = dt.now()

yhat = model.predict(X_val)

running_secs1 = (dt.now() - start1).seconds


# Calculate the errors of the model 
rmse = math.sqrt(mean_squared_error(y_val,yhat))
mae = mean_absolute_error(y_val+1e-20, yhat.reshape(len(y_val)))
r2 = r2_score(y_val, yhat)
print("Root mean squared error: {}".format(rmse)) 
print("Mean absolute percentage error: {}".format(mae)) 
print("r2 score:: {}".format(r2))

#plot the prediction and the observation data
plt.figure(figsize=(18, 6), dpi=80)
plt.plot(yhat, color='red', label='predict')
plt.plot(y_val, color='blue', label='actual')
plt.legend()

plt.ylabel("Temperature / C")















