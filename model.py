import os.path
import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json, load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_absolute_error,accuracy_score,mean_squared_error 

from matplotlib import pyplot as plt
import seaborn as sns

import math


# extract and prepare GDP dataset
gdp = pd.read_csv("0_gdp.csv")
gdp_df = pd.melt(gdp, id_vars='country', value_vars=['1960','1961', '1962', '1963', '1964','1965', '1966', '1967', '1968', '1969',  
                                                     '1970','1971', '1972', '1973', '1974','1975', '1976', '1977', '1978', '1979', 
                                                     '1980','1981', '1982', '1983', '1984','1985', '1986', '1987', '1988', '1989', 
                                                     '1990','1991', '1992', '1993', '1994','1995', '1996', '1997', '1998', '1999', 
                                                     '2000','2001', '2002', '2003', '2004','2005', '2006', '2007', '2008', '2009',
                                                     '2010','2011', '2012', '2013', '2014','2015', '2016', '2017', '2018', '2019',
                                                     '2020','2021'])
gdp_df = gdp_df.rename(columns = {'country':'country', 'variable':'Year', 'value':'gdp'})
gdp_df = gdp_df.sort_values(by=['country','Year'])
gdp_df = gdp_df.reset_index(drop=True)


# extract and prepare population dataset
population = pd.read_csv("0_population.csv")
population_df = pd.melt(population, id_vars='country', value_vars=['1960','1975','1980','1985',
                                                                   '1990','1995','2000','2005','2010',
                                                                   '2015','2016','2017','2018','2019','2020','2021','2022'])
population_df = population_df.rename(columns = {'country':'country', 'variable':'Year', 'value':'population'})
population_df = population_df.sort_values(by=['country','Year'])
population_df = population_df.reset_index(drop=True)


# extract and prepare greenhouse gas emission dataset
emission = pd.read_csv("0_CO2.csv")

# extract location dataset
geo = pd.read_csv("0_geo.csv")

# extract area dataset
area = pd.read_csv("0_area.csv")

# expand dataset 
df = gdp_df.merge(population_df, on=['country','Year'], how='left')
df['country'] = df['country'].astype(str)
df['Year'] = df['Year'].astype(int)
emission['country'] = emission['country'].astype(str)
emission['Year'] = emission['Year'].astype(int)
df = emission.merge(df, on=['country','Year'], how='left')
df = df.merge(geo, on='country', how='left')
country = df.merge(area, on='country', how='left')
country['gdp'] = country['gdp'].fillna(country.groupby('country')['gdp'].transform(lambda x: x.min()))
country['population'] = country['population'].fillna(country.groupby('country')['population'].transform(lambda x: x.min()))
country['gdp'] = country['gdp'].astype('int', errors = 'ignore')
country['population'] = country['population'].astype('int',errors = 'ignore')


# Add absolute temperature to raw dataset
absolute_t = pd.read_csv("0_Temp.csv")
name = absolute_t['country'].tolist()
temp = absolute_t['Absolute values'].tolist()

dataset = pd.DataFrame({})
train = pd.DataFrame({})
validation = pd.DataFrame({})
test = pd.DataFrame({})

for i in range(len(name)):
    read_name = name[i]+".csv"
    #save_name = "abs_"+name[i]+".csv"
    
    if os.path.isfile(read_name):
        df = pd.read_csv(read_name)
        
        train_split = int(len(df)*0.8)
        
        # convert to timestamp
        df['Day'] = 1
        df['time'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        
        # add absolute values
        df['Monthly'] = df['Monthly']+temp[i]
        df['Annual'] = df['Annual']+temp[i]
        df['Five-year'] = df['Five-year']+temp[i]
        df['Ten-year'] = df['Ten-year']+temp[i]
        df['Twenty-year'] = df['Twenty-year']+temp[i]
        
        # merge the country into dataset by country name
        df['country'] = name[i]
        df['countryid'] = i+1
               
        df_merge0 = df.merge(country, on=['country', 'Year'], how='left')
        
        
        #select features
        df_merge = df_merge0[['time','Year', 'Month', 'country','countryid',
                             'latitude', 'longitude', 'population', 'Area (sq. mi.)',
                             'gdp', 'Annual_CO2_emissions', 
                             'Monthly', 'Annual','Five-year', 'Ten-year']]
        
        df_merge = df_merge.reset_index(drop=True)
        
        
        train_df0 = df_merge[df_merge['Year']<2001]
        train_df = train_df0.reset_index(drop=True)
        
        val_df0 = df_merge[(df_merge['Year']>2000)&(df_merge['Year']<2017)]
        val_df = val_df0.reset_index(drop=True)
        
        test_df0 = df_merge[df_merge['Year']>2016]
        test_df = test_df0.reset_index(drop=True)

        dataset = pd.concat([dataset,df_merge])
        
        train = pd.concat([train,train_df])
        validation = pd.concat([validation,val_df])
        test = pd.concat([test,test_df])

    else:
        pass



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

plt.ylabel("Capacity / Ah")















