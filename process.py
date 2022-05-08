import os.path
import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

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


















