#!/usr/bin/env python
# coding: utf-8

# ## Word Bank Data Analysis based on Climate Factors

# In[1]:


# Importing Necessary Library

import pandas as pd
import numpy as np
import requests

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


# ## Getting Public Data From World Bank through API

#     Indicators and Country List are organised as per Features of the Data

# In[2]:


BASE_URL='http://api.worldbank.org/v2/' # Base URL used in all the API calls


INDICATOR_CODES=['EN.ATM.CO2E.PC', 'EG.ELC.ACCS.ZS', 'AG.LND.FRST.ZS','AG.LND.AGRI.ZS'] # List of indicators 

COUNTRY_LIST=['USA', 'India', 'China', 'Japan', 'Canada', 'Great Britain', 'South Africa'] # List of countries

# mapping of feature codes to more meaningful names
featureMap={
    'EN.ATM.CO2E.PC':'CO2 emissions (metric tons per capita)',
    'EG.ELC.ACCS.ZS':'Access to electricity (% of population)',
    'AG.LND.FRST.ZS':'Forest area (% of land area)',
    'AG.LND.AGRI.ZS':'Agricultural land (% of land area)'
}

# Mapping of country codes to their actual names
countryMap={
    "US": "USA",
    "IN":"India",
    "CN": "China",
    "JP": "Japan",
    "CA": "Canada",
    "GB": "Great Britain",
    "ZA": "South Africa"
}


params = dict() # constant parameters

params['format']='json' # JSON response

params['per_page']='100' # default page size - API call per feature

params['date']='1960:2018' # Range of years 


#     Loading the JSON data

# In[3]:


# Function to get JSON data from the endpoint
def loadJSONData(country_code): 
    dataList=[]
    
    
    for indicator in INDICATOR_CODES: 
        
        
        url=BASE_URL+'countries/'+country_code.lower()+'/indicators/'+indicator
        
        
        response = requests.get(url, params=params)
        
        
        if response.status_code == 200 and ("message" not in response.json()[0].keys()):
            
            indicatorVals=[]
            
            
            if len(response.json()) > 1:
                
                
                for obj in response.json()[1]:
                    
                    # check for empty values
                    if obj['value'] is "" or obj['value'] is None:
                        indicatorVals.append(None)
                    else:
                    # if a value is present, add it to the list of indicator values
                        indicatorVals.append(float(obj['value']))
                dataList.append(indicatorVals)
        else:
            # print an error message if the API call failed
            print("Error in Loading the data. Status Code: " + str(response.status_code))
            
    
    dataList.append([year for year in range(2018, 1959, -1)])
    
    return dataList

#----------------------------------------------------------------------------------------------------

# function to invokde the loadJSONData function 
def getCountrywiseDF(country_code):
    
    
    col_list=list(featureMap.values())
    # append the year column name
    col_list.append('Year')
    
    print("------------------Loading data for: "+countryMap[country_code]+"-----------------------")
    
    
    dataList=loadJSONData(country_code)
    
     
    df=pd.DataFrame(np.column_stack(dataList), columns=col_list)
    
    # add the country column by extracting the country name from the map using the country code
    df['Country'] = countryMap[country_code]
    
    
    display(df.head())
    
    
    return df


# In[4]:


US_df=getCountrywiseDF('US')
IN_df=getCountrywiseDF('IN')
CN_df=getCountrywiseDF('CN')
JP_df=getCountrywiseDF('JP')
CA_df=getCountrywiseDF('CA')
GB_df=getCountrywiseDF('GB')
ZA_df=getCountrywiseDF('ZA')

print("Data Loading Completed")


# # Data Pre-processing

# In[5]:


# store all the DataFrames in a list to iteratively apply pre-processing steps
list_df=[US_df.copy(), IN_df.copy(), CN_df.copy(), JP_df.copy(), CA_df.copy(), GB_df.copy(), ZA_df.copy()]


# ## Dropping features with majority values missing

# In[6]:


# Function to identify missing features and remove features that aren't useful

def remove_missing_features(df):
    
    # validation for dataframe
    if df is None:
        print("No DataFrame received!")
        return
    
    # create a copy of the dataframe to avoid changing the original
    df_cp=df.copy()
    
    print("Removing missing features for: " + df_cp.iloc[0]['Country'])
    
    # find features with non-zero missing values
    n_missing_vals=df.isnull().sum()

    # get the index list of the features with non-zero missing values
    n_missing_index_list = list(n_missing_vals.index)
    
    
    # no. of rows we get the ratio of missing values - multipled by 100 to get percentage
    
    missing_percentage = n_missing_vals[n_missing_vals!=0]/df.shape[0]*100
    
    
    # list to maintain the columns to drop
    cols_to_trim=[]
    
    # iterate over each key value pair
    for i,val in enumerate(missing_percentage):
        
        if val > 75:
            
            cols_to_trim.append(n_missing_index_list[i])

    

    if len(cols_to_trim) > 0:
        
        df_cp=df_cp.drop(columns=cols_to_trim)
        print("Dropped Columns:" + str(cols_to_trim))
    else:
        print("No columns dropped")

    # return the updated dataframe
    return df_cp


# In[7]:


#  DF for each country.
# The function remove_missing_features 
list_df=list(map(remove_missing_features, list_df))


# ##  Fill missing values

# In[8]:


# Function to fill the remaining missing values with average values for columns

def fill_missing_values(df):
    
    # validation for dataframes
    if df is None:
        print("No DataFrame received")
        return

    
    df_cp=df.copy()
    
    print("Filling missing features for: " + df_cp.iloc[0]['Country'])
    
    
    cols_list=list(df_cp.columns)
    
    # exclude the last column - Country
    
    cols_list.pop()
    
    # replace all None values with NaN, fillna only works on nans
    df_cp.fillna(value=pd.np.nan, inplace=True)
    
    # replace all NaN values with the mean of the column values
    for col in cols_list:
        df_cp[col].fillna((df_cp[col].mean()), inplace=True)

    print("Filling missing values completed")
    return df_cp


# In[9]:


# each DF for each country.
# The function fill_missing_features 

list_df=list(map(fill_missing_values, list_df))


# ## Changing the type of Numeric but Categorical Features

# In[10]:


# Function to change year type
def change_year_type(df):
    
    print("Changing type of Year for: " + df.loc[0]['Country'])
    # validation to check if year column exists in the dataframe
    if 'Year' in df.columns:
        # convert year to a string
        df['Year'] = df.Year.astype(str)
    
    print("Completed changing type")
    # return the updated df
    return df


# In[11]:


# call the function on each DF for each country.

list_df=list(map(change_year_type, list_df))


# ## Check the number of features

# In[12]:


# each dataframe has the same number of columns

print('Total number of features: %d\n'%(list_df[0].shape[1]))
list_df[0].dtypes


# ## Storing the cleaned dataset into CSV files

# In[13]:


# Function to write processed data to CSV files
def write_data():
    # validation to check if the number of countries and the dataframes match
    assert len(list(countryMap.keys()))==len(list_df)
    
    # iterate over country names from the country map and the list of dataframes simultaneously
    for country_name, df_data in zip(COUNTRY_LIST, list_df):
        print("Writing data for: " + country_name)
        file_name=country_name+".csv"
        # convert to CSV
        try:
            df_data.to_csv(file_name, index=False)
            print("Successfully created: " + file_name)
        except:
            # in case an error occurs in I/O
            print("Error in writing to: " + file_name)
        


# In[14]:


# writing data
write_data()


# # Analyse and Summarise the cleaned dataset

# ## Read the cleaned data from the CSV files

# In[15]:


# read the cleaned data from every CSV
try:
    df_cleaned_us=pd.read_csv('USA.csv')
    df_cleaned_in=pd.read_csv('India.csv')
    df_cleaned_cn=pd.read_csv('China.csv')
    df_cleaned_jp=pd.read_csv('Japan.csv')
    df_cleaned_ca=pd.read_csv('Canada.csv')
    df_cleaned_gb=pd.read_csv('Great Britain.csv')
    df_cleaned_za=pd.read_csv('South Africa.csv')
    print("Successfully read all the files")
except:
    # handling I/O exceptions
    print("Unexpected error in reading a file. Check the file path and if a file exists with the name given.")

# data of one country to check loading
print("Displaying data for USA: ")
display(df_cleaned_us.head())

# list of clean dataframes 
list_cleaned_df=[df_cleaned_us, df_cleaned_in, df_cleaned_cn, df_cleaned_jp, df_cleaned_ca, df_cleaned_gb, df_cleaned_za]


# ## Preparing a combined DataFrame for further analysis

# In[16]:


combined_df=pd.concat(list_cleaned_df,sort=False)
combined_df.head(200)


# ## Descriptive Statistics

# In[17]:


# copying data so that the original data is not affected
# drop columns year and country
df_copy=combined_df.drop(['Year', 'Country'], axis='columns')
df_copy.describe()


# ## Correlation Matrix
#     To assess Relations between Variables

# In[18]:


# create a copy
df_cleaned_us_copy=df_cleaned_us.copy()
# Exclude the categorical features from the matrix
df_cleaned_us_copy.drop(['Year', 'Country'], inplace=True, axis='columns')

# plot a correlation matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_cleaned_us_copy.corr(), cmap='RdBu', center=0,ax=ax)
plt.savefig('correlation_us.png')
plt.show()


# ## Plotting the Cleaned Data

# ### Comparing CO2 Emissions of Countries in 2000 and 2018:

# In[19]:


# refer to the list of countries
list_countries = COUNTRY_LIST
# intialise two dataframes df_00- year 2000, df_18 - year 2018
df_00 = pd.DataFrame()
df_18 = pd.DataFrame()

# for each dataframe in the list of cleaned dataframes
for i,df in enumerate(list_cleaned_df):
    # pick the value of CO2 Emission for year 2000 and 2018
    df_00[list_countries[i]] = df[df['Year'] == 2000]["CO2 emissions (metric tons per capita)"]
    df_18[list_countries[i]] = df[df['Year'] == 2018]["CO2 emissions (metric tons per capita)"]

# The resulting dataframes have countries and yesrs 2000 and 2018
# Transpose of Data is taken
df_00 = df_00.T
df_18 = df_18.T

pd.options.display.float_format = '{:,.1f}'.format  # set other global format

# rename the columns to the year
df_00 = df_00.rename(columns={18 : 2000})
df_18 = df_18.rename(columns={0 : 2018})

# join the dataframes for both the years
df_both_years= df_00.join(df_18)

# the index is the Country name, hence we add it as a column into the data frame.
df_both_years['Countries'] = df_both_years.index

# drop the original index
df_both_years.reset_index(drop=True)

print("Data of Total Population for 2000 and 2018 for all countries: ")
display(df_both_years)


# In[20]:


plt.figure(figsize=(7, 5))
# plot the chart
df_both_years.plot(kind='bar',x='Countries',y=[2000, 2018])


# ### Avg. Access to electricity (%) and  Agricultural land (%) for countries across all the years

# In[21]:


def group_df(feature):
    # create a new dataframe
    df_grouped=pd.DataFrame()

    # find average for each country
    df_grouped['Avg. ' + feature]=combined_df.groupby('Country')[feature].mean()

    # set the index as a column - countries
    df_grouped['Country']=df_grouped.index

    # drop the index
    df_grouped.reset_index(drop=True, inplace=True)

    # sort the rows 
    df_grouped.sort_values('Avg. '+feature, inplace=True, ascending=False)

    print("Avg. " + feature)
    display(df_grouped)
    
    return df_grouped

def plot_bar(df, x_feature, y_feature):
    # bar plot
    plt.figure(figsize=(8, 5))
    sns.set(style="whitegrid")
    ax = sns.barplot(
        data= df,
        x= x_feature,
        y= "Avg. " + y_feature)


# In[22]:


df_pop=group_df('Access to electricity (% of population)')
plot_bar(df_pop, 'Country', 'Access to electricity (% of population)')

print("========================================================")
df_agr=group_df('Agricultural land (% of land area)')
plot_bar(df_agr, 'Country', 'Agricultural land (% of land area)')


# ### Forest area (% of land area) for all countries in the last 10 years

# In[23]:


# function to to form a dataframe 
def extract_columns(df_cleaned):
    df=pd.DataFrame()
    # pick data for the recent 10 years
    df['Year']=df_cleaned.loc[:10, 'Year']
    df['Forest area (% of land area)']=df_cleaned.loc[:10, 'Forest area (% of land area)']
    df['Country']=df_cleaned.loc[:10, 'Country']
    return df

# function to fetch a single dataframe with 3 features from each country
def form_land_df():
    # function call to extract_columns()
    indf=extract_columns(df_cleaned_in)
    usdf=extract_columns(df_cleaned_us)
    cndf=extract_columns(df_cleaned_cn)
    jpdf=extract_columns(df_cleaned_jp)
    cadf=extract_columns(df_cleaned_ca)
    gbdf=extract_columns(df_cleaned_gb)
    zadf=extract_columns(df_cleaned_za)
    
    land_df=pd.concat([indf, usdf, cndf, jpdf, cadf, gbdf, zadf], ignore_index=True)
    return land_df

# get the combined DF
land_df=form_land_df()

print("Few records from the Dataframe containing Year, Forest area (% of land area):")
display(land_df.head())

# set figure size
plt.figure(figsize=(7, 5))
sns.set(style="whitegrid")
# plot 
ax=sns.lineplot(x='Year', y='Forest area (% of land area)', 
                hue='Country', style="Country",palette="Set2", 
                markers=True, dashes=False, data=land_df, linewidth=2.5)


# ### CO2 emissions (metric tons per capita) vs. Forest area (% of land area) for India and China

# In[24]:


# function to extract specific columns from the DFs for India and China
def form_in_cn_df():
    # for India
    indf=df_cleaned_in[['CO2 emissions (metric tons per capita)', 'Forest area (% of land area)', 'Country']]
    # for China
    cndf=df_cleaned_cn[['CO2 emissions (metric tons per capita)', 'Forest area (% of land area)', 'Country']]
    # combine the two dataframes
    in_cn_df=pd.concat([indf, cndf])
    return in_cn_df

# get the desired data
in_cn_df=form_in_cn_df()
print("Few records from the selected features: ")
display(in_cn_df.head())

# scatter plot
plt.figure(figsize=(7, 5))
sns.set(style="whitegrid")
ax=sns.scatterplot(x='Forest area (% of land area)', 
                   y='CO2 emissions (metric tons per capita)', 
                   hue='Country', palette="bright", data=in_cn_df)


# ### Forest area (% of land area) vs. CO2 emissions (metric tons per capita) for Canada upto 2015

# In[25]:


# read the columns from the df for Canada
df=df_cleaned_ca.loc[3:, ['Forest area (% of land area)','CO2 emissions (metric tons per capita)', 'Year']]

print("First few records of the data: ")
display(df.head())

# line plot
plt.figure(figsize=(6, 5))
sns.set(style="whitegrid")
sns.lineplot(x='Forest area (% of land area)', y='CO2 emissions (metric tons per capita)', palette="colorblind",data=df, linewidth=2.5)


# ### INDICATORS
#     CO2 emissions (metric tons per capita)
#     Access to electricity (% of population)
#     Forest area (% of land area)

# In[26]:


# Pick the columns Year, and 3 different power consumptions from the dataframe 
plt.plot(df_cleaned_in.loc[5:, ['Year']],df_cleaned_in.loc[5:, ['CO2 emissions (metric tons per capita)']],'.-')
plt.plot(df_cleaned_in.loc[5:, ['Year']],df_cleaned_in.loc[5:, ['Access to electricity (% of population)']],'.-')
plt.plot(df_cleaned_in.loc[5:, ['Year']],df_cleaned_in.loc[5:, ['Forest area (% of land area)']],'.-')

plt.legend(['CO2 emissions (metric tons per capita)', 'Access to electricity (% of population)', 'Forest area (% of land area)'], loc='best')
plt.title("Levels of Indicators\n")
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.show()


# In[ ]:




