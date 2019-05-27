#The effect of Greenspace on stress on a community area level
    #holding constant SES, food availibiliyt and crime
    #may include self-reported health
#or Greespace and suicide

#I am concrned about the time-ing of all these, would need to assume
#pple not moving too often and that greenspace is not changeing w/in timeframe
#those may be reasonable, hard to say in low SES for moveing
#Looks like the veg and greens space only done on select years
#Looks like the survey also only done on select years


#collecting data for covariates
#includes per capita income
#2005 – 2011
('https://data.cityofchicago.org/api/views/iqnk-2tcu/rows.csv?accessType=DOWNLOAD')

#grocery stores 2011 by community area
'https://data.cityofchicago.org/api/views/4u6w-irs9/rows.csv?accessType=DOWNLOAD'

#produce carts (by address) 2012 at ealiest
'https://data.cityofchicago.org/api/views/divg-mhqk/rows.csv?accessType=DOWNLOAD'

#git Hub has more datasets but Monday TA not sure how to scrape them

#Bluespace 2011
#Percentage of community area land cover that is open water.
('https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/720/Blue_Index__Land_Cover___Ave_Annual__v2.xlsx')

#total population per area 2012-2016
#https://chicagohealthatlas.org/indicators/total-population

#crimes 2001-present: would need some cleaing!
#https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2/data

#overall health self-report 2015-17
#can't seem to get?
#info about it https://www.chicago.gov/city/en/depts/cdph/supp_info/healthy-communities/healthy-chicago-survey.html


#import packages
import pandas as pd
import datetime

import pandas_datareader.data as web
from pandas_datareader import wb
import requests
#from bs4 import BeautifulSoup
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf 

b_path = '~Sarah/Desktop/Programming' #set path
os.chdir(os.path.expanduser(b_path)) #set working directory


#read in data and organize:
#datasets: SES, vegitation, stress, greenspace, cause of death, health centers 

#Chicago SES metrics by community area 2008 – 2012
SES = requests.get('https://data.cityofchicago.org/api/views/kn9c-c2s2/rows.csv?accessType=DOWNLOAD')
SESdata = SES.text
with open('Chicago SES df', 'w') as ofile:
    ofile.write(SESdata)
SESdf = pd.read_csv(os.path.join(b_path, 'Chicago SES df'))
SESdf.columns
SESdf.rename(columns = {'PERCENT HOUSEHOLDS BELOW POVERTY':'PER_HOUSEHOLDS_BELOW_POVERTY', 'HARDSHIP INDEX':'HARDSHIP_INDEX'}, inplace=True)

#vegitation index by community area 2017
#Annual average of vegetation index 
#(Normalized Difference Vegetation Index) determined from satellite images. 
#The index range is -1 to 1 where higher numbers correspond to more vegetation.
vegitation = requests.get('https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/719/Vegetation_Index__new___Ave_Annual_.xlsx')
vegdata = vegitation.content
with open('veg_df.xls', 'wb') as ofile:
    ofile.write(vegdata)
vegdf = pd.read_excel(os.path.join(b_path, 'veg_df.xls')) 
#vegdf.head()
#vegdf.columns
#vegdf['Geo_Group']
#vegdf['Community Area Number'].head()

vegdf.dropna(axis=1,inplace=True) #drop empty columns 
#vegdf.drop(['Map_Key'],axis=1,inplace=True)) #drop a col that is not meaningful here

#Works, yay!
#community name has community number before it, split into two columns
tempcol = vegdf['Geo_Group'].str.split("-", expand = True)
vegdf['temp'] = tempcol[0] 
vegdf['Community Area Name'] = tempcol[1]
#citation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html
#citation: https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/

vegdf.drop(columns = ['Geo_Group', 'temp', 'Category', 'SubCategory', 'Geography'], inplace = True)
vegdf.rename(columns = {'Geo_ID':'Community Area Number', 'Ave_Annual_Number':'Ave_Annual_vegitation_num'}, inplace=True)

#stressed 2015-16
#https://www.chicagohealthatlas.org/indicators/frequently-stressed
stress = requests.get('https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/654/Frequently_stressed.xlsx')
stressdata = stress.content
with open('stress_df.xls', 'wb') as ofile:
    ofile.write(stressdata)
stressdf = pd.read_excel(os.path.join(b_path, 'stress_df.xls')) 
#stressdf.head()
#stressdf.columns
stressdf['Geo_Group']
#stressdf['Geo_ID']
#stressdf['SubCategory']

stressdf.dropna(axis=1,inplace=True) #drop empty columns 

#community name has community number before it, split into two columns
tempcol = stressdf['Geo_Group'].str.split("-", expand = True)
stressdf['temp'] = tempcol[0]
stressdf['Community Area Name'] = tempcol[1]
stressdf.drop(columns = ['Geo_Group', 'temp', 'Category', 'SubCategory', 'Flag', 'Geography'], inplace = True)
#citation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html
#citation: https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/
stressdf.rename(columns = {'Geo_ID':'Community Area Number', 'Lower_95CI_Weight_Percent': 'stress_Lower_95CI_Weight_Percent',\
   'Upper_95CI_Weight_Percent':'stress_Upper_95CI_Weight_Percent', 'Weight_Percent': 'stress_Weight_Percent',\
        }, inplace=True)

#Greenspace 2011
#Percentage of community area land cover that is vegeatation.
green = requests.get('https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/721/Green_Index__Land_Cover___Ave_Annual__v2.xlsx')
greendata = green.content
with open('green_df.xls', 'wb') as ofile:
    ofile.write(greendata)
green_df = pd.read_excel(os.path.join(b_path, 'green_df.xls'))
#green_df.shape
green_df.dropna(axis=1,inplace=True) #drop empty columns 
#green_df.columns
#green_df['Geo_Group']
#green_df['Geography'].head()

tempcol = green_df['Geo_Group'].str.split("-", expand = True)
green_df['temp'] = tempcol[0]
green_df['Community Area Name'] = tempcol[1]
green_df.drop(columns = ['Geo_Group', 'temp', 'Category', 'SubCategory','Geography', 'Map_Key'], inplace = True)
#citation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html
#citation: https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/
green_df.rename(columns = {'Geo_ID':'Community Area Number', 'Ave_Annual_Number': 'Ave_Annual_perc_green'}, inplace=True)


#Cause of death by community area 2006-2010
death = requests.get('https://data.cityofchicago.org/api/views/j6cj-r444/rows.csv?accessType=DOWNLOAD')
data = death.text
#data
with open('chicago_death_df', 'w') as ofile:
    ofile.write(data)

death_df = pd.read_csv(os.path.join(b_path, 'chicago_death_df'))
death_df.head()
death_df['Community Area Name']
death_df.shape #(1404, 17) -> ,9)
death_df.dropna(axis=1,inplace=True) #drop empty columns 
death_df.columns
death_df.rename(columns = {'Average Annual Deaths 2006 - 2010':'Avg_Annual_Deaths_06_10', 'Community Area': 'Community Area Number'}, inplace=True)
death_df.index
avg_an_death = death_df.pivot(index = 'Community Area Number', columns='Cause of Death', values='Avg_Annual_Deaths_06_10')
avg_an_death.drop(0, axis = 0, inplace = True)
avg_an_death.rename(columns = {'Suicide (intentional self-harm)':'Suicide', 'Diabetes-related':'Diabetes_related'}, inplace = True)
#remember to not rename avg anual deaths bc making df wide instead
avg_an_death.columns

#health centers by community area 
healthcr = requests.get('https://data.cityofchicago.org/api/views/cjg8-dbka/rows.csv?accessType=DOWNLOAD') 
healthcrdata = healthcr.text
with open('chicago_health_centers_df', 'w') as ofile:
    ofile.write(healthcrdata)

healthcr_df = pd.read_csv(os.path.join(b_path, 'chicago_health_centers_df'))
healthcr_df.head()
healthcr_df.columns
healthcr_df.shape
healthcr_df.dtypes
#healthcr_df.groupby('Community Area (#)').sum()

#sneeky fix
healthcr_df['count_of_health_crs'] = 1 
healthcr_df

count_of_crs = healthcr_df.groupby('Community Areas').sum().reset_index()
count_of_crs.columns

#this way didn't work for my purposes, but I think it is the way to do it
#get the count of how many health centers are in ea community area
count_centers = healthcr_df.pivot_table(index=['Community Areas'], aggfunc='size')
count_centers
#cite https://datatofish.com/count-duplicates-pandas/
count_centers.dtypes

#doesn't work
#int(count_of_crs['Community Areas'])

#good news I don't need to fix it!
count_of_crs['Community Areas'][0] == SES_green_death['Community Area Number'][0]

count_of_crs.drop(columns = ['Boundaries - ZIP Codes', 'Zip Codes', 'Census Tracts', 'Wards', ':@computed_region_awaf_s7ux'], inplace = True)




#out of date
stress_veg = stressdf.merge(vegdf, on='Community Area Number', how = 'inner')
stress_veg.columns 
stress_veg.dtypes
stress_veg.shape #(8,53) #(8,24)
#stress_veg['Upper_95CI_Weight_Percent']

big_df = stress_veg.merge(SESdf, on = 'Community Area Number', how = 'inner')
big_df.shape
big_df.columns
big_df.dtypes


#initial model, to see what we've got so far
temp_model = smf.ols('stress_Weight_Percent ~ Ave_Annual_vegitation_num' , data=big_df)
result = temp_model.fit()
result.summary()
#I'm not p-hacking, you're p-hacking....

#let's look
plt.plot(big_df['stress_Weight_Percent'], big_df['Ave_Annual_vegitation_num'], 'o')
#ok this dataset is way to small!
#cite: https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html



#lets test if SES and greenspace are related:
#suspect not bc downtown
SES_green = SESdf.merge(green_df, on='Community Area Number', how = 'inner')
SES_green.shape
SESdf.shape #lost one entry
green_df.shape #lost no entries
SES_green.columns


test_model = smf.ols('HARDSHIP_INDEX ~ Ave_Annual_perc_green' , data=SES_green)
result = test_model.fit()
result.summary()

test_model = smf.ols('PER_HOUSEHOLDS_BELOW_POVERTY ~ Ave_Annual_perc_green' , data=SES_green)
result = test_model.fit()
result.summary()

#test that I am running this right with a for certain correlation
test_model = smf.ols('PER_HOUSEHOLDS_BELOW_POVERTY ~ HARDSHIP_INDEX' , data=SES_green)
result = test_model.fit()
result.summary()
#yup

#where is the most greenspace? -this jsut calls what is the max
#there is high variance in pec green 13-0
green_df['Ave_Annual_perc_green'].max()
green_df['Ave_Annual_perc_green'].min()

green_df.iloc[0:10]
SESdf.iloc[0:10]

#lets take a look
plt.plot(SES_green['HARDSHIP_INDEX'], SES_green['Ave_Annual_perc_green'], 'o')
#oh! almost all very near 0 and then big values at all levels
plt.plot(SES_green['HARDSHIP_INDEX'], np.log(SES_green['Ave_Annual_perc_green']), 'o')
#interesting, do I maybe need to log my greenspace variable so that can see variation within 
# the many low greenspce areas?


#I wonder if there's a relxn between suicide and greenspace, probably not but...

#old plot
#we've got an outlier! #was using SESdf at the time (ea have a totals row)
avg_an_death['Suicide (intentional self-harm)'].max()
avg_an_death['Suicide'].mean()
np.argmax(avg_an_death['Suicide (intentional self-harm)'])
#cite https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
avg_an_death.iloc[0]
#whoops, its the totals! 

plt.plot(SES_green['HARDSHIP_INDEX'], avg_an_death['Suicide (intentional self-harm)'], 'o')
plt.show()
#hmm, neg rlxn betw hardship and suicide
#lets see if significant:
SES_green_death = SES_green.merge(avg_an_death, on='Community Area Number', how = 'inner')
SES_green_death.shape
SES_green_death.columns

model = smf.ols('Suicide ~ HARDSHIP_INDEX + Ave_Annual_perc_green' , data=SES_green_death)
result = model.fit()
result.summary()

plt.plot(SES_green['Ave_Annual_perc_green'], avg_an_death['Suicide'], 'o')
plt.show()

model = smf.ols('Suicide ~ HARDSHIP_INDEX + Ave_Annual_perc_green' , data=SES_green_death)
result = model.fit()
result.summary()

#lets look at some other causes of death, I expect related to SES
plt.plot(SES_green['HARDSHIP_INDEX'], avg_an_death['Diabetes_related'], 'o')
plt.show()
#does not look related, which is a big surprise
model = smf.ols('Diabetes_related ~ HARDSHIP_INDEX + Ave_Annual_perc_green' , data=SES_green_death)
result = model.fit()
result.summary()
#something is off, expect SES to be predictive of diabetese deaths




SES_green_death_healthcr = SES_green_death.merge(count_of_crs, left_on='Community Area Number', right_on='Community Areas', how = 'outer')
SES_green_death_healthcr.shape
SES_green_death_healthcr['count_of_health_crs']

#fill in Nan with 0 (bc if not in the previous database then doesn't have a health center)
SES_green_death_healthcr.fillna(value = 0, inplace=True)
#cite: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html

#lets see if there is a relationship between SES and health center placement
plt.plot(SES_green_death_healthcr['HARDSHIP_INDEX'], SES_green_death_healthcr['count_of_health_crs'], 'o')
plt.show()

model = smf.ols('count_of_health_crs ~ HARDSHIP_INDEX' , data=SES_green_death_healthcr)
result = model.fit()
result.summary()

#nope no relationship, good news


#adding into my model
model = smf.ols('Suicide ~ HARDSHIP_INDEX + Ave_Annual_perc_green + count_of_health_crs' , data=SES_green_death_healthcr)
result = model.fit()
result.summary()

#lets see if health centers help explain diabetese deaths 
model = smf.ols('Diabetes_related ~ HARDSHIP_INDEX + Ave_Annual_perc_green + count_of_health_crs' , data=SES_green_death_healthcr)
result = model.fit()
result.summary()
#odd, maybe just the presence of a health center isn't enough to help




#no don't do that either
'''
string = str(count_centers)
tempcol = string.str.split(r"\n", expand = True)
'''

#no dice
group = healthcr_df.groupby('Community Areas').sum()
for v in group['Community Areas']:
    int(v)


healthcr_df['Community Area (#)']

'''
ax.set_ylabel('Percent Reprorted Stressed')
ax.set_xlabel('Average Anual Vegitation in their community area of residentse')
plt.show()

fig, ax = plt.subplots(figsize=(12,6)) 
ax.plot(x, y, 'b:', label='Blue Line')
ax.plot(x, y[::-1], 'r--', label='Red Line') 
ax.set_ylabel('Random Walk')
ax.set_xlabel('Periods')
plt.title('Our Plots')
ax.legend(loc = 'upper right') 
'''


