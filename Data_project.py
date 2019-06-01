'''
Your repository must contain the following:

Your code and commit history
The initial, unmodified dataframes you download
Saved .png versions of your plots
The final versions of the dataframe(s) you built
Your writeup (Word or markdown are both fine)

summarize the data with plots, tables and summary statistics, 
and then fit a simple model to it using Numpy or Statsmodels.
'''

#I got carried away.
#need to:
#make plots
#make tables ???? summary statistics???
#save df files before and after edits?
#Organize! do fn make sense here? do classes?



#outline: -don't forget to remove
def outline():
    pass
# setup: import packages, std and path
#Data
    #download data (I need the origional data saved)
    #organize: drop cols, re-name...
    #merge dfs
#Explore
    #Summary stats?
    #exploratory plots?
    #exploratory regressions?
#Analyze
    #Regressions?
    #make regression tables?
    #plots
#Output?
#Run? (if use classes)


#The effect of Greenspace on average anula death rate, various causes
#study shows reduced cardio-metabolic disorders w/ greenspace (neighbouhood level)
#so: 'such as hypertension, high blood glucose, obesity (both overweight and
#obese), high cholesterol, myocardiac infarction, heart disease, stroke, and diabetes;'

#I am concrned about the time-ing of all these, would need to assume greenspace was
#the same in earlier years.
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

#what if I did crime instead:
#standby
'''
#https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2/data
crime = requests.get('https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD')
start = datetime.date(year=2008, month=1,  day=1) #selecting what to grab
end   = datetime.date(year=2012, month=12, day=31)
crimedata = crime.text
with open('Chicago crime df', 'w') as ofile:
    ofile.write(SESdata)
crime_df= pd.read_csv(os.path.join(b_path, 'Chicago crime df'))
'''


#setup:

#import packages

import pandas as pd
#import datetime #not currently useing

import pandas_datareader.data as web
from pandas_datareader import wb
import requests
#from bs4 import BeautifulSoup
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf 

b_path = '~Sarah/Desktop/Programming/Final_Project' #set path
os.chdir(os.path.expanduser(b_path)) #set working directory 


#list of url, file name tuples to be used when downloading data
urls = [('https://data.cityofchicago.org/api/views/kn9c-c2s2/rows.csv?accessType=DOWNLOAD', 'Chicago_SES.csv'), 
        ('https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/721/Green_Index__Land_Cover___Ave_Annual__v2.xlsx', 'Chicago_Green.xls'),
        ('https://data.cityofchicago.org/api/views/j6cj-r444/rows.csv?accessType=DOWNLOAD', 'Chicago_Death.csv'),
        ('https://data.cityofchicago.org/api/views/cjg8-dbka/rows.csv?accessType=DOWNLOAD', 'Chicago_health_cr.csv')]

#download the data
def download_data(url, filename):
    response = requests.get(url)
    if filename.endswith('.csv'):
        open_as = 'w'
        output = response.text
        #return open_as
    elif filename.endswith('.xls'):
        open_as = 'wb'
        output = response.content
        #return open_as
    else:
        return 'unexpected file type in download_data'
    
    with open(filename, open_as) as ofile:
        ofile.write(output)

#call the funtion
for url, filename in urls:
    download_data(url, filename)

#read in data
def read_data(path, filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(path, filename))
    elif filename.endswith('.xls'):
        df = pd.read_excel(os.path.join(path, filename))
    else:
        return 'unexpected file type in read_data'
    df.dropna(axis=1,inplace=True) #drop empty columns 
    return df


   # all_files = [os.path.join(path, f) for f in os.listdir(os.path.expanduser(path)) 
    #            if f.endswith('.csv') | f.endswith('.xls')]

#call the function
#list of df names: currently not using this
df_names = []
for url, filename in urls:
    df_names.append(filename[:-4] + '_df') 

df_contents = []
for url, filename in urls:
    df_contents.append(read_data(b_path, filename))


#I can't figure out how to read the df in with a nave an dthen also use that name
#bc I do have to do different things with each df

#so I'm gona hard code for now

#name the dfs
SESdf = df_contents[0]
green_df = df_contents[1]
death_df = df_contents[2]
healthcr_df = df_contents[3]

#the origional dataframe: extra step
#full origioanal df the load it back in from my computor 
#..... but don't worry about it. 

#read in data and organize:
#datasets: SES, greenspace, Avg anual deaths and cause of death, health centers 

#Chicago SES metrics by community area 2008 – 2012
SES = requests.get('https://data.cityofchicago.org/api/views/kn9c-c2s2/rows.csv?accessType=DOWNLOAD')
SESdata = SES.text
with open('Chicago SES df', 'w') as ofile:
    ofile.write(SESdata)

SESdf = pd.read_csv(os.path.join(b_path, 'Chicago SES df'))
#SESdf.columns
SESdf.rename(columns = {'PERCENT HOUSEHOLDS BELOW POVERTY':'PER_HOUSEHOLDS_BELOW_POVERTY', 'HARDSHIP INDEX':'HARDSHIP_INDEX'}, inplace=True)


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


#Average anual deaths and cause of death by community area 2006-2010
death = requests.get('https://data.cityofchicago.org/api/views/j6cj-r444/rows.csv?accessType=DOWNLOAD')
data = death.text
#data
with open('chicago_death_df', 'w') as ofile:
    ofile.write(data)

death_df = pd.read_csv(os.path.join(b_path, 'chicago_death_df'))
#death_df.head()
#death_df['Community Area Name']
#death_df.shape #(1404, 17) -> ,9)
death_df.dropna(axis=1,inplace=True) #drop empty columns 
#death_df.columns
death_df.rename(columns = {'Community Area': 'Community Area Number'}, inplace=True)
#death_df.index
avg_an_death = death_df.pivot(index = 'Community Area Number', columns='Cause of Death', values='Average Annual Deaths 2006 - 2010')
avg_an_death.drop(0, axis = 0, inplace = True) #drop the Chicago Total
avg_an_death.rename(columns = {'Suicide (intentional self-harm)':'Suicide', 'Diabetes-related':'Diabetes_related', 'All Causes': 'All_Causes'}, inplace = True)
#avg_an_death.columns
#avg_an_death.shape
#avg_an_death.head()

#health centers by community area 
healthcr = requests.get('https://data.cityofchicago.org/api/views/cjg8-dbka/rows.csv?accessType=DOWNLOAD') 
healthcrdata = healthcr.text
with open('chicago_health_centers_df', 'w') as ofile:
    ofile.write(healthcrdata)

healthcr_df = pd.read_csv(os.path.join(b_path, 'chicago_health_centers_df'))
#healthcr_df.head()
#healthcr_df.columns
#healthcr_df.shape
#healthcr_df.dtypes

#nieve fix for counting health centers per community area 
healthcr_df['count_of_health_crs'] = 1 
healthcr_df

count_of_crs = healthcr_df.groupby('Community Areas').sum().reset_index()
count_of_crs.columns

#this way didn't work for my purposes, but I think it is the way to do it
#get the count of how many health centers are in ea community area
#count_centers = healthcr_df.pivot_table(index=['Community Areas'], aggfunc='size')
#count_centers
#cite https://datatofish.com/count-duplicates-pandas/
#count_centers.dtypes

count_of_crs.drop(columns = ['Boundaries - ZIP Codes', 'Zip Codes', 'Census Tracts', 'Wards', ':@computed_region_awaf_s7ux'], inplace = True)



#Summary statistics ? not like this
green_df['Ave_Annual_perc_green'].mean()
green_df['Ave_Annual_perc_green'].max()
green_df['Ave_Annual_perc_green'].min()
green_df['Ave_Annual_perc_green'].var()

SESdf.columns
SESdf['HARDSHIP_INDEX'].mean()
SESdf['HARDSHIP_INDEX'].max()
SESdf['HARDSHIP_INDEX'].min()
SESdf['HARDSHIP_INDEX'].var()

#Avg deaths all causes
avg_an_death['All_Causes'].mean()
avg_an_death['All_Causes'].max()
avg_an_death['All_Causes'].min()
avg_an_death['All_Causes'].var()

#suicide deaths 
avg_an_death['Suicide'].mean()
avg_an_death['Suicide'].max()
avg_an_death['Suicide'].min()
avg_an_death['Suicide'].var()

#diabetse deaths
avg_an_death['Diabetes_related'].mean()
avg_an_death['Diabetes_related'].max()
avg_an_death['Diabetes_related'].min()
avg_an_death['Diabetes_related'].var()

def merge_dfs():
    pass 
    #Merge datasets:
    #nameing convention: he'll use descriptive names for interpediate steps
    #then easy to use name for final df 

    SES_green = SESdf.merge(green_df, on='Community Area Number', how = 'inner')
    #SES_green.shape
    #SESdf.shape #lost one entry, totals
    #green_df.shape #lost no entries
        #SES_green.columns

    SES_green_death = SES_green.merge(avg_an_death, on='Community Area Number', how = 'inner')
    #SES_green_death.shape
    #SES_green_death.columns

    SES_green_death_healthcr = SES_green_death.merge(count_of_crs, left_on='Community Area Number', right_on='Community Areas', how = 'outer')
    #SES_green_death_healthcr.shape
    #SES_green_death_healthcr['count_of_health_crs']
    #SES_green_death_healthcr.columns

    return final df 

#fill in Nan with 0 (bc if not in the previous database then doesn't have a health center)
SES_green_death_healthcr.fillna(value = 0, inplace=True)
#cite: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
#SES_green_death_healthcr.iloc[1]
SES_green_death_healthcr.drop(columns = ['COMMUNITY AREA NAME', 'Year', 'Community Areas', 'All causes in females',
                                        'All causes in males','Injury, unintentional', 'Colorectal cancer', 
                                        'Breast cancer in females','Prostate cancer in males','Stroke (cerebrovascular disease)',
                                        'Lung cancer'], inplace = True)


#Explore covariate relationships:

#lets test if SES and greenspace are related:
#suspect not bc downtown, but in general in the US should be
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

#where is the most greenspace?
np.argmax(SES_green['Ave_Annual_perc_green'])
#cite https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
SES_green.iloc[11]

#the least?
np.argmin(SES_green['Ave_Annual_perc_green'])
SES_green.iloc[1]


#lets take a look
plt.plot(SES_green['HARDSHIP_INDEX'], SES_green['Ave_Annual_perc_green'], 'o')
#cite: https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html

#oh! almost all very near 0 and then big values at all levels
plt.plot(SES_green['HARDSHIP_INDEX'], np.log(SES_green['Ave_Annual_perc_green']), 'o')
#interesting, do I maybe need to log my greenspace variable so that can see variation within 
# the many low greenspce areas?


#I wonder if there's a relxn between suicide and greenspace, probably not but...
plt.plot(SES_green['HARDSHIP_INDEX'], avg_an_death['Suicide'], 'o')
plt.show()
#hmm, neg rlxn betw hardship and suicide
#lets see if significant:

model = smf.ols('Suicide ~ HARDSHIP_INDEX + Ave_Annual_perc_green' , data=SES_green_death)
result = model.fit()
result.summary()

plt.plot(SES_green['Ave_Annual_perc_green'], avg_an_death['Suicide'], 'o')
plt.show()

model = smf.ols('Suicide ~ HARDSHIP_INDEX + Ave_Annual_perc_green' , data=SES_green_death)
result = model.fit()
result.summary()


def plot():
    pass 
#work on the plot: #doesn't look like 77 values! Am I loseing the 0 values?
x = SES_green_death['Ave_Annual_perc_green']
y = SES_green_death['Suicide']
z = SES_green_death['Community Area Name']
fig, ax = plt.subplots()
ax.semilogx(x, y, 'o')
ax.set_ylabel('Average Anual Suicide Rate')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
#cite https://matplotlib.org/examples/pylab_examples/log_demo.html
#lable each point with Community Area Name
for i, txt in enumerate(z):
    ax.annotate(txt, (x[i], y[i]), horizontalalignment='center', verticalalignment='bottom')
#cite: https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
#https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/annotation_demo.html
plt.show()

'''
#Or, this way isn't working
x = SES_green_death['Ave_Annual_perc_green']
y = SES_green_death['Suicide']
z = SES_green_death['Community Area Name']
for i, n in enumerate(z):
    x = x[i]
    y = y[i]
    plt.scatter(x, y, marker='x', color='red')
    plt.text(x+0.3, y+0.3, n, fontsize=9)
plt.show()
#cite https://www.pythonmembers.club/2018/05/08/matplotlib-scatter-plot-annotate-set-text-at-label-each-point/
'''


#lets look at some other causes of death, I expect related to SES
plt.plot(SES_green['HARDSHIP_INDEX'], avg_an_death['Diabetes_related'], 'o')
plt.show()
#does not look related, which is a big surprise
model = smf.ols('Diabetes_related ~ HARDSHIP_INDEX + Ave_Annual_perc_green' , data=SES_green_death)
result = model.fit()
result.summary()
#something is off, expect SES to be predictive of diabetese deaths

model = smf.ols('All_Causes ~ HARDSHIP_INDEX + Ave_Annual_perc_green' , data=SES_green_death)
result = model.fit()
result.summary()

#lets see if there is a relationship between SES and health center placement
plt.plot(SES_green_death_healthcr['HARDSHIP_INDEX'], SES_green_death_healthcr['count_of_health_crs'], 'o')
plt.show()

model = smf.ols('count_of_health_crs ~ HARDSHIP_INDEX' , data=SES_green_death_healthcr)
result = model.fit()
result.summary()

#nope no relationship, good news


#adding into model
model = smf.ols('Suicide ~ HARDSHIP_INDEX + Ave_Annual_perc_green + count_of_health_crs' , data=SES_green_death_healthcr)
result = model.fit()
result.summary()

#lets see if health centers help explain diabetese deaths 
model = smf.ols('Diabetes_related ~ HARDSHIP_INDEX + Ave_Annual_perc_green + count_of_health_crs' , data=SES_green_death_healthcr)
result = model.fit()
result.summary()
#odd, maybe just the presence of a health center isn't enough to help




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


#unaffiliated citations 
#citation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html
#citation: https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/


