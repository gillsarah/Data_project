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

#need to:
# make plots prettier???
# move fn calls to end of file
# output the final df
# output plots
# output tables?????




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
'''
#notes from prof:
def download_prof():
    df1 = download_df1()

    resutls = []
    for d in [df2, df3, df4]:
        df = download_oterwise(d)
        results.append(d)
    df = pd.concat(results)
'''

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
    return df


def prof_help():
    pass
#parse before put in df_conents
#read the data in and have a conditional if file name == this
#here's some parseing
# read in
# parse
# make a list 
#fn to parse this df
#in a loop wehre you know file name: if 

#re-name community Area: for some reasion after pivot cant use Community Area to merge
#but Community Area Name works

def parse_death(death_df):
    death_df.rename(columns = {'Community Area': 'Community Area Number'}, inplace=True)
    avg_an_death = death_df.pivot(index = 'Community Area Number', columns='Cause of Death', values='Average Annual Deaths 2006 - 2010')
    avg_an_death.drop(0, axis = 0, inplace = True) #drop the Chicago Total
    return avg_an_death

def parse_healthcr(healthcr_df):
    healthcr_df['count_of_health_crs'] = 1 
    count_of_crs = healthcr_df.groupby('Community Areas').sum().reset_index()
    return count_of_crs

#the call
parsing_fn = {'Chicago_Death.csv':parse_death,
             'Chicago_health_cr.csv': parse_healthcr}
#didn't use: could to re-org the call of read and parse
#    parsing_fn[filename](df) #calling a value from the dict, that value is a fn, call the fn
#parse_death #shows have an object, can call

df_contents = []
for url, filename in urls:
    df = read_data(b_path, filename)
    if filename == 'Chicago_Death.csv':
        df_contents.append(parse_death(df))
    elif filename == 'Chicago_health_cr.csv':
        df_contents.append(parse_healthcr(df))
    else:
        df_contents.append(df)

'''
df_contents[0].shape #78,9
df_contents[1].shape #(77,27)
df_contents[2].shape #77,18 #check
df_contents[3].shape #47,7 #check
df_contents[3].head()
df_contents[3].columns
df_contents[2].columns #same but odd!
'''


'''
#don't
#call the function
#list of df names: currently not using this
df_names = []
for url, filename in urls:
    df_names.append(filename[:-4] + '_df') 

df2_contents = []
for url, filename in urls:
    df2_contents.append(read_data(b_path, filename))

'''

'''
#drop na cols
for df in df_contents:
    df.dropna(axis=1,inplace=True) #drop empty columns 
'''  
#so I'm gona hard code for now

#name the dfs
#SES_df = df2_contents[0]
#green_df = df2_contents[1]
#death_df = df2_contents[2]
#healthcr_df = df2_contents[3]

'''
#death_df: rename cols, reshape, drop totals col 
#I could skip rename if do merge on left...
death_df.rename(columns = {'Community Area': 'Community Area Number'}, inplace=True)
avg_an_death = death_df.pivot(index = 'Community Area Number', columns='Cause of Death', values='Average Annual Deaths 2006 - 2010')
avg_an_death.drop(0, axis = 0, inplace = True) #drop the Chicago Total
#avg_an_death.columns
#avg_an_death.shape
#healthcr_df: add count col
#nieve fix for counting health centers per community area 
healthcr_df['count_of_health_crs'] = 1 
count_of_crs = healthcr_df.groupby('Community Areas').sum().reset_index()
count_of_crs.columns
'''

#this way didn't work for my purposes, but I think it is the way to do it
#get the count of how many health centers are in ea community area
#count_centers = healthcr_df.pivot_table(index=['Community Areas'], aggfunc='size')
#count_centers
#cite https://datatofish.com/count-duplicates-pandas/
#count_centers.dtypes

#for some reason the merge between green and avg_an_death works if I re-name Community Area but not if I don't
def merge_dfs(SES_df,green_df,avg_an_death,count_of_crs): 
    #Merge datasets:    
    SES_green = SES_df.merge(green_df, left_on='Community Area Number', right_on = 'Geo_ID', how = 'inner')

    SES_green_death = SES_green.merge(avg_an_death, on='Community Area Number', how = 'inner')

    SES_green_death_healthcr = SES_green_death.merge(count_of_crs, left_on='Community Area Number', right_on='Community Areas', how = 'outer')
    #fill in Nan with 0 (bc if not in the previous database then doesn't have a health center)
    SES_green_death_healthcr['count_of_health_crs'].fillna(value = 0, inplace=True)
    #cite: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
    SES_green_death_healthcr.dropna(axis=1,inplace=True)
    
    #drop columns that do not provide useful information/may not apply to all entries in the row after 
    # the merge (e.g. SubCategory or Map_Key from green_df), or duplicates eg Geo_ID
    SES_green_death_healthcr.drop(columns = ['Geo_Group', 'Geo_ID', 'Category', 'SubCategory',
                                            'Geography', 'Map_Key'], inplace = True)

    return SES_green_death_healthcr 

#old way
#full_df = merge_dfs(SES_df,green_df,avg_an_death,count_of_crs)
#new way
full_df_2 = merge_dfs(df_contents[0], df_contents[1], df_contents[2], df_contents[3])
#full_df.columns
#full_df.shape #(77,32) #38 when don't drop first #77,31? 
full_df_2.shape #77,31 same as most recent run of other method

#reduce number of cols for convinience, drop ones outside of investigation
use_df = full_df_2.drop(columns = ['PERCENT OF HOUSING CROWDED',  'PERCENT AGED 16+ UNEMPLOYED',
                        'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA', 'PERCENT AGED UNDER 18 OR OVER 64', 
                        'Year','Injury, unintentional', 'All causes in females', 'All causes in males', 
                        'Alzheimers disease', 'Assault (homicide)', 'Breast cancer in females', 'Cancer (all sites)', 
                        'Colorectal cancer', 'Breast cancer in females', 'Prostate cancer in males', 'Firearm-related',
                        'Kidney disease (nephritis, nephrotic syndrome and nephrosis)', 'Liver disease and cirrhosis', 
                        'Lung cancer','Indicator'])


#use_df.columns
def re_name(use_df):
    use_df.rename(...)
    col_name.title()
    col_name.replace(" ", "_")
    #make cammal case and _
    #if () cutt after(
    pass
#cite https://stackoverflow.com/questions/8347048/how-to-convert-string-to-title-case-in-python

#remane for use in ols
use_df.rename(columns = {'Ave_Annual_Number': 'Ave_Annual_perc_green', 'PER CAPITA INCOME ': 'Per_Capita_Income', 'PERCENT HOUSEHOLDS BELOW POVERTY':'Perc_Households_Below_Poverty', 
                        'HARDSHIP INDEX':'HARDSHIP_INDEX','Suicide (intentional self-harm)':'Suicide', 
                        'Diabetes-related':'Diabetes_related', 'Stroke (cerebrovascular disease)': 'Stroke',
                        'Coronary heart disease': 'Coronary_heart_disease', 'COMMUNITY AREA NAME':'Community Area Name'}, inplace=True) 


#Summary statistics
def summary_stats(df):
    summary = df.describe()
    summary.drop(columns = ['Community Area Number'], inplace = True)
    summary = summary.transpose()
    return summary
#cite https://stackoverflow.com/questions/33889310/r-summary-equivalent-in-numpy 
#50% have 0 green. 75 percentile is <1
#call
summary_stats(use_df)
#    return tempdf.groupby('COMMUNITY AREA NAME').mean()

#'Ave_Annual_perc_green', 'HARDSHIP_INDEX', 'All_Causes'...
#summary_stat_tab = summary_stats(use_df)

#where is the most greenspace?
def max_green(use_df):
    i = np.argmax(use_df['Ave_Annual_perc_green'])
    max_green = use_df.iloc[i]
    return max_green

def min_green(use_df):
    min_green_list = []
    for i in range(len(use_df['Ave_Annual_perc_green'])):
        if use_df['Ave_Annual_perc_green'][i] == 0:
            min_green_list.append(i)
        else:
            pass
    for j in min_green_list:
        return use_df.iloc[j]

#cite https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list

#analysis
def ols(use_df):
    ys = ['Stroke', 'Coronary_heart_disease', 'Diabetes_related', 'Suicide']
    for i in ys:
        print('Independent Variable: ' + i)
        m = smf.ols(i + '~ Ave_Annual_perc_green + HARDSHIP_INDEX + count_of_health_crs' , data = use_df)
        result = m.fit()
        print(result.summary())
        print( )


#Limitations in variation in greenspace (nearly half of the dataset is at 0 avg an perc green!)
#look at a restricted sample, exclude 0 values.
some_green = use_df['Ave_Annual_perc_green'] != 0
some_green_df = use_df[some_green]

zero_green = use_df['Ave_Annual_perc_green'] == 0
zero_green_df = use_df[zero_green]
#cite https://chrisalbon.com/python/data_wrangling/pandas_selecting_rows_on_conditions/

#call
ols(use_df)
ols(some_green_df)

#even more sig if exclude 0s
#call
summary_stats(some_green_df)



#lets see if health centers help explain diabetese deaths 
#odd, maybe just the presence of a health center isn't enough to help

#Explore covariate relationships:

#lets test if SES and greenspace are related:
#suspect not bc downtown, but in general in the US should be
test_model = smf.ols('HARDSHIP_INDEX ~ Ave_Annual_perc_green' , data=use_df)
result = test_model.fit()
result.summary()

test_model = smf.ols('PER_HOUSEHOLDS_BELOW_POVERTY ~ Ave_Annual_perc_green' , data=use_df)
result = test_model.fit()
result.summary()

#test that I am running this right with a for certain correlation
test_model = smf.ols('PER_HOUSEHOLDS_BELOW_POVERTY ~ HARDSHIP_INDEX' , data=use_df)
result = test_model.fit()
result.summary()
#yup

#lets take a look
plt.plot(use_df['HARDSHIP_INDEX'], use_df['Ave_Annual_perc_green'], 'o')
#cite: https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html

#oh! almost all very near 0 and then big values at all levels
plt.plot(use_df['HARDSHIP_INDEX'], np.log(use_df['Ave_Annual_perc_green']), 'o')
#interesting, do I maybe need to log my greenspace variable so that can see variation within 
# the many low greenspce areas?


#I wonder if there's a relxn between suicide and greenspace, probably not but...
plt.plot(use_df['HARDSHIP_INDEX'], use_df['Suicide'], 'o')
plt.show()
#hmm, neg rlxn betw hardship and suicide
#lets see if significant:

model = smf.ols('Suicide ~ HARDSHIP_INDEX + Ave_Annual_perc_green' , data=use_df)
result = model.fit()
result.summary()

plt.plot(use_df['Ave_Annual_perc_green'], use_df['Suicide'], 'o')
plt.show()

#try a bar
#re-shape df index as perc green
temp_df = some_green_df.set_index('Ave_Annual_perc_green')
temp_df.head()

temp_df['Suicide'].plot(kind='bar', subplots=True, figsize=(10,10))
plt.show()
#ok, works, but not informative (would need to put a lot of work in)
#maybe better spent making scatters pretty

model = smf.ols('Suicide ~ Ave_Annual_perc_green' , data=use_df)
result = model.fit().params
# generate x-values for your regression line (two is sufficient)
x = np.arange(1, 3)

# scatter-plot data
ax = use_df.plot(x='Ave_Annual_perc_green', y='Suicide', kind='scatter')

# plot regression line on the same axes, set x-axis limits
ax.plot(x, result.const + result.Ave_Annual_perc_green * x)
ax.set_xlim([1, 2])

def plot_model():
    pass

import seaborn as sns
import matplotlib.ticker as ticker

sns.regplot(x='Ave_Annual_perc_green', y='Suicide', data=use_df)
fig, ax = sns.regplot(x='Ave_Annual_perc_green', y='Coronary_heart_disease', data=use_df)
ax.set_ylabel('Average Anual Deaths by Stroke')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
plt.show()

sns.regplot(x='Ave_Annual_perc_green', y='Suicide', data=some_green_df)
#cite for plot regn
#https://stackoverflow.com/questions/42261976/how-to-plot-statsmodels-linear-regression-ols-cleanly


#cite for seaborn 
#https://www.datacamp.com/community/tutorials/seaborn-python-tutorial#show
fig, ax = plt.subplots()
sns.regplot(use_df['Ave_Annual_perc_green'], use_df['Coronary_heart_disease'],'o')
ax.set_xscale('symlog')
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.set_ylabel('Average Anual Deaths by Heart Disease')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
plt.show()

fig, ax = plt.subplots()
sns.regplot(use_df['Ave_Annual_perc_green'], use_df['Suicide'],'o')
ax.set_xscale('symlog')
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.set_ylabel('Average Anual Deaths by Suicide')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
plt.show()

#another way to try: does not work!
model = smf.ols('Suicide ~ Ave_Annual_perc_green' , data=use_df)
result = model.fit()

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(result, 0, ax=ax)
plt.show()

#technically interesting but prob not gonna use
#cite for the belwo: https://www.statsmodels.org/dev/examples/notebooks/generated/regression_plots.html 
#shows the influence of ea point on the sloap fo the line, cool but not gonna include
model = smf.ols('Suicide ~ HARDSHIP_INDEX + Ave_Annual_perc_green' , data=use_df)
result = model.fit()
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(result, ax=ax, criterion="cooks")
plt.show()

fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.plot_partregress('Suicide ', 'HARDSHIP_INDEX', ['HARDSHIP_INDEX',  'Ave_Annual_perc_green'], data=use_df, ax=ax)
plt.show()

fix, ax = plt.subplots(figsize=(12,14))
fig = sm.graphics.plot_partregress('Suicide ', 'HARDSHIP_INDEX', ['Ave_Annual_perc_green'], data=use_df, ax=ax)
plt.show()

fix, ax = plt.subplots(figsize=(12,14))
fig = sm.graphics.plot_partregress('Suicide', 'Ave_Annual_perc_green', ['HARDSHIP_INDEX'], data=use_df, ax=ax)
plt.show()

#end above section 

#this is very funny! And I have no idea what it is doing
plt.plot(use_df.Suicide, result.fittedvalues, 'r')
#cite https://stackoverflow.com/questions/48682407/r-abline-equivalent-in-python

def plot():
    pass 


#best yet, other attempts below
fig, ax = plt.subplots()
ax.plot(use_df['Ave_Annual_perc_green'], use_df['Suicide'],'o')
ax.set_xscale('symlog')
ax.set_ylabel('Average Anual Suicide Rate')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
plt.show()
#cite https://stackoverflow.com/questions/16904755/logscale-plots-with-zero-values-in-matplotlib

fig, ax = plt.subplots()
ax.plot(use_df['Ave_Annual_perc_green'], use_df['Diabetes_related'],'o')
ax.set_xscale('symlog')
#ax.get_xaxis().get_major_formatter().set_scientific(False)
ax.xaxis.set_major_locator(plt.MaxNLocator(10)) #still sci notation
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.set_ylabel('Average Anual Diabetese Relaed Deaths')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
plt.show()
#citation https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html
#https://matplotlib.org/3.1.0/gallery/ticks_and_spines/tick-locators.html
#https://matplotlib.org/3.1.0/gallery/ticks_and_spines/tick-formatters.html

fig, ax = plt.subplots()
ax.plot(use_df['Ave_Annual_perc_green'], use_df['Stroke'],'o')
ax.set_xscale('symlog')
ax.ticklabel_format(axis = 'both', style = 'plain')
ax.set_ylabel('Average Anual Deaths by Stroke')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
plt.show()
#not working: https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html

#winner
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(use_df['Ave_Annual_perc_green'], use_df['Coronary_heart_disease'],'o')
ax.set_xscale('symlog')
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.set_ylabel('Average Anual Deaths by Heart Disease')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
plt.show()


fig, ax = plt.subplots()
ax.plot(use_df['Ave_Annual_perc_green'], use_df['All Causes'],'o')
ax.set_xscale('symlog')
ax.xaxis.set_major_locator(plt.FixedLocator([0,1,2,3,4,5,6,7,8,9,10,15]))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.set_ylabel('Average Anual Deaths')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
plt.show()




#work on the plot: #doesn't look like 77 values! Am I loseing the 0 values?
#want to make another plot over this one with the 0s?
#want to resize the points for income
def this_plot():
    pass
x = use_df['Ave_Annual_perc_green']
y = use_df['Suicide']
z = use_df['Community Area Name']
s = use_df['Per_Capita_Income']
a_list = use_df['Per_Capita_Income']/1000
         

#yah!
ax = use_df.plot(kind='scatter', x='Ave_Annual_perc_green', y='Suicide', s = a_list, label = 'Income')
#cite https://github.com/pandas-dev/pandas/issues/16827
#ax.plot(use_df['Ave_Annual_perc_green'], use_df['Suicide'], 'o' )
#ax.semilogx(x, y,'o')
ax.legend()
ax.set_ylabel('Average Anual Suicide Rate')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
#cite https://matplotlib.org/examples/pylab_examples/log_demo.html (not longre useing)
#lable point with Community Area Name, only if enough green (for viewability)

for i, txt in enumerate(z): 
    if x[i] >= 2:
        ax.annotate(txt, (x[i], y[i]), horizontalalignment='center', verticalalignment='bottom')
    else:
        pass
#cite: https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
#https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/annotation_demo.html
plt.show()

zero_green = use_df['Ave_Annual_perc_green'] == 0
zero_green_df = use_df[zero_green]
#cite https://chrisalbon.com/python/data_wrangling/pandas_selecting_rows_on_conditions/

#When you restrict sample to low greenspace, get slight upward curve 

low_green = use_df['Ave_Annual_perc_green'] <=2
low_green_df = use_df[low_green]


z2 = use_df['Community Area Name']
a_list2 = some_green_df['Per_Capita_Income']/1000

ax = some_green_df.plot(kind='scatter', x= 'Ave_Annual_perc_green', y= 'Suicide', s = a_list2, label = 'Income')
#cite https://github.com/pandas-dev/pandas/issues/16827
#ax.plot(use_df['Ave_Annual_perc_green'], use_df['Suicide'], 'o' )
#ax.semilogx(x, y,'o')
ax.legend()
ax.set_ylabel('Average Anual Suicide Rate')
ax.set_xlabel('Percent of Area that is Green')
#cite https://matplotlib.org/examples/pylab_examples/log_demo.html (not longre useing)
#lable point with Community Area Name, only if enough green (for viewability)

for i, txt in enumerate(z2): 
    ax.annotate(txt, (x[i], y[i]), horizontalalignment='center', verticalalignment='bottom')
#cite: https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
#https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/annotation_demo.html
plt.show()

sns.regplot(low_green_df['Ave_Annual_perc_green'], low_green_df['Coronary_heart_disease'],'o')
sns.regplot(low_green_df['Ave_Annual_perc_green'], low_green_df['Suicide'],'o')
sns.regplot(use_df['Ave_Annual_perc_green'], use_df['Suicide'],'o')

#whoops, copy paste and got rid of z1 and list1
ax = low_green_df.plot(kind='scatter', x= 'Ave_Annual_perc_green', y= 'Suicide', s = a_list1, label = 'Income')
#cite https://github.com/pandas-dev/pandas/issues/16827
#ax.plot(use_df['Ave_Annual_perc_green'], use_df['Suicide'], 'o' )
#ax.semilogx(x, y,'o')
ax.legend()
ax.set_ylabel('Average Anual Suicide Rate')
ax.set_xlabel('Percent of Area that is Green (Avearge Anual)')
#cite https://matplotlib.org/examples/pylab_examples/log_demo.html (not longre useing)
#lable point with Community Area Name, only if enough green (for viewability)

for i, txt in enumerate(z1): 
    ax.annotate(txt, (x[i], y[i]), horizontalalignment='center', verticalalignment='bottom')
#cite: https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
#https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/annotation_demo.html
plt.show()

#use size
a_list = use_df['HARDSHIP_INDEX']
use_df.plot(kind='scatter', x='Ave_Annual_perc_green', y='Suicide', s = a_list )         
plt.show()

fig, ax = plt.subplots()
ax.plot(use_df['Per_Capita_Income'], use_df['Suicide'], 'o')
ax.semilogx(use_df['Ave_Annual_perc_green'], use_df['Suicide'], 'o' )
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



model = smf.ols('Diabetes_related ~ Per_Capita_Income' , data=use_df)
result = model.fit()
result.summary()

model = smf.ols('All_Causes ~ HARDSHIP_INDEX + Ave_Annual_perc_green' , data=SES_green_death_healthcr)
result = model.fit()
result.summary()

#lets see if there is a relationship between SES and health center placement
plt.plot(SES_green_death_healthcr['HARDSHIP_INDEX'], SES_green_death_healthcr['count_of_health_crs'], 'o')
plt.show()

model = smf.ols('count_of_health_crs ~ HARDSHIP_INDEX' , data=SES_green_death_healthcr)
result = model.fit()
result.summary()

#nope no relationship, good news







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


