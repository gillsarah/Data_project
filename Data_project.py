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

#need to move fn calls to end of file



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
    #df.dropna(axis=1,inplace=True) #drop empty columns 
    return df
#for some reason dropping na in this step dropps the Community Area Name col!!!???
def help1_resolved():
    'for some reason dropna is dropping non-empty cols'
    'I think it might drop if any NA not just if all NA!'
    'Neer mind, fixed it in the merge step!'
    pass
   # all_files = [os.path.join(path, f) for f in os.listdir(os.path.expanduser(path)) 
    #            if f.endswith('.csv') | f.endswith('.xls')]


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
def parse_death(death_df):
    avg_an_death = death_df.pivot(index = 'Community Area Number', columns='Cause of Death', values='Average Annual Deaths 2006 - 2010')
    avg_an_death.drop(0, axis = 0, inplace = True) #drop the Chicago Total
    #do stuff
    return avg_an_death
def parse_healthcr(healthcr_df):
    healthcr_df['count_of_health_crs'] = 1 
    count_of_crs = healthcr_df.groupby('Community Areas').sum().reset_index()
    #do stuff
    return count_of_crs
#the call
parsing_fn = {'Chicago_Death.csv':parse_death,
             'Chicago_health_cr.csv': parse_healthcr}

for url, filename in urls:
    df = read_data(b_path, filename)
    if filename == 'Chicago_Death.csv':
        parse_death(df)
    elif filename == 'Chicago_health_cr.csv':
        parse_healthcr(df)
    else:
        return df

    


    parsing_fn[filename](df) #calling a value from the dict, that value is a fn, call the fn

parse_death #shows have an object, can call


#don't
#call the function
#list of df names: currently not using this
df_names = []
for url, filename in urls:
    df_names.append(filename[:-4] + '_df') 

df_contents = []
for url, filename in urls:
    df_contents.append(read_data(b_path, filename))



'''
#drop na cols
for df in df_contents:
    df.dropna(axis=1,inplace=True) #drop empty columns 
'''  
#I can't figure out how to read the df in with a nave an dthen also use that name
#I could concatinate these list, but then how do I work with them?
#bc I do have to do different things with each df

#so I'm gona hard code for now

#name the dfs
SES_df = df_contents[0]
green_df = df_contents[1]
death_df = df_contents[2]
healthcr_df = df_contents[3]

'''
SES_df.columns
#droping na on SES drops the community area for some reason!
green_df.columns
green_df.dropna(axis=1,inplace=True)
#fine
death_df.columns
death_df.dropna(axis=1,inplace=True)
#fine
healthcr_df.columns
healthcr_df['Community Areas']
healthcr_df['Community Area (#)']
healthcr_df.dropna(axis=1,inplace=True)
#lose Community Areas!
'''


#green_df: rename cols,  re-format Community Area Name colum (was Geo_Group)
#I could skip rename if do merge on left... and rename perc green after merge
green_df.rename(columns = {'Geo_ID':'Community Area Number', 
                            'Ave_Annual_Number': 'Ave_Annual_perc_green'}, inplace=True)
#I could skip this step! Don't use this new col anymore!
tempcol = green_df['Geo_Group'].str.split("-", expand = True)
green_df['temp'] = tempcol[0]
green_df['Community Area Name'] = tempcol[1]
#citation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html
#citation: https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/




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


#this way didn't work for my purposes, but I think it is the way to do it
#get the count of how many health centers are in ea community area
#count_centers = healthcr_df.pivot_table(index=['Community Areas'], aggfunc='size')
#count_centers
#cite https://datatofish.com/count-duplicates-pandas/
#count_centers.dtypes

def help_resolved():
    #when I merge by both: on=['Community Area Number', 'Community Area Name']
    #I loose 1 col and 5 crows
    #its bc formatting differnece!
    pass
    #green_df['Community Area Name'] == SES_df['Community Area Name']
    #had to undue the rename! if want to work with it rename again

def merge_dfs(SES_df,green_df,avg_an_death,count_of_crs): 
    #Merge datasets:
    #nameing convention: he'll use descriptive names for interpediate steps
    #then easy to use name for final df 
    
    SES_green = SES_df.merge(green_df, on='Community Area Number', how = 'inner')
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
    #fill in Nan with 0 (bc if not in the previous database then doesn't have a health center)
    SES_green_death_healthcr['count_of_health_crs'].fillna(value = 0, inplace=True)
    #cite: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
    SES_green_death_healthcr.dropna(axis=1,inplace=True)
    
    #drop columns that do not provide useful information/may not apply to all entries in the row after 
    # the merge (e.g. SubCategory or Map_Key from green_df), , or provides duplicate information but 
    # with occasional fomatting differences ('Community Area Name')
    SES_green_death_healthcr.drop(columns = ['Geo_Group', 'temp', 'Category', 'SubCategory',
                                            'Geography', 'Map_Key', 'Community Area Name', ], inplace = True)

    return SES_green_death_healthcr 

full_df = merge_dfs(SES_df,green_df,avg_an_death,count_of_crs)
#full_df.columns
full_df.shape #(77,32) #38 when don't drop first


#reduce number of cols, drop ones outside of investigation
use_df = full_df.drop(columns = ['PERCENT OF HOUSING CROWDED',  'PERCENT AGED 16+ UNEMPLOYED',
                        'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA', 'PERCENT AGED UNDER 18 OR OVER 64', 
                        'Year','Injury, unintentional', 'All causes in females', 'All causes in males', 
                        'Alzheimers disease', 'Assault (homicide)', 'Breast cancer in females', 'Cancer (all sites)', 
                        'Colorectal cancer', 'Breast cancer in females', 'Prostate cancer in males', 'Firearm-related',
                        'Kidney disease (nephritis, nephrotic syndrome and nephrosis)', 'Liver disease and cirrhosis', 
                        'Lung cancer','Indicator'])

#use_df.columns

'''
SES_green_death_healthcr['COMMUNITY AREA NAME'][5]
SES_green_death_healthcr['Community Area Name'][5]
print(SES_green_death_healthcr['COMMUNITY AREA NAME'][75])
print(SES_green_death_healthcr['Community Area Name'][75])
print(SES_green_death_healthcr['COMMUNITY AREA NAME'][72])
print(SES_green_death_healthcr['Community Area Name'][72])

temp_list = []
for i in range(len(SES_green_death_healthcr['COMMUNITY AREA NAME'])):
    temp_list.append(SES_green_death_healthcr['COMMUNITY AREA NAME'][i] == SES_green_death_healthcr['Community Area Name'][i])
'''

#remane for use in ols
use_df.rename(columns = {'PER CAPITA INCOME ': 'Per_Capita_Income', 'PERCENT HOUSEHOLDS BELOW POVERTY':'Perc_Households_Below_Poverty', 
                        'HARDSHIP INDEX':'HARDSHIP_INDEX','Suicide (intentional self-harm)':'Suicide', 
                        'Diabetes-related':'Diabetes_related', 'Stroke (cerebrovascular disease)': 'Stroke',
                        'Coronary heart disease': 'Coronary_heart_disease', 'COMMUNITY AREA NAME':'Community Area Name'}, inplace=True) 


#Summary statistics
def summary_stats(df):
    pass

#    return tempdf.groupby('COMMUNITY AREA NAME').mean()

#'Ave_Annual_perc_green', 'HARDSHIP_INDEX', 'All_Causes'...
#summary_stat_tab = summary_stats(use_df)



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

#where is the most greenspace?
np.argmax(use_df['Ave_Annual_perc_green'])
#cite https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
use_df.iloc[11]

#the least?
np.argmin(use_df['Ave_Annual_perc_green'])
use_df.iloc[1]


#lets take a look
plt.plot(use_df['HARDSHIP_INDEX'], use_df['Ave_Annual_perc_green'], 'o')
#cite: https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html

#oh! almost all very near 0 and then big values at all levels
plt.plot(use_df['HARDSHIP_INDEX'], np.log(use_df['Ave_Annual_perc_green']), 'o')
#interesting, do I maybe need to log my greenspace variable so that can see variation within 
# the many low greenspce areas?


#I wonder if there's a relxn between suicide and greenspace, probably not but...
plt.plot(use_df['HARDSHIP_INDEX'], avg_an_death['Suicide'], 'o')
plt.show()
#hmm, neg rlxn betw hardship and suicide
#lets see if significant:

model = smf.ols('Suicide ~ HARDSHIP_INDEX + Ave_Annual_perc_green' , data=use_df)
result = model.fit()
result.summary()

plt.plot(use_df['Ave_Annual_perc_green'], avg_an_death['Suicide'], 'o')
plt.show()

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

sns.regplot(x='Ave_Annual_perc_green', y='Suicide', data=use_df)
fig, ax = sns.regplot(x='Ave_Annual_perc_green', y='Coronary_heart_disease', data=use_df)
ax.set_ylabel('Average Anual Deaths by Stroke')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
plt.show()

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

#another way to try
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
import matplotlib.ticker as ticker
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


zero_green = use_df['Ave_Annual_perc_green'] == 0
zero_green_df = use_df[zero_green]
#cite https://chrisalbon.com/python/data_wrangling/pandas_selecting_rows_on_conditions/


#work on the plot: #doesn't look like 77 values! Am I loseing the 0 values?
#want to make another plot over this one with the 0s?
#want to resize the points for income
x = use_df['Ave_Annual_perc_green']
y = use_df['Suicide']
z = use_df['Community Area Name']
s = use_df['Per_Capita_Income']
fig, ax = plt.subplots()
ax.plot(zero_green_df['Ave_Annual_perc_green'], zero_green_df['Suicide'], 'o' )
ax.semilogx(x, y,'o')

ax.set_ylabel('Average Anual Suicide Rate')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
#cite https://matplotlib.org/examples/pylab_examples/log_demo.html
#lable each point with Community Area Name
for i, txt in enumerate(z):
    ax.annotate(txt, (x[i], y[i]), horizontalalignment='center', verticalalignment='bottom')
#cite: https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
#https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/annotation_demo.html
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


#lets look at some other causes of death, I expect related to SES
plt.plot(SES_green_death_healthcr['HARDSHIP_INDEX'], avg_an_death['Diabetes_related'], 'o')
plt.show()
#does not look related, which is a big surprise
model = smf.ols('Diabetes_related ~ HARDSHIP_INDEX + Ave_Annual_perc_green' , data=SES_green_death_healthcr)
result = model.fit()
result.summary()
#something is off, expect SES to be predictive of diabetese deaths
model = smf.ols('Diabetes_related ~ Per_Capita_Income + Ave_Annual_perc_green' , data=use_df)
result = model.fit()
result.summary()

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


