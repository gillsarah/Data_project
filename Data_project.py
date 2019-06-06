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

#CMAP Community snapshot


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
#but Community_Area_Name works

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
def re_name(df):
    df.columns = [c.replace("Ave_Annual_Number ", "Ave_Annual_Perc_Green") for c in df.columns]
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df.columns = [c.replace("-", "_") for c in df.columns]
    df.columns = [c.title() for c in df.columns]
    df.columns = [c.split(sep = ' (')[0] for c in df.columns]

re_name(use_df)
re_name(some_green_df)
    #make cammal case and _
    #if () cutt after(

#cite https://stackoverflow.com/questions/8347048/how-to-convert-string-to-title-case-in-python
#cite https://stackoverflow.com/questions/39741429/pandas-replace-a-character-in-all-column-names



#remane for use in ols
#use_df.rename(columns = {'Ave_Annual_Number': 'Ave_Annual_Perc_Green', 'PER CAPITA INCOME ': 'Per_Capita_Income', 'PERCENT HOUSEHOLDS BELOW POVERTY':'Perc_Households_Below_Poverty', 
#                        'HARDSHIP INDEX':'Hardship_Index','Suicide (intentional self-harm)':'Suicide', 
 #                       'Diabetes-related':'Diabetes_related', 'Stroke (cerebrovascular disease)': 'Stroke',
  #                      'Coronary heart disease': 'Coronary_Heart_Disease', 'Community_Area_Name':'Community_Area_Name'}, inplace=True) 


#Summary statistics
#may need to re-name col to drop (once I use the fn)
def summary_stats(df):
    summary = df.describe()
    summary.drop(columns = ['Community_Area_Number'], inplace = True)
    summary = summary.transpose()
    return summary
#cite https://stackoverflow.com/questions/33889310/r-summary-equivalent-in-numpy 


#Limitations in variation in greenspace (nearly half of the dataset is at 0 avg an perc green!)
#look at a restricted sample, exclude 0 values.
some_green = use_df['Ave_Annual_Perc_Green'] != 0
some_green_df = use_df[some_green]

zero_green = use_df['Ave_Annual_Perc_Green'] == 0
zero_green_df = use_df[zero_green]
#cite https://chrisalbon.com/python/data_wrangling/pandas_selecting_rows_on_conditions/

#where is the most greenspace?
def max_green(use_df):
    i = np.argmax(use_df['Ave_Annual_Perc_Green'])
    max_green = use_df.iloc[i]
    return max_green

def min_green(use_df):
    min_green_list = []
    for i in range(len(use_df['Ave_Annual_Perc_Green'])):
        if use_df['Ave_Annual_Perc_Green'][i] == 0:
            min_green_list.append(i)
        else:
            pass
    for j in min_green_list:
        return use_df.iloc[j]

#cite https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list

#analysis
def ols(use_df):
    ys = ['Stroke', 'Coronary_Heart_Disease', 'Diabetes_Related', 'Suicide']
    for i in ys:
        print('Independent Variable: ' + i)
        m = smf.ols(i + '~ Ave_Annual_Perc_Green + Hardship_Index + Count_Of_Health_Crs' , data = use_df)
        result = m.fit()
        print(result.summary())
        print( )



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
def covt_check(df, y_string, x_string):
    print(y_string + ' on ' + x_string)
    test_model = smf.ols(y_string + '~' + x_string , data = df)
    result = test_model.fit()
    print(result.summary())
    print( )




#lets take a look
#plot covariate relationships 
def covt_plots(df,y00,x00,y01,x01,y10,x10,y11,x11):
    fig, ax = plt.subplots(2,2, figsize= (10,8))
    plt.tight_layout(pad= 4, w_pad=4, h_pad=4)
    ax[0][0].plot(df[y00], df[x00], 'o')
    ax[0][0].set_ylabel(y00)
    ax[0][0].set_xlabel(x00)
    ax[0][1].plot(df[y01], df[x01], 'o')
    ax[0][1].set_ylabel(y01)
    ax[0][1].set_xlabel(x01)
    ax[1][0].plot(df[y10], df[x10], 'o')
    ax[1][0].set_ylabel(y10)
    ax[1][0].set_xlabel(x10)
    ax[1][1].plot(df[y11], df[x11], 'o')
    ax[1][1].set_ylabel(y11)
    ax[1][1].set_xlabel(x11)
    plt.show()

#cite https://matplotlib.org/users/tight_layout_guide.html
#cite: https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html

#call


plt.plot(use_df['Hardship_Index'], use_df['Ave_Annual_Perc_Green'], 'o')


#oh! almost all very near 0 and then big values at all levels
plt.plot(use_df['Hardship_Index'], np.log(use_df['Ave_Annual_Perc_Green']), 'o')
#interesting, do I maybe need to log my greenspace variable so that can see variation within 
# the many low greenspce areas?


#I wonder if there's a relxn between suicide and greenspace, probably not but...
plt.plot(use_df['Hardship_Index'], use_df['Suicide'], 'o')
plt.show()
#hmm, neg rlxn betw hardship and suicide
#lets see if significant:

model = smf.ols('Suicide ~ Hardship_Index + Ave_Annual_Perc_Green' , data=use_df)
result = model.fit()
result.summary()

plt.plot(use_df['Ave_Annual_Perc_Green'], use_df['Suicide'], 'o')
plt.show()

#try a bar
#re-shape df index as perc green
temp_df = some_green_df.set_index('Ave_Annual_Perc_Green')
temp_df.head()

temp_df['Suicide'].plot(kind='bar', subplots=True, figsize=(10,10))
plt.show()
#ok, works, but not informative (would need to put a lot of work in)
#maybe better spent making scatters pretty
def plot_model_line(use_df):
    model = sm.OLS(use_df['Suicide'], sm.add_constant(use_df['Ave_Annual_Perc_Green']))
    p = model.fit().params
    x = np.arange(0.1, 14)
    # scatter-plot data
    ax = use_df.plot(x='Ave_Annual_Perc_Green', y='Suicide', kind='scatter')
    # plot regression line on the same axes, set x-axis limits
    ax.plot(x, p['const'] + p['Ave_Annual_Perc_Green'] * x, 'k-')
    ax.set_ylabel('Average Anual Deaths by Suicide')
    ax.set_xlabel('Avearge Anuall Percent of Area that is Green')
    plt.show()
    #cite: https://stackoverflow.com/questions/42261976/how-to-plot-statsmodels-linear-regression-ols-cleanly

#call
plot_model_line(some_green_df)

def plot_model():
    pass

import seaborn as sns
import matplotlib.ticker as ticker

sns.regplot(x='Ave_Annual_Perc_Green', y='Suicide', data=use_df)
fig, ax = sns.regplot(x='Ave_Annual_Perc_Green', y='Coronary_Heart_Disease', data=use_df)
ax.set_ylabel('Average Anual Deaths by Stroke')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
plt.show()

sns.regplot(x='Ave_Annual_Perc_Green', y='Suicide', data=some_green_df)
#cite for plot regn
#https://stackoverflow.com/questions/42261976/how-to-plot-statsmodels-linear-regression-ols-cleanly


#cite for seaborn 
#https://www.datacamp.com/community/tutorials/seaborn-python-tutorial#show
fig, ax = plt.subplots()
sns.regplot(use_df['Ave_Annual_Perc_Green'], use_df['Coronary_Heart_Disease'],'o')
ax.set_xscale('symlog')
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.set_ylabel('Average Anual Deaths by Heart Disease')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
plt.show()

fig, ax = plt.subplots()
sns.regplot(use_df['Ave_Annual_Perc_Green'], use_df['Suicide'],'o')
ax.set_xscale('symlog')
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.set_ylabel('Average Anual Deaths by Suicide')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
plt.show()


#cutt
def log_plot():
    pass 


#best yet, other attempts below
#cite https://stackoverflow.com/questions/16904755/logscale-plots-with-zero-values-in-matplotlib
#citation https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html
#https://matplotlib.org/3.1.0/gallery/ticks_and_spines/tick-locators.html
#https://matplotlib.org/3.1.0/gallery/ticks_and_spines/tick-formatters.html


#winner
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(some_green_df['Ave_Annual_Perc_Green'], some_green_df['Coronary_Heart_Disease'], 'o')
ax.set_xscale('symlog')
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.set_ylabel('Average Anual Deaths by Heart Disease')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
ax.spines['right'].set_visible(False) #remove spines
ax.spines['top'].set_visible(False)
plt.show()

#does not work:
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(some_green_df['Ave_Annual_Perc_Green'], some_green_df['Coronary_Heart_Disease'], a_list, 'o')
ax.set_xscale('symlog')
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.set_ylabel('Average Anual Deaths by Heart Disease')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
ax.spines['right'].set_visible(False) #remove spines
ax.spines['top'].set_visible(False)
plt.show()

#simpler, does not work:
some_green_df.plot(kind = 'scatter', x = 'Ave_Annual_Perc_Green', y = 'Coronary_Heart_Disease', s = 'Per_Capita_Income')

#but this does: 
#semylog and size for income
ax = some_green_df.plot(kind = 'scatter', x = 'Ave_Annual_Perc_Green', y = 'Coronary_Heart_Disease', s = a_list, figsize=(8,4))
ax.set_xscale('symlog')
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.set_ylabel('Average Anual Deaths by Heart Disease')
ax.set_xlabel('Avearge Anuall Percent of Area that is Green (Log-scale)')
ax.spines['right'].set_visible(False) #remove spines
ax.spines['top'].set_visible(False)
plt.show()

#want to resize the points for income
def this_plot(df, y):
    x = 'Ave_Annual_Perc_Green'
    z = df['Community_Area_Name']
    a_list = df['Per_Capita_Income']/1000
    ax = df.plot(kind='scatter', x=x, y= y , s = a_list)
    #cite https://github.com/pandas-dev/pandas/issues/16827
    ax.set_ylabel('Average Anual Deaths by ' + y)
    ax.set_xlabel('Percent of Area that is Green')
    #cite https://matplotlib.org/examples/pylab_examples/log_demo.html (not longre useing)
    #lable point with Community_Area_Name, only if enough green (for viewability)
    for i, txt in enumerate(z): 
        if df[x][i] >= 2:
            ax.annotate(txt, (df[x][i], df[y][i]), horizontalalignment='center', verticalalignment='bottom')
        else:
            pass
    #cite: https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
    #https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/annotation_demo.html
    #remove spines
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    # Make a legend for per-capita income
    for a_list in [10, 20, 30]:
        plt.scatter([], [], c='k', alpha=0.3, s=a_list,
                    label=str(a_list) + 'k')
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Per-Capita Income')
    #cite https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html
    plt.show()
    #ok to hard code can pas in the [.,.,.] too

#call
this_plot(some_green_df, 'Suicide')
this_plot(use_df, 'Suicide')



#prof likes this plot, says maybe remove the spines, says adding the text is hard (so yah)
#says good amount of info on it


#When you restrict sample to low greenspace, get slight upward curve 

low_green = use_df['Ave_Annual_Perc_Green'] <=2
low_green_df = use_df[low_green]

some_green_df['Ave_Annual_Perc_Green']
some_green_df.reset_index(inplace = True)
x = some_green_df['Ave_Annual_Perc_Green']
y = some_green_df['Suicide']
z2 = some_green_df['Community_Area_Name']
a_list2 = some_green_df['Per_Capita_Income']/1000

#in the fn
ax = some_green_df.plot(kind='scatter', x= 'Ave_Annual_Perc_Green', y= 'Suicide', s = a_list2, label = 'Income')
#cite https://github.com/pandas-dev/pandas/issues/16827
#ax.plot(use_df['Ave_Annual_Perc_Green'], use_df['Suicide'], 'o' )
#ax.semilogx(x, y,'o')
ax.legend()
ax.set_ylabel('Average Anual Suicide Rate')
ax.set_xlabel('Percent of Area that is Green')
#cite https://matplotlib.org/examples/pylab_examples/log_demo.html (not longre useing)
#lable point with Community_Area_Name, only if enough green (for viewability)
for i, txt in enumerate(z2): 
    print(i,txt)
    if x[i] >= 2:
        ax.annotate(txt, (x[i], y[i]), horizontalalignment='center', verticalalignment='bottom')
    else:
        pass
ax.spines['right'].set_visible(False) #remove spines
ax.spines['top'].set_visible(False)
#cite: https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
#https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/annotation_demo.html
plt.show()


def output():
    #save final dfs
    #save plots
    pass
    



def main():
    #run the functions
    #50 percentile have 0 green. 75 percentile is <1
    print('Summary statistics for full sample')
    print(summary_stats(use_df))
    print( )
    print('Summary statistics for constricted sample (omit zero percent green)')
    print(summary_stats(some_green_df))
    print( )

def temporary():   
    ols(use_df)
    ols(some_green_df)
    #call
    covariate_check_list = [('Hardship_Index', 'Ave_Annual_Perc_Green'), ('Perc_Households_Below_Poverty', 'Hardship_Index')]
    for i, j in covariate_check_list:
        covt_check(use_df, i, j)   
    covariate_check_list = [('Hardship_Index', 'Ave_Annual_Perc_Green'), ('Perc_Households_Below_Poverty', 'Hardship_Index')]
    for i, j in covariate_check_list:
        covt_check(some_green_df, i, j)  
    covt_plots(use_df,'Hardship_Index', 'Ave_Annual_Perc_Green','Hardship_Index', 'All_Causes',\
                     'Hardship_Index','Count_Of_Health_Crs','Count_Of_Health_Crs','All_Causes')
    covt_plots(some_green_df,'Hardship_Index', 'Ave_Annual_Perc_Green', 'All_Causes','Hardship_Index',\
                     'Hardship_Index','Count_Of_Health_Crs','All_Causes','Count_Of_Health_Crs')







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

#technically interesting but prob not gonna use
#cite for the belwo: https://www.statsmodels.org/dev/examples/notebooks/generated/regression_plots.html 
#shows the influence of ea point on the sloap fo the line, cool but not gonna include
model = smf.ols('Suicide ~ Hardship_Index + Ave_Annual_Perc_Green' , data=use_df)
result = model.fit()
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(result, ax=ax, criterion="cooks")
plt.show()

fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.plot_partregress('Suicide ', 'Hardship_Index', ['Hardship_Index',  'Ave_Annual_Perc_Green'], data=use_df, ax=ax)
plt.show()

fix, ax = plt.subplots(figsize=(12,14))
fig = sm.graphics.plot_partregress('Suicide ', 'Hardship_Index', ['Ave_Annual_Perc_Green'], data=use_df, ax=ax)
plt.show()

fix, ax = plt.subplots(figsize=(12,14))
fig = sm.graphics.plot_partregress('Suicide', 'Ave_Annual_Perc_Green', ['Hardship_Index'], data=use_df, ax=ax)
plt.show()

#end above section 

#this is very funny! And I have no idea what it is doing
plt.plot(use_df.Suicide, result.fittedvalues, 'r')
#cite https://stackoverflow.com/questions/48682407/r-abline-equivalent-in-python

sns.regplot(low_green_df['Ave_Annual_Perc_Green'], low_green_df['Coronary_Heart_Disease'],'o')
sns.regplot(low_green_df['Ave_Annual_Perc_Green'], low_green_df['Suicide'],'o')
sns.regplot(use_df['Ave_Annual_Perc_Green'], use_df['Suicide'],'o')