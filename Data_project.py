
########
#setup#
########

#import packages

import pandas as pd
import pandas_datareader.data as web
from pandas_datareader import wb
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf 

#set path
b_path = '~Sarah/Desktop/Programming/Final_Project' 
#b_path = '~Sarah/Documents/GitHub/spring-2019-final-project-ippp19_sarah'
os.chdir(os.path.expanduser(b_path)) #set working directory 


#list of url, file name tuples to be used when downloading data
URLS = [('https://data.cityofchicago.org/api/views/kn9c-c2s2/rows.csv?accessType=DOWNLOAD', 'Chicago_SES.csv'), 
        ('https://citytech-health-atlas-data-prod.s3.amazonaws.com/uploads/uploader/path/721/Green_Index__Land_Cover___Ave_Annual__v2.xlsx', 'Chicago_Green.xls'),
        ('https://data.cityofchicago.org/api/views/j6cj-r444/rows.csv?accessType=DOWNLOAD', 'Chicago_Death.csv'),
        ('https://data.cityofchicago.org/api/views/cjg8-dbka/rows.csv?accessType=DOWNLOAD', 'Chicago_health_cr.csv')]

###########
#functions#
###########

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

#read in data
def read_data(path, filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(path, filename))
    elif filename.endswith('.xls'):
        df = pd.read_excel(os.path.join(path, filename))
    else:
        return 'unexpected file type in read_data'
    return df


#re-name community Area: for some reasion after the pivot, can't use Community Area to merge on
#but Community Area Number works 
#re-shape df so that numbers reflect average annual deaths

def parse_death(death_df):
    death_df.rename(columns = {'Community Area': 'Community Area Number'}, inplace=True)
    avg_an_death = death_df.pivot(index = 'Community Area Number', columns='Cause of Death', 
                                  values='Average Annual Deaths 2006 - 2010')
    avg_an_death.drop(0, axis = 0, inplace = True) #drop the Chicago Total
    return avg_an_death

#get a count of healthcare centers per community area:
def parse_healthcr(healthcr_df):
    healthcr_df['count_of_health_crs'] = 1 
    count_of_crs = healthcr_df.groupby('Community Areas').sum().reset_index()
    return count_of_crs

#Merge datasets:
def merge_dfs(SES_df,green_df,avg_an_death,count_of_crs):     
    SES_green = SES_df.merge(green_df, left_on='Community Area Number', right_on = 'Geo_ID', how = 'inner')

    SES_green_death = SES_green.merge(avg_an_death, on='Community Area Number', how = 'inner')

    SES_green_death_healthcr = SES_green_death.merge(count_of_crs, left_on='Community Area Number', 
                                                     right_on='Community Areas', how = 'outer')
    #fill in Nan with 0 (bc if not in the previous database then doesn't have a health center)
    SES_green_death_healthcr['count_of_health_crs'].fillna(value = 0, inplace=True)
    
    #drop colums with Nan (all cols dropped for this df are completely empty)
    SES_green_death_healthcr.dropna(axis=1,inplace=True)

    #drop columns that do not provide useful information/may not apply to all entries in the row after 
    # the merge (e.g. SubCategory or Map_Key from green_df), or duplicates eg Geo_ID
    SES_green_death_healthcr.drop(columns = ['Geo_Group', 'Geo_ID', 'Category', 'SubCategory',
                                            'Geography', 'Map_Key'], inplace = True)
    return SES_green_death_healthcr 

#cite: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html

#reduce number of columns for convinience, drop ones outside of investigation
def drop_col(full_df):
    use_df = full_df.drop(columns = 
                        ['PERCENT OF HOUSING CROWDED',  'PERCENT AGED 16+ UNEMPLOYED',
                        'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA', 'PERCENT AGED UNDER 18 OR OVER 64', 
                        'Year','Injury, unintentional', 'All causes in females', 'All causes in males', 
                        'Alzheimers disease', 'Assault (homicide)', 'Breast cancer in females', 
                        'Cancer (all sites)', 'Colorectal cancer', 'Prostate cancer in males',
                        'Firearm-related', 'Kidney disease (nephritis, nephrotic syndrome and nephrosis)', 
                        'Liver disease and cirrhosis', 'Lung cancer', 'Indicator'])
    return use_df

#rename columns for use in ols
def re_name(df):
    df.rename(columns = {"Ave_Annual_Number": "Ave_Annual_Perc_Green"}, inplace = True)
    df.columns = [c.split(sep = ' (')[0] for c in df.columns]
    df.columns = [c.rstrip() for c in df.columns]
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df.columns = [c.replace("-", "_") for c in df.columns]
    df.columns = [c.title() for c in df.columns]
    return df

#cite https://stackoverflow.com/questions/8347048/how-to-convert-string-to-title-case-in-python
#cite https://stackoverflow.com/questions/39741429/pandas-replace-a-character-in-all-column-names


#Summary statistics
def summary_stats(df):
    summary = df.describe()
    summary.drop(columns = ['Community_Area_Number'], inplace = True)
    summary = summary.transpose()
    summary = summary.round(2)
    return summary

#cite https://stackoverflow.com/questions/33889310/r-summary-equivalent-in-numpy 


#Limitations in variation in greenspace (over half of the dataset is at 0 avg ann perc green!)
#look at a restricted sample, exclude 0 values.
def restricted_df(use_df):
    some_green = use_df['Ave_Annual_Perc_Green'] != 0
    some_green_df = use_df[some_green]
    some_green_df.reset_index(drop = True, inplace = True) #reset the index, needed for plotting later
    return some_green_df

#cite https://chrisalbon.com/python/data_wrangling/pandas_selecting_rows_on_conditions/  
#cite https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html

#zero_green = use_df['Ave_Annual_Perc_Green'] == 0
#zero_green_df = use_df[zero_green]

#where is the most greenspace? returns characteristics of this community area
def max_green(use_df):
    i = np.argmax(use_df['Ave_Annual_Perc_Green'])
    max_green = use_df.iloc[i]
    return max_green

#cite https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list

#analysis
def ols(use_df, y):
    print('Dependent Variable: ' + y) #for output display
    m = smf.ols(y + '~ Ave_Annual_Perc_Green + Hardship_Index + Count_Of_Health_Crs' , data = use_df)
    result = m.fit()
    print(result.summary()) #show results in output
    print( ) #for output display readability
    #prepare results to save to png
    plt.rc('figure',figsize=(9, 5.5))
    plt.text(0.01, 0.05, str(result.summary()), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.title('Death by '+ y +' on Green-Space, Controling for Area SES and Health Centers')

#cite https://stackoverflow.com/questions/46664082/save-statsmodels-results-in-python-as-image-file

#Explore covariate relationships:
def covt_check(df, y_string, x_string):
    print(y_string + ' on ' + x_string)
    test_model = smf.ols(y_string + '~' + x_string , data = df)
    result = test_model.fit()
    print(result.summary())
    print( )

#plot covariate relationships: 
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
    plt.savefig('appendix/Covariate_Relationships')
    plt.close()
    #plt.show()

#cite https://matplotlib.org/users/tight_layout_guide.html
#cite: https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html


#look at line of best fit for greenspace-deaths relationship:
def plot_model_line(use_df, y, ylabel):
    model = sm.OLS(use_df[y], sm.add_constant(use_df['Ave_Annual_Perc_Green']))
    p = model.fit().params
    x = np.arange(0.1, 14) #range of line (on x axis)
    # scatter-plot data
    ax = use_df.plot(x='Ave_Annual_Perc_Green', y=y, kind='scatter')
    # plot regression line on the same axes
    ax.plot(x, p['const'] + p['Ave_Annual_Perc_Green'] * x, 'k-')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Avearge Anuall Percent of Area that is Green')
    plt.savefig('appendix/Best_Fit_for_Green_on_Deaths_by_' +y)
    plt.close()
    #plt.show()

#cite: https://stackoverflow.com/questions/42261976/how-to-plot-statsmodels-linear-regression-ols-cleanly

#plot: x:greenspace, y:avg anual deaths, size:per capita income in $1000
def death_green_SES_plot(df, y, ylabel):
    x = 'Ave_Annual_Perc_Green'
    z = df['Community_Area_Name']
    a_list = df['Per_Capita_Income']/1000
    ax = df.plot(kind='scatter', x=x, y= y , s = a_list, figsize= (10,8))
    #cite https://github.com/pandas-dev/pandas/issues/16827
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Percent of Area that is Green')
    #lable point with Community_Area_Name, only if enough green or max death (for viewability) 
    temp_list = []
    for i, txt in enumerate(z): 
        if df[x][i] >= 2:
            ax.annotate(txt, (df[x][i], df[y][i]), horizontalalignment='center', verticalalignment='bottom')
        elif df[y][i] == df[y].max():
            temp_list.append(i)
            temp_list.append(txt)
            if len(temp_list) >=4:
                ax.annotate(temp_list[1], (df[x][temp_list[0]], df[y][temp_list[0]]), 
                            horizontalalignment='left', verticalalignment='bottom')
                ax.annotate(temp_list[3], (df[x][temp_list[2]], df[y][temp_list[2]]), 
                            horizontalalignment='left', verticalalignment='top')
            else:
                ax.annotate(temp_list[1], (df[x][temp_list[0]], df[y][temp_list[0]]), 
                            horizontalalignment='left', verticalalignment='bottom')
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
    #plt.savefig('SES_Green_and_Deaths_by_' +y)
    #plt.close()
    #plt.show()


##########
#fn calls#
##########

#call the download funtion
for url, filename in URLS:
    download_data(url, filename)

def main():
    #call the read function
    df_contents = []
    for url, filename in URLS:
        df = read_data(b_path, filename)
        if filename == 'Chicago_Death.csv':
            df_contents.append(parse_death(df))
        elif filename == 'Chicago_health_cr.csv':
            df_contents.append(parse_healthcr(df))
        else:
            df_contents.append(df)  
    #call the merge function
    full_df= merge_dfs(df_contents[0], df_contents[1], df_contents[2], df_contents[3])
    #call the drop_col function -> generate primary df
    use_df = drop_col(full_df)
    use_df = re_name(use_df)
    #generate restirctd sample df    
    some_green_df = restricted_df(use_df)
    #show summary statistics
    #print('Summary statistics for the full sample')
    print(summary_stats(use_df))
    #print( )
    print('Summary statistics for constricted sample (omit zero percent green)')
    print(summary_stats(some_green_df))
    print( )  
    print('The area with the highest percent green:')
    print(max_green(use_df))
    print( )
    #set list of tuples to plot and regress
    to_plot = [('Suicide','Average Anual Deaths by Suicide'), 
                ('Diabetes_Related','Average Anual Diabetes Related Deaths'),
                ('Coronary_Heart_Disease','Average Anual Deaths from Coronary Heart Disease'),
                ('Stroke','Average Anual Deaths by Stroke')]
    #plot restricted sample
    """
    for col, ylab in to_plot:
        death_green_SES_plot(some_green_df, col, ylab)
        plt.title('There is a Negative Relationship between Greenspace and Mortality Rate')
        plt.suptitle('Plot of Community Areas with Non-Zero Greenspace')
        plt.savefig('SES_Green_and_Deaths_by_' +col)
        plt.close()
        """
    #plot full sample
    #for col, ylab in to_plot:
    #   death_green_SES_plot(use_df, col, ylab)
     #   plt.savefig('appendix/SES_Green_and_Deaths_by_' +col)
      #  plt.close()
    #ols on restricted sample
    '''
    for y, ylab in to_plot:
        ols(some_green_df, y)
        plt.savefig(y + '_reg_output.png')
        plt.close()
        '''
    #ols on full sample
    #for y, ylab in to_plot:
    #    ols(use_df, y)
    #    plt.savefig('appendix/' + y + '_reg_output.png')
    #    plt.close()
    #save dfs
    some_green_df.to_csv('non_zero_green_Chicago_Deaths_Green_and_SES.csv')  
    #use_df.to_csv('appendix/full_sample_Chicago_Deaths_Green_and_SES.csv')
    #cite https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html 
    
    #appendix materials:
    print(some_green_df.columns)
    print( )
    print('Percent of area that is green and Area SES are not statistically significanlty associated')
    for col, ylab in to_plot:
        plot_model_line(some_green_df, col, ylab)
    covariate_check_list = [('Ave_Annual_Perc_Green', 'Hardship_Index'), 
                            ('Ave_Annual_Perc_Green', 'Per_Capita_Income'),
                            ('Ave_Annual_Perc_Green', 'Percent_Households_Below_Poverty')]
    for i, j in covariate_check_list:
        covt_check(use_df, i, j)   
    #for i, j in covariate_check_list:
    #    covt_check(some_green_df, i, j) 
    #I shoudl re-code so that I can use the same list for covt check and plot! some kind of fig add
    # bonus, then I could put the line of best fit plots into one fig! 
    #covt_plots(use_df,'Hardship_Index', 'Ave_Annual_Perc_Green','Hardship_Index', 'All_Causes',\
    #                 'Hardship_Index','Count_Of_Health_Crs','Count_Of_Health_Crs','All_Causes')
    #covt_plots(some_green_df,'Hardship_Index', 'Ave_Annual_Perc_Green', 'All_Causes','Hardship_Index',\
    #                 'Hardship_Index','Count_Of_Health_Crs','All_Causes','Count_Of_Health_Crs')
    print( )
    print('Dataframe, plots and regression results have been saved to' + b_path)



