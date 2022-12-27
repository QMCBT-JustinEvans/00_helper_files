#################################################
#################### IMPORTS ####################
#################################################

# ---------------- #
# Common Libraries #
# ---------------- #

import os
import requests
import numpy as np
import pandas as pd

# Working with Dates & Times
from sklearn.model_selection import TimeSeriesSplit
from datetime import timedelta, datetime

import statsmodels.api as sm

# to evaluate performance using rmse
from sklearn.metrics import mean_squared_error
from math import sqrt 

# for tsa 
import statsmodels.api as sm

# holt's linear trend model. 
from statsmodels.tsa.api import Holt




# Plots, Graphs, & Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.dates import DateFormatter

# plotting defaults
plt.rc('figure', figsize=(13, 7))
plt.style.use('seaborn-whitegrid')
plt.rc('font', size=16)

# --------- #
# Data Sets #
# --------- #

from pydataset import data
# Call from the vega_datasets library like so:
    ## df = data.

from vega_datasets import data
# Call from the vega_datasets library like so:
    ## df = data.sf_temps()


# -------------- #
# Action Imports #
# -------------- #

# Warnings 
import warnings 
warnings.filterwarnings("ignore")

# ------------ #
# JUPYTER ONLY #
# ------------ #
"""
# Disable autosave
%autosave 0

# Increases Display Resolution for Graphs 
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'

# Left Align Tables in Jupyter Notebook
from IPython.core.display import HTML
table_css = 'table {align:left;display:block}'
HTML('{}'.format(table_css))

"""

# ------------- #
# Local Imports #
# ------------- #

# env containing sensitive access credentials
from env import user, password, host

# importing sys
import sys
 
# adding 00_helper_files to the system path
sys.path.insert(0, '/Users/qmcbt/codeup-data-science/00_helper_files')

# Import Helper Files
import QMCBT_00_quicktips as qt
import QMCBT_01_acquire as acq
import QMCBT_02_prepare as prep
import QMCBT_03_explore as exp
import QMCBT_04_visualize as viz
import QMCBT_05_model as mod
import QMCBT_wrangle as w


######################################################
#################### ACQUIRE DATA ####################
######################################################

def get_db_url(database):
    '''
    Description:
    Returns a formatted string using credentials stored in local env.py file that can be passed to a pd.read_sql() function
    
    Required Imports:
    import pandas as pd
    
    Arguments:
    database - 'this is the name of the database that you wish to retrieve'
    '''

    return f'mysql+pymysql://{user}:{password}@{host}/{database}'

# ----------------------- #
# TITANIC DATA (from SQL) #
# ----------------------- #

def new_titanic_df():
    '''
    This function reads the titanic data from the Codeup database into a DataFrame.
    '''
   
    # Create SQL query.
    sql_query = 'SELECT * FROM passengers'
    
    # Read in DataFrame from Codeup database.
    df = pd.read_sql(sql_query, get_db_url('titanic_db'))
    
    return df

def get_titanic_df():
    '''
    This function reads in titanic data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a DataFrame.
    '''
    
    if os.path.isfile('titanic_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('titanic_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_titanic_df()
        
        # Write DataFrame to a csv file.
        df.to_csv('titanic_df.csv')
        
    return df

# -------------------- #
# IRIS DATA (from SQL) #
# -------------------- #

def new_iris_sql_df():
    '''
    This function reads the iris data from the Codeup database into a DataFrame.
    '''
    
    # Create SQL query.
    sql_query = 'SELECT species_id, species_name, sepal_length, sepal_width, petal_length, petal_width FROM measurements JOIN species USING(species_id)'
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('iris_db'))
    
    return df

def get_iris_sql_df():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a DataFrame.
    '''

    if os.path.isfile('iris_sql_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('iris_sql_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_iris_sql_df()
        
        # Cache data
        df.to_csv('iris_sql_df.csv')
        
    return df

# ------------------------ #
# IRIS DATA (from SEABORN) #
# ------------------------ #

def new_iris_sns_df():
    '''
    This function reads the iris data from the seaborn database into a DataFrame.
    '''
    
    # Read in DataFrame from pydata db.
    df = sns.load_dataset('iris')
    
    return df

def get_iris_sns_df():
    '''
    This function reads in iris data from seaborn database, writes data to
    a csv file if a local file does not exist, and returns a DataFrame.
    '''
  
    if os.path.isfile('iris_sns_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('iris_sns_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_iris_sns_df()
        
        # Cache data
        df.to_csv('iris_sns_df.csv')
        
    return df

# ----------------------- #
# TELCO DATA (from SQL) #
# ----------------------- #

def new_telco_churn_df():
    '''
    This function reads the telco_churn (NOT telco_normalized) data from the Codeup database into a DataFrame.
    '''
    # Create SQL query.
    sql_query = 'SELECT * FROM customers LEFT JOIN internet_service_types USING (internet_service_type_id) LEFT JOIN contract_types USING (contract_type_id) LEFT JOIN payment_types USING (payment_type_id);'
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('telco_churn'))
    
    return df

def get_telco_churn_df():
    '''
    This function reads in telco_churn (NOT telco_normalized) data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a DataFrame.
    '''
    if os.path.isfile('telco_churn_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco_churn_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from telco db into a DataFrame
        df = new_telco_churn_df()
        
        # Cache data
        df.to_csv('telco_churn_df.csv')
        
    return df

# --------------------- #
# STORE DATA (from SQL) #
# --------------------- #

def get_store_data():
    '''
    Returns a dataframe of all store data in the tsa_item_demand database and saves a local copy as a csv file.
    '''
    query = '''
    SELECT *
    FROM items
    JOIN sales USING(item_id)
    JOIN stores USING(store_id) 
    '''
    
    df = pd.read_sql(query, get_db_url('tsa_item_demand'))
    
    df.to_csv('tsa_store_data.csv', index=False)
    
    return df

def wrangle_store_data():
    '''
    Checks for a local cache of tsa_store_data.csv and if not present will run the get_store_data() function which acquires data from Codeup's mysql server
    '''
    filename = 'tsa_store_data.csv'
    
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        df = get_store_data()
        
    return df

# ----------------------- #
# OPSD DATA (from GitHub) #
# ----------------------- #

def get_opsd_data():
    """
    Read in OPS data from local cache if it exists, otherwise pull it from github source and cache it.
    """
    
    if os.path.exists('opsd.csv'):
        return pd.read_csv('opsd.csv')
    else:
        df = pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')
        df.to_csv('opsd.csv', index=False)
    return df

# ----------------------- #
# SAAS DATA (from Codeup) #
# ----------------------- #

def get_saas():
    """
    Read in saas data from cache or Codeup server then write to cache
    """
    
    filename = 'saas.csv'

    if os.path.isfile(filename):
        df = pd.read_csv(filename)

    else:
        url = 'https://ds.codeup.com/saas.csv'
        df = pd.read_csv(url)
        df.to_csv('saas.csv', index=False)
        
    return df