###########################################################
#################### TABLE OF CONTENTS ####################
###########################################################

def TOC():
    """
    Prints a Table of Contents for quick reference of what functions are available for use.
    """    
    
    print("IMPORTS")
    print("* imports() - Prints a list of standard import functions that can be quickly copy pasted for use.")
    print()
    print("TIPS & TRICKS")
    print("* explore_tips() - Displays useful code tips for exploration.")
    print()
    print("JUPYTER MARK UP")
    print("* cell_color() - Prints a short list of Jupyter Workbook markup code to change cell colors.")
    print("* display_pic(url, width, height) - Display a picture from a stored location")
    print()
    print("CHEAT SHEETS (cs_)")
    print("* cs_confusion_matrix() - Display Cheat Sheet to help understand and read a confusion matrix.")
    print("* cs_hypothesis() - Display Cheat Sheet to help create and test Hypothesis.")
    print("* cs_train_val_test() - Display Cheat Sheet to show graphic for train/val/test/split and Xy_split with steps to be performed for modeling.")
    print()
    print("")
    

    
#########################################################
#################### Display Imports ####################
#########################################################

def imports():
    """
    Prints a list of standard import functions that can be quickly copy pasted for use.
    """

    print("""
    # ---------------- #
    # Common Libraries #
    # ---------------- #
      
    # Standard Imports
    import os
    import requests
    import numpy as np
    import pandas as pd

    # Working with Dates & Times
    from sklearn.model_selection import TimeSeriesSplit
    from datetime import timedelta, datetime

    # Working with Math & Stats
    import statsmodels.api as sm
    import scipy.stats as stats

    # to evaluate performance using rmse
    from sklearn.metrics import mean_squared_error
    from math import sqrt 

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

    """)

    print("""
    # --------- #
    # Data Sets #
    # --------- #

    from pydataset import data
    # Call from the vega_datasets library like so:
        ## df = data.

    from vega_datasets import data
    # Call from the vega_datasets library like so:
        ## df = data.sf_temps()

    """)

    print("""
    # -------------- #
    # Action Imports #
    # -------------- #

    # Warnings 
    import warnings 
    warnings.filterwarnings("ignore")

    """)

    print("""
    # ------------ #
    # JUPYTER ONLY #
    # ------------ #
    
    # Disable autosave
    %autosave 0

    # Increases Display Resolution for Graphs 
    %matplotlib inline 
    %config InlineBackend.figure_format = 'retina'

    # Left Align Tables in Jupyter Notebook
    from IPython.core.display import HTML
    table_css = 'table {align:left;display:block}'
    HTML('{}'.format(table_css))

    """)

    print("""
    # ------------- #
    # Local Imports #
    # ------------- #

    # env containing sensitive access credentials
    from env import user, password, host

    # importing sys
    import sys

    # adding 00_helper_files to the system path
    sys.path.insert(0, '/Users/qmcbt/codeup-data-science/00_helper_files')

    # Import Helper Modules
    import QMCBT_00_quicktips as qt
    import QMCBT_01_acquire as acq
    import QMCBT_02_prepare as prep
    import QMCBT_03_explore as exp
    import QMCBT_04_visualize as viz
    import QMCBT_05_model as mod
    import QMCBT_wrangle as w

    """)

    
    
#######################################################
#################### TIPS & TRICKS ####################
#######################################################

def explore_tips():
    """
    Displays useful code tips for exploration.
    """    
    print("""
    
    df.describe(include='all') - displays all column stats to include object type
    
    
    """)

########################################################
#################### JUPYTER MARKUP ####################
########################################################

# ----------- #
# Color Cells #
# ----------- #

def cell_color():
    """
    Prints a short list of Jupyter Workbook markup code to change cell colors.
    """
    print("""
    
    <div class="alert alert-info"> </div> - Change Cell color to BLUE
    <div class="alert alert-success"> </div> - Change Cell color to GREEN
    <div class="alert alert-warning"> </div> - Change Cell color to YELLOW
    <div class="alert alert-danger"> </div> - Change Cell color to RED
    ! (followed by terminal command) - Run terminal code in current directory
    %who - displays all assigned variables
   
    """)

# --------------- #
# Display Picture #
# --------------- #

# import image module
from IPython.display import Image

def display_pic(url, width=920, height=474):
    """
    Display a picture from a stored location
    
    Required Imports:
    from IPython.display import Image    
    
    Arguments:
       url = The location link of the file either online or local
     width = The width in pixels to display the picture
    height = The height in pixels to display the picture
    """
        
    # get the image
    return Image(url=url, width=width, height=height)
    
    
######################################################
#################### CHEAT SHEETS ####################
######################################################

# ---------------------------- #
# Confusion Matrix Cheat Sheet #
# ---------------------------- #

def cs_confusion_matrix():
    """
    Displays a graphic with definitions to help with understanding and reading the confusion matrix.
    """
    
    print("""
    
    POSITIVE (+) = insert Positive statement here  
    NEGATIVE (-) = insert Negative statement here    
     
    RECALL    
    TP / (TP + FN)    
    Use for less Type II errors when FN is worst outcome    
    Maximize for RECALL if Cost of FN > Cost of FP    
     
    PRECISION    
    TP / (TP + FP)    
    Use for less Type I errors when FP is worst outcome    
    Maximize for PRECISION if Cost of FP > Cost of FN    
     
    ACCURACY    
    (TP + TN)/(FP+FN+TP+TN)    
    prediction TRUE / total    
    Maximize for ACCURACY if neither RECALL or PRECISION outweigh eachother  
    
    Classification Confusion Matrix (actual_col, prediction_row)(Positive_first, Negative_second)  
                          +------------------------------------------+  
                          | actual Positive (+) | actual Negative(-) |  
    +---------------------+---------------------+--------------------+  
    |  pred Positive (+)  |     TP              |     FP (Type I)    |  
    +---------------------+---------------------+--------------------+  
    |  pred Negative (-)  |     FN (Type II)    |     TN             |  
    +---------------------+---------------------+--------------------+  
    
    sklearn Confusion Matrix (prediction_col, actual_row)(Negative_first, Positive_second)  
                          +--------------------------------------+  
                          | pred Negative(-) | pred Positive (+) |  
    +---------------------+------------------+-------------------+  
    | actual Negative (-) |        TN        |    FP (Type I)    |  
    +---------------------+------------------+-------------------+  
    | actual Positive (+) |   FN (Type II)   |         TP        |    
    +---------------------+------------------+-------------------+  
     
    FP: We predicted it was a POSITIVE when it was actually a NEGATIVE    
       FP = We FALSELY predicted it was POSITIVE    
    False = Our prediction was False, it was actually the opposite of our prediction    
    Oops... TYPE I error!    
     
    FN: We predicted it was a NEGATIVE when it was actually a POSITIVE    
       FN = We FALSELY predicted it was NEGATIVE    
    False = Our prediction was False, it was actually the opposite of our prediction    
    Oops... TYPE II error!    
     
    TP: We predicted it was a POSITIVE and it was actually a POSITIVE    
      TP = We TRUELY predicted it was POSITIVE    
    True = Our prediction was True, it was actually the same as our prediction    
    
    TN: We predicted it was a NEGATIVE and it was actually a NEGATIVE    
      TN = We TRUELY predicted it was NEGATIVE    
    True = Our prediction was True, it was actually the same as our prediction   

    """)

# ---------------------- #
# Hypothesis Cheat Sheet #
# ---------------------- #

def cs_hypothesis():
    """
    Displays Cheat Sheet to help create and test Hypothesis.
    """
        
    print("""
    
    **Set Hypothesis**  

    * One Tail (```<= | >```) or Two Tails (```== | !=```)?
        * two_tail (feature_1, feature_2)  


    * One Sample or Two Samples?    
        * two_sample (feature_1, feature_2)  


    * Continuous or Discreat?  
        * Discreat (feature_1) vs Discreat (feature_2) = **$Chi^2$**
            * T-Test = ```Discreat``` vs ```Continuous```  
            * Pearson’s = ```Continuous``` vs ```Continuous``` (linear)  
            * $Chi^2$ = ```Discreat``` vs ```Discreat```  


    * $𝐻_0$: The opposite of what I am trying to prove  
        * $H_{0}$: feature_1 **is NOT** ```dependent``` on feature_2  
        * ```feature_1``` != ```feature_2```  


    * $𝐻_𝑎$: What am I trying to prove  
        * $H_{a}$: feature_1 **is** ```dependent``` on feature_2  
        * ```feature_1``` == ```feature_2```
        
    """)

# --------------------------------------- #    
# Train, Validate, Test SPLIT Cheat Sheet #
# --------------------------------------- #

def cs_train_val_test():
    """
    Display Cheat Sheet that shows graphic for train/test/split and Xy_split with steps to be performed for modeling.
    """

    print("""
    _______________________________________________________________  
    |                              DF                             |  
    |-------------------+-------------------+---------------------|  
    |       Train       |       Validate    |          Test       |  
    +-------------------+-------------------+-----------+---------+  
    | x_train | y_train |   x_val  |  y_val |   x_test  |  y_test |  
    +-------------------------------------------------------------+  
     
    * 1. tree_1 = DecisionTreeClassifier(max_depth = 5)  
    * 2. tree_1.fit(x_train, y_train)  
    * 3. predictions = tree_1.predict(x_train)  
    * 4. pd.crosstab(y_train, predictions)  
    * 5. val_predictions = tree_1.predict(x_val)  
    * 6. pd.crosstab(y_val, val_predictions)  

    """)