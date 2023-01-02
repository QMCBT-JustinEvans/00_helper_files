#################################################
#################### IMPORTS ####################
#################################################

import pandas as pd
import numpy as np





###########################################################
#################### TABLE OF CONTENTS ####################
###########################################################

def TOC():
    """
    DESCRIPTION:
    Prints a Table of Contents for quick reference of what functions are available for use.
    ___________________________________
    REQUIRED IMPORTS:
    NONE
    ___________________________________
    ARGUMENTS:
    NONE
    """    
    
    print("CUSTOM EXPLORATION FUNCTIONS")
    print("* nunique_column_all(df): PRINT NUNIQUE OF ALL COLUMNS")
    print("* nunique_column_objects(df): PRINT NUNIQUE OF COLUMNS THAT ARE OBJECTS")
    print("* nunique_column_qty(df): PRINT NUNIQUE OF COLUMNS THAT ARE *NOT* OBJECTS")
    print("* numeric_range(df): COMPUTE RANGE FOR ALL NUMERIC VARIABLES")

    
    
#####################################################
#################### EXPLORATION ####################
#####################################################
    
def tnt():
    """
    DESCRIPTION:
    Prints a list of Tips and Tricks (tnt) code that is useful for exploration.
    ___________________________________
    REQUIRED IMPORTS:
    NONE
    ___________________________________
    ARGUMENTS:
    NONE
    """    
    
    print("USEFUL EXPLORATORY CODE")
    print("* df.head() - Display first five rows")
    print("* df.shape - display the row, column count of the df")
    print("* df.shape[0] - read row count")
    print("* df.describe().T - display column stats of DataFrame")
    print("* df.columns.to_list() - Display list of column names")
    print("* df.COLUMNNAME.value_counts(dropna=False) - Displays a count of each value to include NaNs")
    print("* df.dtypes - Displays the d-type of each column")
    print("* df.select_dtypes(include='object').columns - Displays list of columns with d-type object")
    print("* df.select_dtypes(include='float').columns - Displays list of columns with d-type float")
    print("* pd.crosstab(df.COLUMN_1, df.COLUMN_2)")

def nunique_column_all(df):
    """
    DESCRIPTION:
    Displays column values and the count of their occurance.
    ___________________________________
    REQUIRED IMPORTS:
    NONE
    ___________________________________
    ARGUMENTS:
    df = DataFrame
    """   
    
    # loop value_counts() for each column in the df
    for col in df.columns:
        print(df[col].value_counts())
        print()

def nunique_column_objects(df): 
    """
    DESCRIPTION:
    Displays count of unique values for each d-type object column in the DataFrame argument.
    ___________________________________
    REQUIRED IMPORTS:
    NONE
    ___________________________________
    ARGUMENTS:
    df = DataFrame
    """   
    
    # PRINT NUNIQUE OF COLUMNS THAT ARE OBJECTS    
    for col in df.columns:
        if df[col].dtypes == 'object':
            print(f'{col} has {df[col].nunique()} unique values.')

def nunique_column_qty(df): 
    """
    DESCRIPTION:
    Displays count of unique values for each non-object d-type column in the DataFrame argument.
    ___________________________________
    REQUIRED IMPORTS:
    NONE
    ___________________________________
    ARGUMENTS:
    df = DataFrame
    """   
    
    # PRINT NUNIQUE OF COLUMNS THAT ARE *NOT* OBJECTS
    for col in df.columns:
        if df[col].dtypes != 'object':
            print(f'{col} has {df[col].nunique()} unique values.')

def numeric_range(df):
    """
    DESCRIPTION:
    Displays the numeric range of all columns that are int or float d-type within the DataFrame argument.
    ___________________________________
    REQUIRED IMPORTS:
    NONE
    ___________________________________
    ARGUMENTS:
    df = DataFrame
    """   

    # COMPUTE RANGE FOR ALL NUMERIC VARIABLES
    numeric_list = df.select_dtypes(include = 'float').columns.tolist()
    numeric_range = df[numeric_list].describe().T
    numeric_range['range'] = numeric_range['max'] - numeric_range['min']
    return numeric_range



############################################################################
#################### WORKING NOTES FOR FUTURE FUNCTIONS ####################
############################################################################

"""
# BUILD A FUNCTION THAT DOES THIS FOR ALL "FLOAT" COLUMNS

float_cols = train_iris.select_dtypes(include='float').columns

Plot numeric columns
plot_float_cols = float_cols 
for col in plot_float_cols:
    plt.hist(train_iris[col])
    plt.title(col)
    plt.show()
    plt.boxplot(train_iris[col])
    plt.title(col)
    plt.show()


# BUILD A FUNCTION THAT DOES THIS FOR ALL "OBJECT" COLUMNS

train.species.value_counts()
plt.hist(train_iris.species_name)


# BUILD A FUNCTION THAT DOES THIS

test_var = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for var in test_var:
    t_stat, p_val = t_stat, p_val = stats.mannwhitneyu(virginica[var], versicolor[var], alternative="two-sided")
    print(f'Comparing {var} between Virginica and Versicolor')
    print(t_stat, p_val)
    print('')
    print('---------------------------------------------------------------------')
    print('')


sns.pairplot(DF, hue='TARGET_COLUMN', corner=True)
plt.show()

"""
