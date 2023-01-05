#################################################
#################### IMPORTS ####################
#################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt





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
    
    print("VISUALIZATIONS")
    print("* tnt() - Prints a list of Tips and Tricks (tnt) code that is useful for visualization.")
    print("* plot_distribution(df): Plots the Histogram and Swarmplot overlaid by Boxplot of the numeric columns in the df argument.")



########################################################
#################### VISUALIZATIONS ####################
########################################################

def tnt():
    """
    DESCRIPTION:
    Prints a list of Tips and Tricks (tnt) code that is useful for visualization.
    ___________________________________
    REQUIRED IMPORTS:
    NONE
    ___________________________________
    ARGUMENTS:
    NONE
    """    
    print("USEFUL VISUALIZATION WEBSITES")
    print("* https://chart.guide/ - Helps you chose and design charts and graphs based on type of data.")
    print('* https://i.redd.it/dtnetzv9slu71.png - A collection of all of seaborn’s color palettes.')
    print()
    print("USEFUL VISUALIZATION CODE")
    print("* sns.pairplot(DF, hue='TARGET_COLUMN', corner=True)")

def plot_distribution(df):
    """
    DESCRIPTION:
    Plots the Histogram and Swarmplot overlaid by Boxplot of the numeric columns in the df argument.
    ___________________________________
    REQUIRED IMPORTS:
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    ___________________________________
    ARGUMENTS:
    df = DataFrame
    """        

    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col].hist()
            plt.title(col)
            plt.show()
            ax = sns.boxplot(x=col, data=df, color='#99c2a2')
            ax = sns.swarmplot(x=col, data=df, color='#7d0013', alpha=0.5)
            plt.show()
            print('___________________________________________________________')



############################################################################
#################### WORKING NOTES FOR FUTURE FUNCTIONS ####################
############################################################################

"""
# BUILD A FUNCTION THAT DOES THIS FOR ALL "FLOAT" COLUMNS

float_cols = train_iris.select_dtypes(include='float').columns

# Plot numeric columns
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


"""