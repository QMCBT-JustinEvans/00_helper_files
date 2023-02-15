#################################################
#################### IMPORTS ####################
#################################################

import pandas as pd
import numpy as np

# import splitting functions
from sklearn.model_selection import train_test_split



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
    print("CLEAN DATA")
    print("prep_iris_df(iris_df)")
    print("prep_titanic_df(titanic_df)")
    print("prep_telco_churn_df(telco_churn_df)")
    print("prep_store_data(df, datetime_column)")
    print("* dmy_conversion(df, datetime_column)")
    print("* set_index(df, datetime_column)")
    print()
    print("SPLIT DATA")
    print("train_val_test_split(df, target)")

    
#################################################
#################### PREPARE ####################
#################################################

# ---------- #
# CLEAN DATA #
# ---------- #

def prep_iris_df(iris_df):
    iris_df = iris_df.drop(columns='species_id')
    iris_df = iris_df.rename(columns={'species_name': 'species'})
    dummy_iris_df = pd.get_dummies(iris_df.species, drop_first=True)
    iris_df = pd.concat([iris_df, dummy_iris_df], axis=1)
    return iris_df

def prep_titanic_df(titanic_df):
    titanic_df = titanic_df.drop(columns=['passenger_id', 'class', 'deck'])
    dummy_df = pd.get_dummies(data=titanic_df['sex'], drop_first=True)
    dummy_df2 = pd.get_dummies(data=titanic_df['embark_town'], drop_first=False)
    titanic_df = pd.concat([titanic_df, dummy_df, dummy_df2], axis=1)
    return titanic_df

def prep_telco_churn_df(telco_churn_df):
    telco_churn_df = telco_churn_df.drop(columns=['payment_type_id', 'contract_type_id', 
                                              'internet_service_type_id', 'customer_id'])
    encoded_df = pd.DataFrame()
    encoded_df['male_encoded'] = telco_churn_df.gender.map({'Male': 1, 'Female': 0})
    encoded_df['partner_encoded'] = telco_churn_df.partner.map({'Yes': 1, 'No': 0})
    encoded_df['dependents_encoded'] = telco_churn_df.dependents.map({'Yes': 1, 'No': 0})
    encoded_df['phone_service_encoded'] = telco_churn_df.phone_service.map({'Yes': 1, 'No': 0})
    encoded_df['multiple_lines_encoded'] = telco_churn_df.phone_service.map({'Yes': 1, 'No': 0})
    encoded_df['online_security_encoded'] = telco_churn_df.phone_service.map({'Yes': 1, 'No': 0})
    encoded_df['online_backup_encoded'] = telco_churn_df.phone_service.map({'Yes': 1, 'No': 0})
    encoded_df['device_protection_encoded'] = telco_churn_df.phone_service.map({'Yes': 1, 'No': 0})
    encoded_df['streaming_tv_encoded'] = telco_churn_df.phone_service.map({'Yes': 1, 'No': 0})
    encoded_df['streaming_movies_encoded'] = telco_churn_df.phone_service.map({'Yes': 1, 'No': 0})
    encoded_df['paperless_billing_encoded'] = telco_churn_df.paperless_billing.map({'Yes': 1, 'No': 0})
    encoded_df['churn_encoded'] = telco_churn_df.churn.map({'Yes': 1, 'No': 0})
    encoded_df['tech_support_encoded'] = telco_churn_df.churn.map({'Yes': 1, 'No': 0})
    encoded_cols = encoded_df.columns

    dummy_df = pd.get_dummies(data=telco_churn_df[['internet_service_type', 
                                                   'contract_type', 
                                                   'payment_type'
                                                  ]], drop_first=False)
    
    telco_churn_df = pd.concat([telco_churn_df, encoded_df, dummy_df], axis=1)
    
    drop_cols = ['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
             'online_security', 'online_backup', 'device_protection', 'streaming_tv', 
             'streaming_movies', 'paperless_billing', 'churn', 'internet_service_type', 
             'contract_type', 'payment_type', 'tech_support']
    
    telco_churn_df = telco_churn_df.drop(columns = drop_cols)
    
    return telco_churn_df.T

def dmy_conversion(df, datetime_column):
    """
    DESCRIPTION:
    This function ensures the datetime_column given as an argument is converted to dtype of datetime64.
    Then adds Day, Month, and Year columns and sets the index to the datetime_column
    ___________________________________
    IMPORTS REQUIRED:
    import pandas as pd
    from datetime import timedelta, datetime
    ___________________________________
    ARGUMENTS:
                 df = DataFrame
    datetime_column = The 'column_name' of the column being used to store Date and Time data as datetime data type.
    """
    
    # Ensure datetime_column is dtype datetime64
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    
    # Convert datetime_column column to Day, Month, Year
    df['day'] = df[datetime_column].dt.day
    df['day_of_week'] = df[datetime_column].dt.day_name()
    df['weekday_number'] = df[datetime_column].dt.day_of_week+1
    df['year'] = df[datetime_column].dt.year
    df['month'] = df[datetime_column].dt.month_name()
    df['month_number'] = df[datetime_column].dt.month
    #df['hour'] = df[datetime_column].dt.hour
    #df['minute'] = df[datetime_column].dt.minute
    #df['second'] = df[datetime_column].dt.second

    # Set index
    df = set_index(df, datetime_column)

    # FUTURE FUNCTIONALITY
    # Use IF statements and D,M,Y,H,Min,Sec arguments to determin layers of conversion
    
    return df

def prep_store_data(df, datetime_column):
    """
    Combine functions needed to prepare store data for use. 
    """
    
    # Convert sale_date column to datetime format; Add Day, Month, Year columns; set Index to sale_date 
    dmy_conversion(df, datetime_column)
    
    # Create Sales Total column from Item Price multiplied by Sales Amount
    df['sales_total'] = df.sale_amount * df.item_price
    
    return df

def set_index(df, datetime_column):
    df = df.set_index(datetime_column).sort_index()
    
    return df

# ---------- #
# SPLIT DATA #
# ---------- #

def train_val_test_split(df, target):
    train, test = train_test_split(df, test_size=.15, random_state=1992, stratify = df[target])
    train, validate = train_test_split(train, test_size=.25, random_state=1992, stratify = train[target])
    print(' _____________________________________________________________ ')
    print('|                              DF                             |')
    print('|-------------------------------------------------------------|')
    print('|       Train       |       Validate    |          Test       |')
    print('|-------------------|-------------------|---------------------|')
    print('| x_train | y_train |   x_val  |  y_val |   x_test  |  y_test |')
    print(':_____________________________________________________________: ')
    print('')
    print(f"   Train: {train.shape[0]} rows {round(train.shape[0]/df.shape[0],2)}%")
    print(f"Validate: {validate.shape[0]} rows {round(validate.shape[0]/df.shape[0],2)}%")
    print(f"    Test: {test.shape[0]} rows {round(test.shape[0]/df.shape[0],2)}%")

    return train, validate, test