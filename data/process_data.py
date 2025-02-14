import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """ Merge two datasets
    Args:
    messages_filepath: file path of the disaster_messages.csv
    categories_filepath: file path of the disaster_categories.csv
    Returns:
    df: merged dataframe 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df =pd.merge(messages,categories,on='id')
    
    return df
    
def clean_data(df):
    """ Create new columns from a dataframe
    Args:
    df: dataframe to be cleaned
    Returns:
    df3: new dataframe with new columns
    """
    
    categories = df['categories'].str.split(';',expand=True)
    one_row=categories.loc[0]
    category_colnames = one_row.apply(lambda x:x.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x.split('-')[1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    for col in categories.columns:
        if len(categories[col].unique())==1:
            categories.drop(col,axis=1,inplace=True)
        elif len(categories[col].unique())>2:
                categories[col]=categories[col].apply(lambda x: 1 if x !=0 else 0) 
                
    df.drop(['categories'],axis=1,inplace=True)
    df2 = pd.concat([df,categories],axis=1)
    df3=df2.drop_duplicates(keep='first')
    
    return df3
    
    
    
    
        
def save_data(df, database_filename):
    """ save the cleaned dataset to database
    args:
    df: dataframe to be saved
    databese_filename: name of the database
    Returns: None
    """  
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('cleaned_messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()