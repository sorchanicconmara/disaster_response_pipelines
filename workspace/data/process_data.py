import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function to read in csvs for raw messages and categories data and return them merged.
    :param messages_filepath: filepath for raw messages dataset
    :param categories_filepath: filepath for raw categories dataset
    :return: raw datasets merged
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner')

    return df


def clean_data(df):
    '''
    Function to clean raw merged dataset and return cleaned dataset.
    :param df: dataframe output from load_data()
    :return: cleaned dataframe
    '''
    # extract category names from categories column and assign as column names
    categories = df.categories.str.split(";", expand=True)
    category_colnames = categories.iloc[0].str.split('-', expand=True)[0]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-", expand=True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype('str').astype('int64')

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)

    return df


def save_data(df, database_filepath):
    '''
    Function to write dataframe to SQLlite DB using SQLAlchemy and pandas to_sql()
    :param df: dataframe output from clean_data()
    :param database_filepath: filepath for database to save data to
    :return: None
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    table_name = database_filepath.replace("data/", "").replace(".db", "") + "_TABLE"
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    '''
    Main function. Reads in CLI input from user, loads data, cleans data and saves data to database.
    :return:
    '''
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