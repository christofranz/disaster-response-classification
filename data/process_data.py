import sys

import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories into a merged dataframe.

    :param message_filepath:  Path to the csv file containing the messages
    :type: string
    :param categories_filepath: Path to the csv file containing the categories
    :type: string
    
    :return: Pandas dataframe of the messages and categories
    """
    # load the csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge the dataframes on the id and return
    df = pd.merge(messages, categories, on=["id"])
    return df



def clean_data(df):
    """
    Clean the dataframe with the messages and categories.

    :param df: Pandas dataframe containing the messages and categories
    :return: Cleaned dataframe
    """
    # create a dataframe with separate columns for each category
    categories = df.categories.str.split(pat=";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # lambda function to obtain the category name from the string
    get_category = lambda x: x[:-2]
    # all names of the categories in a list
    category_colnames = [get_category(category_value) for category_value in row]
    # rename columns accordingly
    categories.columns = category_colnames

    # convert category values to numbers
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop original category column
    df.drop(["categories"], axis=1, inplace=True)

    # concatenate the new categories with the messages
    df = pd.concat([df, categories], axis=1)

    # drop duplicate messages
    df.drop_duplicates(subset="message", inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save cleaned dataframe with as a sqlite database.

    :param df: Pandas dataframe to store
    :param database_filename: Filename for the database
    e.g. DistasterResponses.db
    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("DisasterResponses", engine, if_exists="replace", index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print("Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}"
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)
        
        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)
        
        print("Cleaned data saved to database!")
    
    else:
        print("Please provide the filepaths of the messages and categories "\
              "datasets as the first and second argument respectively, as "\
              "well as the filepath of the database to save the cleaned data "\
              "to as the third argument. \n\nExample: python process_data.py "\
              "disaster_messages.csv disaster_categories.csv "\
              "DisasterResponse.db")


if __name__ == "__main__":
    main()