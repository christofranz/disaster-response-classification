import pickle
import re
import sys

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Load database and split into feature and response variables.

    :param database_filepath: Path to the database containing cleaned data
    :return: X - Pandas Series of the messages
             Y - Pandas Dataframe with the categories of the messages
             category_names: List of strings of the categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponses", engine)
    # feature variable
    X = df['message']
    # response variable
    Y = df.iloc[:,4:]
    # category names
    category_names = list(Y.columns)

    return X, Y, category_names

def tokenize(text):
    """
    Create tokens from the text.

    :param text: String of the text
    :return: List of tokens extracted from the text
    """
    # remove everything not alpha-numerical in the text
    # and normalize the text to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    
    # create tokens out of the text
    tokens = word_tokenize(text)
    
    # init stemmer, lemmatizer and stop words
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    # stemmatize and lemmatize the tokens
    stemmed = [stemmer.stem(w) for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in stemmed]
    
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # in case your hardware takes too long, feel free to comment out
    # some of the following parameters inside the dict
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    accuracy_list = []
    precision_list = []
    recall_list = []
    for idx, category in enumerate(category_names):
        # classification report per category
        report = classification_report(Y_test.iloc[:, idx], Y_pred[:, idx], output_dict=True)
        # accuracy, precision and recall for the category
        accuracy_list.append(accuracy_score(Y_test.iloc[:, idx], Y_pred[:, idx]))
        precision_list.append(report["weighted avg"]["precision"])
        recall_list.append(report["weighted avg"]["recall"])
        # print report for each category
        print(category)
        print(report)
        
    # print overall metrics
    print("Average accuracy: {}".format(np.mean(np.array(accuracy_list))))
    print("Average weighted precision: {}".format(np.mean(np.array(precision_list))))
    print("Average weighted recall: {}".format(np.mean(np.array(recall_list))))


def save_model(model, model_filepath):
    """
    Save model as a pickle file.

    :param model: Estimator model to be saved
    :param model_filepath: Total filepath where the model will be saved
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
