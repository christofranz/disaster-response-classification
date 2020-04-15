import json

import joblib
import pandas as pd
import plotly
from flask import Flask, jsonify, render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def contains_words(message, words):
    """
    Checks if a message contains certain words.

    :param message: Text message to check
    :param words: List of words
    
    :return: Boolean if all words are contained in the message
    """
    for w in words:
        if str(message).lower().find(w) < 0:
            return False
    return True


# load data
engine = create_engine('sqlite:///../data/DisasterResponses.db')
df = pd.read_sql_table('DisasterResponses', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(
                        color = "grey"
                    )
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # occurences of the categories in the data
    category_occurence = list((df.iloc[:, 4:] != 0).sum().sort_values(ascending=False) / df.shape[0])
    category_names = list(df.iloc[:, 4:].columns)
    category_names = [cat.replace("_", " ") for cat in category_names]
    graphs.append(
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_occurence,
                    marker=dict(
                        color = category_occurence,
                        colorscale = 'solar_r'
                    )
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Proportion [%]"
                },
                'xaxis': {
                    'title': {
                        "text": "Category",
                        "standoff": 15
                    },
                    'tickangle': -45,
                    'automargin': True,
                    'tickfont': {'size': 9}
                }
            }
        }
    )

    # distribution of categories when text contains "need" and "water"
    df["contains_words"] = df["message"].apply(lambda x: contains_words(x, ["need", "water"]))
    df_need_water = df[df["contains_words"] == True]
    category_distribution = df_need_water.drop(["contains_words"], axis=1).iloc[:, 4:].sum().sort_values(ascending=False)

    graphs.append(
        {
            'data': [
                Pie(
                    labels=category_distribution.keys()[:15],
                    values=category_distribution[:15],
                )
            ],

            'layout': {
                'title': 'Distribution of Categories <br>'
                'when the Message Contains the words "need", "water"',
                'annotations': [dict(
                    x=0.5,
                    y=-0.25,
                    xref='paper',
                    yref='paper',
                    text='From all messages containing the words "need" and "water",' 
                    '<br> how is the distribution of the categories? One message can have <br>'
                    'several categories associated to it. Only the 15 categories with the highest <br>'
                    'distribution are considered for this visualization.',
                    showarrow = False)]
            }
        }
    )

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
