import joblib
import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.externals import joblib
from sqlalchemy import create_engine

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Class with functions to extract the starting verb in a sentence. Extra feature will be used by the classfier.
    '''

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse_TABLE', engine)

# load model
model = joblib.load("../models/classifier_ppl_grdsearch.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Used for distribution of message genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Used for distribution of message categories
    cats = df.iloc[:, 4:]
    cat_names = cats.columns
    cat_counts = (cats == 1).sum().sort_values(ascending=False)

    # Used for distribution of genre vs aid related status
    aid_related = df[df.aid_related == 1].groupby('genre').count()['message']
    not_aid_related = df[df.aid_related == 0].groupby('genre').count()['message']
    aid_types = aid_related.index

    # Used for distribution of genre vs medical help status
    med_help = df[df.medical_help == 1].groupby('genre').count()['message']
    no_med_help = df[df.medical_help == 0].groupby('genre').count()['message']
    med_help_types = med_help.index
    
    # create visuals
    graphs = [
        {
            "data": [
                {
                    "uid": "f4de1f",
                    "hole": 0,
                    "name": "Genre",
                    "pull": 0,
                    "type": "pie",
                    "domain": {
                        "x": [
                            0,
                            1
                        ],
                        "y": [
                            0,
                            1
                        ]
                    },

                    "textinfo": "label+value",
                    "hoverinfo": "all",
                    "labels": genre_names,
                    "values": genre_counts
                }
            ],
            "layout": {
                "title": "Distribution of Message Genres"
            },
            "frames": []
        },
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle' : 40,
                    'margin' : 30
                }
            }
        },
        {
            'data': [
                Bar(
                    x=aid_types,
                    y=aid_related,
                    name = "Aid Related"
                ),
                Bar(
                    x=aid_types,
                    y=not_aid_related,
                    name="Not Aid Related"
                ),
            ],

            'layout': {
                'title': 'Distribution of Message by Genre their "Aid Related" status',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=med_help_types,
                    y=med_help,
                    name="Medical Help"
                ),
                Bar(
                    x=med_help_types,
                    y=no_med_help,
                    name="No Medical Help"
                ),
            ],

            'layout': {
                'title': 'Distribution of Message by Genre their "Medical Help" status',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'group'
            }
        }

    ]
    
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
    app.run(host='0.0.0.0', port=3001, debug=False)


if __name__ == '__main__':
    main()