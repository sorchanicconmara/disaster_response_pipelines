import pickle
import re
import sys

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    '''
    Function to load data from sqllite databse, split data into features and labels and returning them.
    :param database_filepath: filepath for sqllite database with data saved from process_data.py save_data()
    :return:
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    table_name = database_filepath.replace("data/", "").replace(".db", "") + "_TABLE"
    df = pd.read_sql_table(table_name, engine)

    #drop any columns with a single unique value as these add nothing to model
    nunique_cols = df.columns[df.nunique() <= 1]
    df = df.drop(nunique_cols, axis=1)

    X = df.message
    y = df.iloc[:, 4:]
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    '''
    Function which takes test input and tokenizes it.
    :param text: text to be tokenized
    :return: clean_tokens - list of tokens from text
    '''
    # replace urls with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatizer to grouping forms of the same word into root form
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens


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


def build_model():
    '''
    Function to build model pipeline for the classifier.
    :return: pipeline
    '''
    ppl = Pipeline([
        ('Feats', FeatureUnion([

            ('Text_Ppl', Pipeline([
                ('Count_Vectoriser', CountVectorizer(tokenizer=tokenize)),
                ('Tfidf_Transf', TfidfTransformer())
            ])),

            ('Starting_Verb_Transf', StartingVerbExtractor())
        ])),

        ('Multi_Output_Classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return ppl


def evaluate_model(model_ppl, X_test, y_test, category_names):
    '''
    Function to apply model pipeline from build_model() to test set and print model evaluation metrics.
    :param model_ppl: machine learning model pipeline
    :param X_test: Features from test set
    :param y_test: Targets from test set
    :param category_names: Output labels
    :return:
    '''
    # predictions for test set
    y_pred = model_ppl.predict(X_test)

    cf_rep = classification_report(y_test, y_pred, target_names = category_names)

    print(80 * '*')
    print(cf_rep)
    print(80 * '*')
    print('Accuracy Score:\n', accuracy_score(y_test, y_pred))
    print(80 * '*')
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))


def save_model(model, model_filepath):
    '''
    Function to save model object as a PKL
    :param model: model object (pipeline)
    :param model_filepath: filepath to save model to
    :return:
    '''

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    '''
    Main function to train classifier.
    - Load data from sqlite database
    - Split data into training and testing sets
    - Build model
    - Train model using training set
    - Evaluate model performance using test set
    - Export trained model as PKL
    :return:
    '''
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