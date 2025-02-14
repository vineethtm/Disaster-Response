import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

def load_data(database_filepath):
    """ load the data from the database and returns splitted data
    args:
    database_filepath: name of the database
    returns:
    X: message dataframe
    y: category columns
    categories: category names
    """
    
    engine=create_engine('sqlite:///'+database_filepath)
    df=pd.read_sql_table('cleaned_messages',con=engine)  
    X=df['message']
    y=df.iloc[:,4::]
    category_names=y.columns.tolist()
    return X,y,category_names


def tokenize(text):
    """ tokenise the text
    Args: 
    text:text to be tokenized
    Returns:
    clean tokens: tokenised text
    """   
    text=re.sub(r"[^a-zA-Z0-9]"," ",text)
    tokens=word_tokenize(text)   
    stop_words=stopwords.words('english') 
    tokens=[word for word in tokens if word not in stop_words]
    lemmatizer= WordNetLemmatizer()    
    clean_tokens=[]   
    for tok in tokens:
        clean_tok=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)    
    return clean_tokens

class array_transformer(BaseEstimator, TransformerMixin):
    """ convert a list to array"""

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_array = X.toarray()
        return X_array
    
f1_scorer=make_scorer(f1_score,average='weighted')


def build_model():
    """ Build a model
    Main steps: vectorize the text, perform PCA and apply Decions tree classifier
    Returns:
    cv: best model
   """
    pipeline=Pipeline([('tfidfvect',TfidfVectorizer(tokenizer=tokenize)),
                   ('toarray',array_transformer()),
                  ('scaler',StandardScaler()),('pca',PCA()),
                  ('clf',MultiOutputClassifier(estimator=DecisionTreeClassifier()))])
    
    parameters= {
        'tfidfvect__max_features':[5000],
        'pca__n_components':[100,200] ,
    'clf__estimator__max_depth':[5,10]}
    cv=GridSearchCV(pipeline,param_grid=parameters,scoring=f1_scorer,cv=5)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluate the performance of the trained model on test set
    Args:
    model: trained model
    X_test: Test data
    Y_test: Values for the different catgories of the test data
    category_names: Names of different categories
    Returns:
    print the f1 score, precision & recall for each categories
    """"  
    Y_test_pred=model.predict(X_test)
    for i in range(len(category_names)):   
        f1score=f1_score(Y_test.iloc[:,i],Y_test_pred[:,i])
        precisionscore=precision_score(Y_test.iloc[:,i],Y_test_pred[:,i])
        recallscore=recall_score(Y_test.iloc[:,i],Y_test_pred[:,i])
        print("For category '{}', \nf1-score:{} \nprecion:{} \nrecall:{} \n".
             format(category_names[i],f1score,precisionscore,recallscore))


def save_model(model, model_filepath):
    """ Save the final model
    Args:
    model: final model
    model_filepath: file name and path
    """
    
    pickle.dump(model,open(model_filepath,'wb'))


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