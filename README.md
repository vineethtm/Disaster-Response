# Disaster Response Pipeline Project
### Introduction
Primary objective is to classify real messages that were sent during disaster events. This machine learning pipeline categorises
these events so that one can send the messages to an appropriate disaster relief agency.

### Data Pipelines
#### ETL Pipeline
This part of the pipeline perform the Extract, Transform, and Load process. It will will read the dataset, clean the data, and then store it in a SQLite database
#### Machine Learning Pipeline
This part will split the data into a training set and a test set and create a machine learning pipeline that uses NLTK scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Final model is then exported  to a pickle file
#### Flask App
This will accept a message, make a prediction using the saved model & display the results on a webpage

### Structure of the Project

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
