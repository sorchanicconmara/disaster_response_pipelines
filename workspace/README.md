# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains the classifier and saves it as a pkl
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


**NOTE:** Due to Git file size limitations, the classifier.pkl is not stored online. After cloning the repo, running the commands described in step 1 will create the classifier object in the models directory.

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
