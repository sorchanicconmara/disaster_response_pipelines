# Disaster Response Pipeline

#### ETL Pipeline, ML Pipeline and Flask Web App for Analysing Message Data for Disaster Response

---

#### Author: Sorcha Nic Conmara | September 2021

---
### File Structure:

- **app/**
    - **templates/** - html templates used by Flask app
        - **go.html** - template for query input
        - **master.html** - template for home page
    - **run.py** - runs Flask app

- **data/** 
    - **disaster_categories.csv** - raw category data from Figure Eight
    - **disaster_messages.csv** - raw messages data from Figure Eight
    - **DisasterResponse.db** - SQLite database where data is stored
    - **process_data.py** - script to process data
    
- **models/**
    - **train_classifier.py** - script that trains and saves classifier
    
- **README.md** - readme

- **requirements.txt** - project dependencies

___

### Instructions:
1. Run the following commands in the project's root directory (workspace) to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains the classifier and saves it as a pkl
        `python models/train_classifier.py data/DisasterResponse.db models/classifier_ppl_grdsearch.pkl`
      
2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/

**NOTE:** Due to Git file size limitations, the classifier_ppl_grdsearch.pkl is not stored online. After cloning the repo, running the commands described in step 1 will create the classifier object in the models directory.

___

### Libraries Used (requirements.txt):
- click==8.0.1
- colorama==0.4.4
- Flask==2.0.1
- greenlet==1.1.1
- itsdangerous==2.0.1
- Jinja2==3.0.1
- joblib==1.0.1
- MarkupSafe==2.0.1
- nltk==3.6.2
- numpy==1.21.1
- pandas==1.3.1
- plotly==5.3.0
- python-dateutil==2.8.2
- pytz==2021.1
- regex==2021.8.3
- scikit-learn==0.24.2
- scipy==1.7.1
- six==1.16.0
- SQLAlchemy==1.4.22
- tenacity==8.0.1
- threadpoolctl==2.2.0
- tqdm==4.62.1
- Werkzeug==2.0.1
---

### Acknowledgements
* [Figure Eight | Data](https://appen.com/)
* [Udacity | Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)