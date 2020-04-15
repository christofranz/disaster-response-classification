# Disaster Response Pipeline Project

### Table of Contents

1. [Installation and Instructions](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing](#licensing)

## Installation and Instructions<a name="installation"></a>

### Installation

You will need the standard data science libraries found in the Anaconda distribution of Python. Especially, the following packages are required:

- NumPy
- Pandas
- Matplotlib
- Plotly
- Nltk
- Flask
- Sklearn
- Sqlalchemy
- Pickle

Alternatively, you can directly install them with `pip install -r requirements.txt`. The code should run with no issues using Python versions 3.*.

### Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`. In case the script does not finish on your hardware, you can
        reduce the parameters for the gridsearch by outcommenting them in the `build_model` function of the script.

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ 

    - here you can type any text and the model will classify it into 36 different categories

## Project Motivation<a name="motivation"></a>

This project shall illustrate how ETL and ML-Pipelines can be utalized, using disaster response message classification as an example. Via the web app
text can be entered and it gets classified according to 36 different preselected categories. Keyword searching for categorizing response messages into
categories might have very limited performance because very seldomly when people talk about water they actually need it. On the other hand, people might
use the word thirsty instead of water. Hence,  machine learning algorithms are very interseting for this working area.

With the ETL pipeline data is extracted from a source (here csv-files), transformed or cleaned and loaded into a database with a different structure. When
new data is added, the pipeline can be executed just one more time and the data is available for further processing steps.

Machine Learning pipelines introduce higher simplicity and convenience for the users. Additionally, they can be used to optimize the entire workflow. For example,
Grid Search can be applied to optimize a set of different hyperparameters of a model. The pipeline also helps to prevent data leackage because all transformation
for the data preparation and feature extraction occur within each fold of the cross validation process.

## File Descriptions <a name="files"></a>

The most important files in this repository:

* `app/run.py` - This python file hosts the webapp via flask. The data visualization also takes place here.

* `data/process_data.py` - Contains the ETL pipeline that reads in the csv-files and creates a sqlite database.

* `model/train_classifier.py` - Contains the ML pipeline that trains the model, evaluates it on the test data and saves the model.

* `disaster_messages.csv` - Csv-file with sample response messages during real disasters.

* `disaster_categories.csv` - Csv-file the categories of each message (labeled data).

## Results<a name="results"></a>

heroku link

screenshots

## Licensing<a name="licensing"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
This app was completed as a project of the Udacity Data Science Nanodegree Program. The data was originally provided by Figure Eight.