# Seinfeld Myers-Briggs Predictor
This program analyzes the Seinfeld script and finds out the Myers-Briggs personality type of 22 Seinfeld characters. The
program includes four models that are trained to predict a person's Myers-Briggs personality type given some text. The
models are trained using a dataset containing posts from users on a forum and their personality type.

The file to run the program is Scavetta_Victoria_FinalProject.py.
The MBTI dataset and the Seinfeld script dataset are cleaned in Scavetta_Victoria_text_cleaning.py.
The models are trained and predictions are made using Scavetta_Victoria_models.py.

The results are stored in Seinfeld_MBTI_predictions.csv.

### Important:
To run the program, download the following Kaggle dataset: https://www.kaggle.com/datasets/datasnaek/mbti-type 

The file needs to be in the same folder as the Python files and should be named "mbti_1.csv". This file is too big to push onto GitHub.
