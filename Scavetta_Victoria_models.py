import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def make_predictions():
    """
    Function to train models and make predictions
    """
    mbti = pd.read_csv('preprocessed_mbti.csv')
    mbti.drop('Unnamed: 0', axis=1, inplace=True)
    print(mbti.head())

    # Vectorize with TFIDF
    tv = TfidfVectorizer(min_df=.10, max_df=.70)
    tv_features = tv.fit_transform(mbti['posts'].values.astype('str'))
    vocabulary = np.array(tv.get_feature_names_out())
    tv_features = tv_features.todense()
    print(len(vocabulary))
    vocab_df = pd.DataFrame(data=tv_features, columns=vocabulary)
    print(vocab_df.head())

    # Binarize the four orthogonal types
    mbti = binarize_types(mbti)
    print(mbti.head())

    types = ['E/I', 'N/S', 'F/T', 'J/P']    # This is just being used for formatting/printing purposes
    x = tv_features
    models = []
    for i in range(0, 4):
        models.append(XGBClassifier(n_estimators=200, max_depth=2, nthread=8, learning_rate=0.2))
    for i in range(2, 6):
        y = mbti.iloc[:, i]
        # Split data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=15)
        models[i-2].fit(x_train, y_train)
        # Make predictions for test data
        y_pred = models[i-2].predict(x_test)
        predictions = [round(value) for value in y_pred]
        # Evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print(types[i-2], "Accuracy: %2f%%" % (accuracy * 100.0))

    # Predict MBTI for characters
    script = predict_mbti(models, tv, types)
    print(script.head(22))
    script.drop('Dialogue', axis=1, inplace=True)
    script.drop('E/I_prediction', axis=1, inplace=True)
    script.drop('N/S_prediction', axis=1, inplace=True)
    script.drop('F/T_prediction', axis=1, inplace=True)
    script.drop('J/P_prediction', axis=1, inplace=True)
    script.to_csv('Seinfeld_MBTI_predictions.csv')


def binarize_types(mbti):
    """
    Turns the four orthogonal types into binary\

    :param mbti:    The mbti dataset
    :return:        The dataset with binarized types
    """
    types = {'E': 0, 'I': 1, 'N': 0, 'S': 1, 'F': 0, 'T': 1, 'J': 0, 'P': 1}
    # The columns that need to be binarized are columns 2-5
    for i in range(2, 6):
        mbti.iloc[:, i] = mbti.iloc[:, i].map(types)
    return mbti


def predict_mbti(models, tv, types):
    """
    Predicts MBTI given new text

    :param models:  The four models trained on the orthogonal types
    :param tv:      The TFIDF vectorizer
    :param types:   The orthogonal types
    :return:        The dataframe with a new column containing the predicted MBTI
    """
    script = pd.read_csv('clean_script.csv')
    script.drop('Unnamed: 0', axis=1, inplace=True)
    print(script.head())

    # Vectorize with TFIDF
    tfidf = tv.transform(script['Dialogue'].values.astype('str')).todense()

    # Make predictions and store in new column
    for i in range(len(models)):
        script[f'{types[i]}_prediction'] = models[i].predict(tfidf)

    mbti_pred = []
    type_cols = [column for column in script.columns[2:6]]
    for mbti in np.array(script[type_cols]):
        mbti_pred.append(get_mbti(mbti))
    script['MBTI'] = mbti_pred
    return script


def get_mbti(binarized_type):
    """
    Turns the binarized MBTI back into a readable format

    :param binarized_type:   The binarized MBTI
    :return:                 The readable MBTI
    """
    type_dict = [{0: 'E', 1: 'I'}, {0: 'N', 1: 'S'}, {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]
    mbti = ''
    for i, j in enumerate(binarized_type):
        mbti += type_dict[i][j]
    return mbti
