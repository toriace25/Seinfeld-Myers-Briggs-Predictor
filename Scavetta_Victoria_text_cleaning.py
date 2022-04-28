import pandas as pd
import nltk
import re
import spacy
from nltk import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
import contractions

nlp = spacy.load('en_core_web_sm')
tokenizer = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()


def clean():
    """
    Function to prepare and clean the datasets
    """
    # Cleaning MBTI dataset first
    mbti = pd.read_csv("mbti_1.csv")
    print(mbti.head())

    # Add columns for the 4 binary orthogonal types
    mbti['E/I'] = mbti['type'].apply(lambda x: x[0])
    mbti['N/S'] = mbti['type'].apply(lambda x: x[1])
    mbti['F/T'] = mbti['type'].apply(lambda x: x[2])
    mbti['J/P'] = mbti['type'].apply(lambda x: x[3])
    print(mbti.head())

    print("Cleaning MBTI dataset...")
    mbti['posts'] = mbti.posts.apply(lambda x: clean_post(x))

    print(mbti.head())

    mbti.to_csv('preprocessed_mbti.csv')

    # Now cleaning Seinfeld script
    script = pd.read_csv('scripts.csv')
    script.drop('Unnamed: 0', axis=1, inplace=True)
    script.drop('SEID', axis=1, inplace=True)
    script.drop('Season', axis=1, inplace=True)
    script.drop('EpisodeNo', axis=1, inplace=True)
    print(script.head())
    # Taking a subset of characters to evaluate
    characters = ['JERRY', 'GEORGE', 'ELAINE', 'KRAMER', 'NEWMAN', 'SUSAN', 'UNCLE LEO', 'HELEN', 'MORTY', 'FRANK',
                  'ESTELLE', 'MR. LIPPMAN', 'STEINBRENNER', 'JOE DIVOLA', 'MICKEY', 'BABU', 'WILHELM', 'PETERMAN',
                  'PUDDY', 'PITT', 'BANIA', 'SOUP NAZI']
    # Remove all dialogue from all other characters
    script = script[script['Character'].isin(characters)]
    # Take all the dialogue from each character and condense it into one row per character
    rows = []
    for character in characters:
        character_script = script[script['Character'] == character]
        script_list = character_script['Dialogue'].tolist()
        script_list = ' '.join([char_script for char_script in script_list])
        rows.append({'Character': character, 'Dialogue': script_list})
    new_script = pd.DataFrame.from_dict(rows)
    print(new_script.head(22))

    print("Cleaning Seinfeld dataset...")
    new_script['Dialogue'] = new_script.Dialogue.apply(lambda x: clean_script(x))
    print(new_script.head(22))

    new_script.to_csv('clean_script.csv')


def remove_stopwords(text):
    """
    Removes stopwords from the text
    :param text:            The text to remove stopwords from
    :return:                The text without stopwords
    """
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_urls(text):
    """
    Removes URLs from the text
    :param text:    The text to remove URLs from
    :return:        The text with no URLs
    """
    return re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<"
                  r">]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", '', text)


def remove_mbti(text):
    """
    Removes Myers-Briggs personality types mentioned in text
    :param text:    The text to remove types from
    :return:        The text without personality types mentioned
    """
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip(' ') for token in tokens]
    types = ["istj", "istp", "estp", "estj", "isfj", "isfp", "esfp", "esfj", "infj", "infp", "enfp", "enfj", "intj",
             "intp", "entp", "entj", "istjs", "istps", "estps", "estjs", "isfjs", "isfps", "esfps", "esfjs", "infjs",
             "infps", "enfps", "enfjs", "intjs", "intps", "entps", "entjs"]
    filtered_tokens = [token for token in tokens if token not in types]
    new_text = ' '.join(filtered_tokens)
    return new_text


def expand(text):
    """
    Expands all contractions in the text

    :param text: The text with contractions
    :return:     The text without contractions
    """
    expanded_words = []
    for word in text.split():
        # using contractions.fix to expand the shortened words
        expanded_words.append(contractions.fix(word))

    expanded_text = ' '.join(expanded_words)
    return expanded_text


def clean_post(text):
    """
    Function used to clean the MBTI dataset

    :param text:    The uncleaned text
    :return:        The cleaned text
    """
    # Get rid of ||| symbol, which separates the posts
    text = re.sub(r'\|\|\|', ' ', text).lower()
    text = remove_urls(text)
    text = expand(text)
    # Remove other unneeded symbols
    text = re.sub(r'[^a-z]', ' ', text)
    text = remove_mbti(text)
    # Get rid of extra white spaces
    text = re.sub(r' +', ' ', text)
    text = remove_stopwords(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split(' ')])
    return text


def clean_script(text):
    """
    Function used to clean the Seinfeld dataset

    :param text:    The uncleaned text
    :return:        The cleaned text
    """
    # Remove scene directions
    text = re.sub(r'\(.*\)', '', text).lower()
    text = expand(text)
    # Remove unneeded symbols
    text = re.sub(r'[^a-z]', ' ', text)
    # Get rid of extra white spaces
    text = re.sub(r' +', ' ', text)
    text = remove_stopwords(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split(' ')])
    return text
