# Sentiment Analysis Project on Twitter
# Manit Bhardwaj 

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import spacy

# Read the dataset with name "Emotion_classify_Data.csv" and store it in a variable df
columns = ['id','country','Label','Text']
df = pd.read_csv(r"C:\Manit\AIML\archive\twitter_training.csv", names=columns)

# Print the shape of dataframe
print(df.shape)

# Print top 5 rows
print(df.head(5))
    
df.info()

# Check the distribution of Emotion
df['Label'].value_counts()

# Show sample
for i in range(5):
    print(f"{i+1}: {df['Text'][i]} -> {df['Label'][i]}")

df.dropna(inplace=True)

# load english language model and create nlp object from it
nlp = spacy.load("en_core_web_sm") 

# use this utility function to get the preprocessed text data
def preprocess(text):
    # remove stop words and lemmatize the text
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    
    return " ".join(filtered_tokens) 

df['Preprocessed Text'] = df['Text'].apply(preprocess) 

df

le_model = LabelEncoder()
df['Label'] = le_model.fit_transform(df['Label'])
df.head(5)

X_train, X_test, y_train, y_test = train_test_split(df['Preprocessed Text'], df['Label'], 
                                                    test_size=0.2, random_state=42, stratify=df['Label'])
print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)

# Create classifier
clf = Pipeline([
    ('vectorizer_tri_grams', TfidfVectorizer()),
    ('naive_bayes', (MultinomialNB()))         
])
# Model training
clf.fit(X_train, y_train)

# Get prediction
y_pred = clf.predict(X_test)
# Print score
print(accuracy_score(y_test, y_pred))

# Print classification report
print(classification_report(y_test, y_pred))

clf = Pipeline([
    ('vectorizer_tri_grams', TfidfVectorizer()),
    ('naive_bayes', (RandomForestClassifier()))         
])
clf.fit(X_train, y_train)

# Get the predictions for X_test and store it in y_pred
y_pred = clf.predict(X_test)
# Print Accuracy
print(accuracy_score(y_test, y_pred))

# Print the classfication report
print(classification_report(y_test, y_pred))

test_df = pd.read_csv(r"C:\Manit\AIML\archive\twitter_validation.csv", names=columns)
test_df.head()

test_text = test_df['Text'][10]
print(f"{test_text} ===> {test_df['Label'][10]}")

test_text_processed = [preprocess(test_text)]
test_text_processed

test_text = clf.predict(test_text_processed)

classes = ['Irrelevant', 'Natural', 'Negative', 'Positive']

print(f"True Label: {test_df['Label'][10]}")
print(f'Predict Label: {classes[test_text[0]]}')