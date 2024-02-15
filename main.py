import pandas as pd
import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Set random seed
np.random.seed(500)

# Load the dataset
Corpus = pd.read_csv('data/bbc-text.csv', delimiter=',', encoding='latin-1')

# Display information about the dataset
print(Corpus.info())

# Visualize category distribution
plt.figure(figsize=(10, 8))
sns.countplot(x='category', data=Corpus)
plt.title('Count of Each Category', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.yticks(fontsize=12)
plt.show()

# Preprocessing
Corpus['text'].dropna(inplace=True)
Corpus['text_original'] = Corpus['text']
Corpus['text'] = [entry.lower() if isinstance(entry, str) else '' for entry in Corpus['text']]
Corpus['text'] = [word_tokenize(entry) if isinstance(entry, str) else [] for entry in Corpus['text']]

# Mapping POS tags to WordNet tags
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

# Lemmatization and filtering stopwords
word_Lemmatizer = WordNetLemmatizer()
for index, entry in enumerate(Corpus['text']):
    Final_words = []
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatizer.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    Corpus.loc[index, 'text_final'] = ' '.join(Final_words)

# Saving preprocessed data
output_path = 'preprocessed_data.csv'
Corpus.to_csv(output_path, index=False)

# Splitting the data into training and testing sets
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'], Corpus['category'], test_size=0.3)

# Encoding labels
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# TF-IDF Vectorization
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# Support Vector Machine Classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)

# Evaluating SVM classifier
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)
print(classification_report(Test_Y, predictions_SVM))



# Text Classification
new_text = input("Text: ")

# Preprocess the new text
new_text_lower = new_text.lower()
new_text_tokens = word_tokenize(new_text_lower)
Final_words = []

for word, tag in pos_tag(new_text_tokens):
    if word not in stopwords.words('english') and word.isalpha():
        word_Final = word_Lemmatizer.lemmatize(word, tag_map[tag[0]])
        Final_words.append(word_Final)

new_text_final = ' '.join(Final_words)
new_text_tfidf = Tfidf_vect.transform([new_text_final])

# Predict the category
predicted_category = SVM.predict(new_text_tfidf)

# Decode the predicted category using the LabelEncoder
decoded_category = Encoder.inverse_transform(predicted_category)
print("\nPredicted Category: ", decoded_category[0])
