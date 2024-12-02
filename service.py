###################################################################
# Patika.dev & NewMind AI Bootcamp Bitirme Projesi
# Service Reviews Analysis and Conclusion Generation
###################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape) # satır sütun sayısı
    print("##################### Types #####################")
    print(dataframe.dtypes) # tip bilgileri
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("################ NA #################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    # sayısal sütunlar için temel istatistiksel ölçümleri (ortalama, standart sapma, minimum, maksimum gibi) belirli yüzdelik dilimlerle hesapla
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# müşteri verisi ve satış verisini inceleyelim
data_df = pd.read_csv("data/all_tickets_processed_improved_v3.csv")


check_df(tickets_df)
tickets_df.info()

# "Ticket Description" -> review textimiz
# "Ticket Type" -> target olacak


# sınıflandırma için kullanacağımız sonuçları
#data_df = tickets_df[["Ticket Description", "Ticket Type", "Product Purchased", "Ticket Priority", "Ticket Channel"]]
data_df.info()
check_df(data_df) # eksik veri yok

# Ticket Type ları inceleyelim
data_df["Ticket Type"].unique()
data_df["Ticket Type"].nunique()

# Ticket Descriptionları inceleyelim
data_df["Ticket Description"].head()
data_df["Ticket Description"].nunique()
position = data_df['Ticket Description'].value_counts()


data_df[["Ticket Description","Ticket Type"]].head()
data_df['Ticket Type'].value_counts()
###################################################################


# data_df[review_text] için preprocess uygulayalım

from warnings import filterwarnings
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

def preprocess_reviews(text):

    # ilk atılması gereken adımlardan biri tüm harfleri belirli bir standarda koymak (normalize etmek) çünkü bazıları büyük bazıları küçük
    text = text.lower() # hepsini küçük harf yapalım

    # Noktalama İşaretleri ( Punctuations )
    text = re.sub(r'[^\w\s]', '', text) # noktalama işaretlerinin yerine boşluk getir

    # Numbers
    text = re.sub(r'\d+', '', text)

    # Stopwords -> dilde anlam taşımayan kelimeler
    import nltk
    # nltk.download('stopwords')
    sw = stopwords.words('english')

    # metinlerde her satırı gezip stopwords varsa onları silmeliyiz ya da stopwords dışındakileri seçmeliyiz
    # öncelikle cümleleri boşluklara göre split edip list comp yapısıyla kelimelerin hepsini gezip stopwords olmayanları seçelim, seçtiklerimizi tekrar join ile birleştirelim
    text = " ".join(x for x in text.split() if x not in sw)

    # Rarewords -> nadir geçen kelimeler
    # nadir geçen kelimeleri çıkarmak için kelimelerin frekansını hesaplayıp kaç kere geçtiğini hesaplamalıyız
    # temp_df = pd.Series(' '.join(text).split()).value_counts()

    # frekansı 1 ya da 1 den küçük olanları drop edelim
    # drops = temp_df[temp_df <= 1]
    # text = " ".join(x for x in text.split() if x not in drops)

    # Lemmatization -> kelimeleri köklerine ayırmak, stemming metodu da aynı amaçla kullanılır
    # nltk.download('wordnet')
    text = " ".join([Word(word).lemmatize() for word in text.split()])

    return text


data_df["cleaned Document"] = data_df["Document"].apply(preprocess_reviews)
data_df.columns
data_df['Document'].head(10)
#############################################################################################
data_df.dropna(subset=["Ticket Description"], inplace=True)

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text
data_df["cleaned ticket description"] = data_df["Ticket Description"].apply(clean_text)


# Splitting the dataset into train and test sets
X = data_df["Document"]  # Feature: Text data
y = data_df["Topic_group"]  # Target: Ticket type

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Creating a pipeline for TF-IDF and Random Forest Classifier
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),  # TF-IDF for feature extraction
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))  # Random Forest for classification
])

# Training the model
pipeline.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)

# Display the classification report
print(report)