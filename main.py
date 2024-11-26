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
reviews_df = pd.read_csv("data/healthpulse_reviews.csv")
services_df = pd.read_csv("data/healthpulse_services.csv")
summary_df = pd.read_csv("data/healthpulse_summary.csv")

print("yorumlar/n")
check_df(reviews_df)

print("sağlık hizmetleri bilgileri/n")
check_df(services_df)

print("özet bilgileri/n")
check_df(summary_df)

reviews_df.info()
services_df.info()
summary_df.info()
reviews_df["position"].unique()
reviews_df["review_text"].head()
position = reviews_df['position'].value_counts()

# reviews_df[review_text] için preprocess uygulayalım

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
    temp_df = pd.Series(' '.join(text).split()).value_counts()

    # frekansı 1 ya da 1 den küçük olanları drop edelim
    drops = temp_df[temp_df <= 1]
    text = " ".join(x for x in text.split() if x not in drops)

    # Lemmatization -> kelimeleri köklerine ayırmak, stemming metodu da aynı amaçla kullanılır
    # nltk.download('wordnet')
    text = " ".join([Word(word).lemmatize() for word in text.split()])

    return text


reviews_df["cleaned_review_text"] = reviews_df["review_text"].apply(preprocess_reviews)
reviews_df.columns
reviews_df[['review_text', 'cleaned_review_text']].head()


# tf-idf uygulayalım
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Özellikler ve etiketler
X = reviews_df['cleaned_review_text']
y = reviews_df['position']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF dönüşümü
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Logistic Regression modeli
logistic_model = LogisticRegression()
logistic_model.fit(X_train_tfidf, y_train)

# Tahmin
y_pred = logistic_model.predict(X_test_tfidf)

# Performans değerlendirme
print(classification_report(y_test, y_pred))

reviews_df["cleaned_review_text"]
reviews_df["review_text"]

#####################################################
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# NLTK bileşenlerini indirme
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# Sentiment ifadelerinin korunması
def preserve_sentiment_phrases(text):
    # Duygusal ifadeler: olumsuz ve olumlu ifadeleri koruyoruz
    sentiment_phrases = ['disappointing', 'fantastic', 'not recommend', 'would not recommend', 'unprofessional',
                         'professional', 'well-maintained', 'would recommend', 'great', 'excellent']

    for phrase in sentiment_phrases:
        # Bu ifadeleri korumak için metni işaretle (daha sonra geri getirebilmek için)
        text = re.sub(r'\b' + re.escape(phrase) + r'\b', f" {phrase}_PHRASE ", text, flags=re.IGNORECASE)

    return text


# Ön işleme fonksiyonu
def preprocess_reviews(text):
    # 1. Sentiment ifadelerini koruma
    text = preserve_sentiment_phrases(text)

    # 2. Küçük harfe dönüştürme
    text = text.lower()

    # 3. Noktalama işaretlerini kaldırma
    text = re.sub(r'[^\w\s]', '', text)

    # 4. Sayıları kaldırma
    text = re.sub(r'\d+', '', text)

    # 5. Stopwords'leri kaldırma
    sw = stopwords.words('english')
    text = " ".join(x for x in text.split() if x not in sw)

    # 6. Rare words (nadiren geçen kelimeler) kaldırma
    temp_df = pd.Series(' '.join(text).split()).value_counts()
    drops = temp_df[temp_df <= 1]
    text = " ".join(x for x in text.split() if x not in drops)

    # 7. Lemmatization (kelimeleri köklerine ayırma)
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    # 8. Sentiment ifadelerini geri getirme
    sentiment_phrases = ['disappointing', 'fantastic', 'not recommend', 'would not recommend', 'unprofessional',
                         'professional', 'well-maintained', 'would recommend', 'great', 'excellent']
    for phrase in sentiment_phrases:
        text = text.replace(f" {phrase}_PHRASE ", phrase)

    return text


# Yorumları temizleme
reviews_df["cleaned_review_text"] = reviews_df["review_text"].apply(preprocess_reviews)

# Şimdi temizlenmiş veriye göz atalım
print(reviews_df[['review_text', 'cleaned_review_text']].head())

X = reviews_df['cleaned_review_text']
y = reviews_df['position']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF dönüşümü
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Logistic Regression modeli
logistic_model = LogisticRegression()
logistic_model.fit(X_train_tfidf, y_train)

# Tahmin
y_pred = logistic_model.predict(X_test_tfidf)

# Performans değerlendirme
print(classification_report(y_test, y_pred))
