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
reviews_df["claim_type"].unique()
reviews_df["claim_type"].nunique()
reviews_df["review_text"].nunique()
reviews_df["review_text"].head()
position = reviews_df['claim_type'].value_counts()

# sercvice_id üzerinden 3 datayı birleştirelim.
# reviews ile service datasını birleştirelim
reviews_services = reviews_df.merge(services_df, on="service_id", how="left")
reviews_services.info()

# sonucu summary data ile birleştirelim
data_df = reviews_services.merge(summary_df, on="service_id", how="left")
data_df.info()

# datanın son halinin ilk 5 satırını gözlemleyelin
data_df.head()

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
    # temp_df = pd.Series(' '.join(text).split()).value_counts()

    # frekansı 1 ya da 1 den küçük olanları drop edelim
    # drops = temp_df[temp_df <= 1]
    # text = " ".join(x for x in text.split() if x not in drops)

    # Lemmatization -> kelimeleri köklerine ayırmak, stemming metodu da aynı amaçla kullanılır
    # nltk.download('wordnet')
    text = " ".join([Word(word).lemmatize() for word in text.split()])

    return text


data_df["cleaned_review_text"] = data_df["review_text"].apply(preprocess_reviews)
data_df.columns
data_df['cleaned_review_text'].head(10)
#############################################################################################


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. TF-IDF Vektörizasyonu
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data_df['cleaned_review_text'])

# 2. Elbow Method ile Küme Sayısı Belirleme
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tfidf_matrix)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 3. K-Means Uygulama (Örnek: 4 Küme)
optimal_k = 4  # Elbow grafiğine göre belirlenir
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

# 4. Sonuçları Görselleştirme (İsteğe Bağlı PCA/TSNE ile)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(tfidf_matrix.toarray())
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Cluster Visualization')
plt.show()

####################
# Küme etiketlerini mevcut verilere ekleyelim
data_df['cluster'] = clusters

# Her küme için örnekler çıkaralım
for cluster_id in range(optimal_k):
    print(f"Küme {cluster_id} - Örnek Metinler:")
    cluster_texts = data_df[data_df['cluster'] == cluster_id]['review_text'].head(5)
    for text in cluster_texts:
        print(f"- {text}")
    print("\n")

# Küme ile service_name ve claim_type ilişkisini analiz edelim
cluster_analysis = data_df.groupby(['cluster', 'service_name', 'claim_type']).size().reset_index(name='count')
print(cluster_analysis)


print("merve")

from transformers import pipeline

import torch
print(torch.__version__)
print(torch.cuda.is_available())

# Özetleme modeli
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

for cluster_id in range(optimal_k):
    texts = " ".join(data_df[data_df['cluster'] == cluster_id]['review_text'].tolist())
    summary = summarizer(texts, max_length=50, min_length=25, do_sample=False)
    print(f"Küme {cluster_id} Özeti: {summary[0]['summary_text']}")


import tensorflow as tf
print(tf.__version__)  # TensorFlow sürümünü yazdırır
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


#############################################################################################
# tf-idf uygulayalım
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Özellikler ve etiketler
X = reviews_df['cleaned_review_text']
y = reviews_df['claim_type']

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
reviews_df[["review_text",   "claim_type"]]

