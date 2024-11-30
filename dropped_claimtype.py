
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
position = reviews_df['claim_type'].value_counts() # veriler dengeli dağılmış

# service_id üzerinden 3 datayı birleştirelim
# reviews ile service datasını birleştirelim
reviews_services = reviews_df.merge(services_df, on="service_id", how="left")
reviews_services.info()

# sonucu summary data ile birleştirelim
data_df = reviews_services.merge(summary_df, on="service_id", how="left")
data_df.info()

# datanın son halinin ilk 5 satırını gözlemleyelin
data_df.head()

#############################################################################################
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


from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# BERT modelini yükleme
model = SentenceTransformer('all-MiniLM-L6-v2')

# Yorumları vektörleştirme
embeddings = model.encode(data_df['cleaned_review_text'].tolist())

# Kümeleme (örneğin, 4 sınıf için KMeans)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Küme etiketlerini veri setine ekleme
data_df['predicted_cluster'] = clusters

# Küme dağılımlarını inceleme
cluster_distribution = data_df['predicted_cluster'].value_counts()
cluster_distribution


# Her kümeden birkaç örnek yorum
for cluster_id in range(n_clusters):
    print(f"Küme {cluster_id}:")
    examples = data_df[data_df['predicted_cluster'] == cluster_id]['cleaned_review_text'].head(10)
    print(examples.to_list())
    print("\n")


from sklearn.feature_extraction.text import CountVectorizer

for cluster_id in range(n_clusters):
    cluster_reviews = data_df[data_df['predicted_cluster'] == cluster_id]['cleaned_review_text']
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    word_counts = vectorizer.fit_transform(cluster_reviews).toarray().sum(axis=0)
    keywords = vectorizer.get_feature_names_out()
    print(f"Küme {cluster_id} için anahtar kelimeler:")
    print(dict(zip(keywords, word_counts)))
    print("\n")


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# PCA ile boyut indirgeme
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Görselleştirme
plt.figure(figsize=(10, 6))
for cluster_id in range(n_clusters):
    plt.scatter(
        reduced_embeddings[clusters == cluster_id, 0],
        reduced_embeddings[clusters == cluster_id, 1],
        label=f"Küme {cluster_id}"
    )
plt.legend()
plt.title("Kümeleme Görselleştirmesi")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
