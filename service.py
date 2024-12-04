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



# "Document" -> review textimiz
# "Topic_group" -> target olacak

data_df.info()
check_df(data_df) # eksik veri yok

# Topic_group ları inceleyelim
data_df["Topic_group"].unique()
data_df["Topic_group"].nunique()

# Descriptionları inceleyelim
data_df["Document"].head()
data_df["Document"].nunique()
position = data_df['Document'].value_counts()
data_df['Topic_group'].value_counts()

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
data_df['cleaned Document'].head(10)
#############################################################################################

'''def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text
data_df["cleaned ticket description"] = data_df["Ticket Description"].apply(clean_text)
'''

# Splitting the dataset into train and test sets
X = data_df["cleaned Document"]  # Feature: Text data
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


#############################################################################################
from transformers import pipeline
import joblib

# LLM modelini yükleyelim (Google FLAN-T5 Small)
llm_model = pipeline("text2text-generation", model="google/flan-t5-small")


# Sınıflandırma pipeline'ını daha önce eğittiğimiz model ile yeniden yükleyelim
# Bu adımda, modelin daha önce eğitilmiş versiyonunu yüklemeniz gerekebilir.
# Örneğin:
# pipeline = joblib.load("path_to_your_trained_model.pkl")

# LLM'yi kullanarak anlamlı sonuç (conclusion) üretme fonksiyonu
def generate_conclusion(predicted_label, description):
    # Örneğin, sınıflandırma sonuçlarını ve açıklamayı LLM'ye göndermek için uygun bir prompt hazırlayalım.
    prompt = f"""
    Customer support tickets are classified into the following types:
    - Account Access: 33%
    - Product Inquiry: 33%
    - Technical Issue: 34%

    Given the following description:
    "{description}"

    The predicted topic is: {predicted_label}

    Based on this information, generate a conclusion and suggest actions to improve customer satisfaction.
    """

    # LLM'yi çalıştırma
    response = llm_model(prompt, max_length=150, min_length=50, do_sample=False)
    return response[0]["generated_text"]


# Kullanıcıdan yeni bir açıklama almak için:
new_description = input("Please enter the customer review/complaint: ")

# Sınıflandırma modelini kullanarak tahmin yapalım
predicted_label = pipeline.predict([new_description])[0]  # Bu kısımda sınıflandırma modelinin tahminini alıyoruz.

# Sonuç üretme
conclusion = generate_conclusion(predicted_label, new_description)

# Çıktıyı gösterelim
print(f"Predicted Label: {predicted_label}")
print(f"Generated Conclusion: {conclusion}")
#################################################################################################################################


import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


from transformers import pipeline as hf_pipeline

# NLTK verilerini indir
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Veri okuma
data_df = pd.read_csv("data/all_tickets_processed_improved_v3.csv")


# Veri işleme fonksiyonu
def preprocess_reviews(text):
    # Küçük harfe çevirme
    text = text.lower()

    # Noktalama işaretlerini kaldırma
    text = re.sub(r'[^\w\s]', '', text)

    # Rakamları kaldırma
    text = re.sub(r'\d+', '', text)

    # Stopwords kaldırma
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    text = " ".join(x for x in text.split() if x not in sw)

    # Lemmatizasyon
    from textblob import Word
    text = " ".join([Word(word).lemmatize() for word in text.split()])

    return text


# Temiz veriyi işleme
data_df["cleaned Document"] = data_df["Document"].apply(preprocess_reviews)

# Eğitim ve test verisini ayırma
X = data_df["cleaned Document"]
y = data_df["Topic_group"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model pipeline'ı oluşturma
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),  # TF-IDF
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))  # RandomForest
])

# Modeli eğitme
pipeline.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))  # Modelin performansını yazdırma


# Kullanıcıdan yeni bir açıklama almak için
import pandas as pd
from transformers import pipeline as hf_pipeline

# Hugging Face pipeline ile model yükleme
summarizer = hf_pipeline("text2text-generation", model="google/flan-t5-small")


# Sınıflandırma modelinden alınan tahmin edilen etiket (predicted_label)
def get_predicted_label(description, pipeline):
    """ Verilen açıklamaya göre sınıflandırma yapalım """
    return pipeline.predict([description])[0]  # predicted label döndürür


# Summary için fonksiyon oluşturulması
# Summary için fonksiyon oluşturulması
def generate_conclusion(predicted_label, description, data_df, max_token_length=512):
    """
    generate_conclusion: Verilen predicted_label ve müşteri açıklaması ile genel bir özet oluşturur.

    predicted_label: Tahmin edilen etiket
    description: Müşteri şikayetinin açıklaması
    data_df: Veri seti
    """

    # Etiketin ait olduğu sınıfın verilerini filtrele
    class_descriptions = data_df[data_df['Topic_group'] == predicted_label]['cleaned Document']

    # Sınıftaki diğer açıklamaları birleştirerek genel bir özet oluşturuyoruz
    class_summary = " ".join(class_descriptions)

    # Token uzunluğuna dikkat etme
    # Eğer sınıf özetinin uzunluğu 512 token'ı aşarsa, sadece ilk 512 token'ı alıyoruz
    class_summary_tokens = class_summary.split()[:max_token_length]
    class_summary = " ".join(class_summary_tokens)

    # Hugging Face modeline verilen promptu tasarlıyoruz
    prompt = f"""

    A new customer complaint has been received: "{description}"

    Based on the classification above, the complaint falls under the "{predicted_label}" category.

    Here are some common issues faced by customers in the "{predicted_label}" category:
    {class_summary}

    Based on this, please generate a conclusion that summarizes the key points of the complaint and suggests actions for improving customer satisfaction.
    """

    # Model ile sonuç üretme
    response = summarizer(prompt, max_length=300, min_length=100, do_sample=False)
    return response[0]["generated_text"]

# Kullanıcıdan yeni bir açıklama almak
new_description = input("Please enter the customer review/complaint: ")

# Tahmin edilen etiket
predicted_label = get_predicted_label(new_description, pipeline)
print(f"Predicted Label: {predicted_label}")

# Sonuç üretme
conclusion = generate_conclusion(predicted_label, new_description, data_df)
print(f"Generated Conclusion: {conclusion}")

