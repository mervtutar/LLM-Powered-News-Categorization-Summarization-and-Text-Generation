###################################################################
# Patika.dev & NewMind AI Bootcamp Bitirme Projesi
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



business = pd.read_csv('data/business_data.csv')
business.info()
business.head()
education = pd.read_csv('data/education_data.csv')


entertainment = pd.read_csv('data/entertainment_data.csv')


sports = pd.read_csv('data/sports_data.csv')


technology = pd.read_csv('data/technology_data.csv')


# tüm verileri birleştir
dfs=[business,education,entertainment,sports,technology]
data_df = pd.concat(dfs)
data_df= data_df.sample(frac = 1).reset_index(drop = True) # rastgele sırala

data_df.head()
data_df.info()
data_df["content"].head()
check_df(data_df) # eksik veri yok
# "content" -> review textimiz
# "category" -> target olacak

# Topic_group ları inceleyelim
data_df["category"].unique()
data_df["category"].nunique()

# Descriptionları inceleyelim
data_df["content"].head()
data_df["content"].nunique()
position = data_df['content'].value_counts()
data_df['category'].value_counts() # veri düzgün dağılmış


data_df.to_csv('data/merged_data.csv', index=False)
###################################################################

data_df = pd.read_csv('data/merged_data.csv')
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


data_df["cleaned content"] = data_df["content"].apply(preprocess_reviews)
data_df.columns
data_df['cleaned content'].head(10)
data_df[['content','cleaned content']].head(10)

# Eğitim ve test verisini ayırma
X = data_df["cleaned content"]
y = data_df["category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model pipeline'ı oluşturma
pipeline_rf  = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),  # TF-IDF
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))  # RandomForest
])

# Modeli eğitme
pipeline_rf.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = pipeline_rf.predict(X_test)
print(classification_report(y_test, y_pred))  # Modelin performansını yazdırma



#############################################################################
# 2. kısım
# Hugging Face pipeline'ı ile metin genişletme
from transformers import pipeline
text_generator = pipeline("text-generation", model="distilgpt2")

# Kategorilere göre açıklamalar (İngilizce açıklamalar)
category_explanations = {
    'business': "This text is related to business, including topics such as trade, companies, and economics.",
    'technology': "This text is focused on technology and innovations like artificial intelligence, robotics, and other technological developments.",
    'sports': "This text is related to the sports world, including topics like football, basketball, and other popular sports.",
    'education': "This text is focused on education and academic topics, such as school systems, universities, and educational policies.",
    'entertainment': "This text is about entertainment, including movies, music, theater, and other forms of leisure."
}

# Kullanıcıdan metin almak
user_input = input("Please enter a news article: ")

# Kullanıcının girdiği metni sınıflandırma
predicted_category = pipeline_rf.predict([user_input])[0]
print(f"The predicted category of your text: {predicted_category}")

# Kategoriye dayalı açıklama
category_description = category_explanations.get(predicted_category, "No information available for this category.")

# LLM modelini kullanarak kategoriye uygun metin üretme
# Daha anlamlı ve bağlamsal bir metin üretmek için prompt'u şu şekilde güncelleyebiliriz:
generated_text = text_generator(f"Category: {predicted_category}. Explain what this category generally involves, and expand on it with specific and relevant details about the topic: {user_input}", max_length=200)

# Sonuçları yazdırma
print(f"Category Explanation: {category_description}")
print(f"Generated Text: {generated_text[0]['generated_text']}")

