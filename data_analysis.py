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


# Topic_group ları inceleyelim
data_df["category"].unique()
data_df["category"].nunique()

# Descriptionları inceleyelim
data_df["content"].head()
data_df["content"].nunique()
position = data_df['content'].value_counts()
data_df['category'].value_counts() # veri düzgün dağılmış

# Kategori dağılımını görselleştirelim
category_counts = data_df['category'].value_counts()
content_lengths = data_df['content'].apply(lambda x: len(str(x))).value_counts()
plt.figure(figsize=(10, 5))
category_counts.plot(kind='bar', color='skyblue')
plt.title('Category Distribution')
plt.xlabel('Category')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
###################################################################
# Ön İşleme

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


    # Lemmatization -> kelimeleri köklerine ayırmak, stemming metodu da aynı amaçla kullanılır
    # nltk.download('wordnet')
    text = " ".join([Word(word).lemmatize() for word in text.split()])

    return text


data_df["cleaned content"] = data_df["content"].apply(preprocess_reviews)
data_df.columns
data_df['cleaned content'].head(10)
data_df[['content','cleaned content']].head(10)

data_df.to_csv('data/merged_data.csv', index=False)

