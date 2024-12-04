
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
data_df.head()
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



import pandas as pd
from transformers import pipeline as hf_pipeline
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# Hugging Face T5 modeli yükleme
summarizer = hf_pipeline("text2text-generation", model="t5-base")

# Tahmin ve sonuç üretim fonksiyonu
def generate_conclusion(predicted_label, description, data_df, max_token_length=512):
    """
    Özetleme için LLM fonksiyonu
    """
    # Etiketle ilgili açıklamaları birleştirme
    class_descriptions = data_df[data_df['Topic_group'] == predicted_label]['cleaned Document']
    class_summary = " ".join(class_descriptions)

    # Token uzunluğunu sınırlama
    class_summary_tokens = class_summary.split()[:max_token_length]
    class_summary = " ".join(class_summary_tokens)

    # Özetleme için prompt oluşturma
    prompt = f"""
    A new customer complaint has been received: "{description}"

    Based on the classification above, the complaint falls under the "{predicted_label}" category.

    Here are some common issues faced by customers in the "{predicted_label}" category:
    {class_summary}

    Based on this, please generate a conclusion that summarizes the key points of the complaint and suggests actions for improving customer satisfaction.
    """
    response = summarizer(prompt, max_length=300, min_length=100, do_sample=False)
    return response[0]["generated_text"]

# Kullanıcı girdisi al ve tahmin yap
new_description = input("Please enter the customer review/complaint: ")
predicted_label = pipeline.predict([new_description])[0]
print(f"Predicted Label: {predicted_label}")

# LLM ile özet oluşturma
conclusion = generate_conclusion(predicted_label, new_description, data_df)
print("Generated Conclusion:")
print(conclusion)

##################################################################################

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Veri setini yükleyelim
data_df = pd.read_csv("data/all_tickets_processed_improved_v3.csv")

# Basit veri analizi fonksiyonu
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

check_df(data_df)

# Metin ön işleme fonksiyonu
def preprocess_reviews(text):
    import re
    from nltk.corpus import stopwords
    from textblob import Word

    # Lowercase
    text = text.lower()
    # Noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\s]', '', text)
    # Rakamları kaldır
    text = re.sub(r'\d+', '', text)
    # Stopwords temizliği
    sw = stopwords.words('english')
    text = " ".join(x for x in text.split() if x not in sw)
    # Lemmatization
    text = " ".join([Word(word).lemmatize() for word in text.split()])
    return text

# Metinleri temizleyelim
data_df["cleaned Document"] = data_df["Document"].apply(preprocess_reviews)

# Eğitim ve test verilerini ayıralım
X = data_df["cleaned Document"]
y = data_df["Topic_group"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sınıflandırma pipeline'ı
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Modeli eğitme
pipeline.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Hugging Face GPT-Neo-1.3B modeliyle özetleme
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Kullanıcıdan yeni bir şikayet alalım
new_description = input("Please enter the customer review/complaint: ")

# Random Forest sınıflandırma modeli ile tahmin edilen etiket
predicted_label = pipeline.predict([new_description])[0]
print(f"Predicted Label: {predicted_label}")

# Özetleme fonksiyonu
def generate_conclusion(predicted_label, description, data_df, tokenizer, model, max_token_length=512):
    """
    Özet oluşturma fonksiyonu.
    """
    # Etiketin ait olduğu sınıfın verilerini filtrele
    class_descriptions = data_df[data_df['Topic_group'] == predicted_label]['cleaned Document']
    class_summary = " ".join(class_descriptions)

    # Sınıf özeti için token sınırını uygulama
    class_summary_tokens = class_summary.split()[:max_token_length]
    class_summary = " ".join(class_summary_tokens)

    # Model için prompt tasarlama
    prompt = f"""
    A new customer complaint has been received: "{description}"

    Based on the classification, the complaint falls under the "{predicted_label}" category.

    Here are some common issues faced by customers in the "{predicted_label}" category:
    {class_summary}

    Based on this, please generate a conclusion that summarizes the key points of the complaint and suggests actions for improving customer satisfaction.
    """

    # Tokenize giriş
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_token_length)

    # Uzun metin üretme
    output = model.generate(
        input_ids,
        max_length=1024,
        min_length=300,
        do_sample=True,
        temperature=0.7,
        no_repeat_ngram_size=2
    )

    # Çıktıyı çözümle
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Özet oluşturma
conclusion = generate_conclusion(predicted_label, new_description, data_df, tokenizer, model)
print("Generated Conclusion:", conclusion)
