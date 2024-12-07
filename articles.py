###################################################################
# Patika.dev & NewMind AI Bootcamp Bitirme Projesi
###################################################################

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


data_df = pd.read_csv('data/merged_data.csv')
# Classification
data_df.columns
data_df[["cleaned content", "category"]]


# Eğitim ve test verisini ayırma
X = data_df["cleaned content"]
y = data_df["category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
# Modelleri tanımlama
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(),
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'LightGBM': LGBMClassifier()
}

# Her model için pipeline oluşturma ve performans değerlendirme
for model_name, model in models.items():
    # Model pipeline'ı oluşturma
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),  # TF-IDF
        ("classifier", model)  # Model
    ])

    # Modeli eğitme
    pipeline.fit(X_train, y_train)

    # Test verisi üzerinde tahmin yapma
    y_pred = pipeline.predict(X_test)

    # Modelin performansını yazdırma
    print(f"--- {model_name} ---")
    print(classification_report(y_test, y_pred))  # Modelin performansını yazdırma
    print("=" * 50)  # Ayırıcı çizgi'''

# Logistic Regression pipeline'ı oluşturma
pipeline_lr = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),  # TF-IDF
    ("classifier", LogisticRegression(max_iter=1000))  # Logistic Regression
])

# Modeli eğitme
pipeline_lr.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred_lr = pipeline_lr.predict(X_test)

# Modelin performansını yazdırma
print("Logistic Regression Model Performance:")
print(classification_report(y_test, y_pred_lr))  # Modelin performansını yazdırma


# confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Confusion Matrix hesaplama
cm = confusion_matrix(y_test, y_pred_lr)

# Confusion Matrix'i görselleştirme
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

#############################################################################
# 2. kısım
# Hugging Face pipeline'ı ile metin genişletme
from transformers import pipeline
import joblib  # Eğer modeli daha önce kaydettiyseniz yüklemek için kullanabilirsiniz
from evaluate import load

# Text generation modelini yükleme
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

# Kategoriyi tahmin etme - modelinize göre aşağıdaki satırı güncelleyin
predicted_category = pipeline_lr.predict([user_input])[0]  # 'pipeline_rf' burada RandomForest modelini temsil eder
print(f"The predicted category of your text: {predicted_category}")

# Kategoriye dayalı açıklama
category_description = category_explanations.get(predicted_category, "No information available for this category.")

# Daha anlamlı ve bağlamsal bir metin üretmek için prompt'u şu şekilde güncelleyebiliriz:
prompt = f"""
Category: {predicted_category}.
Description: {category_description}.
The text you provided is about the following topic: {user_input}.
Expand on the given news content and mention its place in the industry, providing more context and explanation."""

# LLM modelini kullanarak kategoriye uygun metin üretme
generated_text = text_generator(prompt, max_length=250, truncation=True)
generated_text_content = generated_text[0]['generated_text']

# Sonuçları yazdırma
print(f"Category: {predicted_category}")
print(f"Category Explanation: {category_description}")
# print(f"Generated Text: {generated_text[0]['generated_text']}")

# Sonuçları yazdırma (promptu dışarıda tutarak)
print(f"Generated Text: {generated_text[0]['generated_text']}")

# ROUGE ve BLEU metriklerini hesaplama
rouge = load("rouge")
bleu = load("bleu")

# ROUGE skorunu hesaplama
rouge_score = rouge.compute(predictions=[generated_text_content], references=[user_input])
print(f"ROUGE Score: {rouge_score}")

# BLEU skorunu hesaplama
bleu_score = bleu.compute(
    predictions=[generated_text_content],  # Tahmin edilen metni düz string olarak aktar
    references=[[user_input]]  # Referans metni içeren bir liste aktar
)

print(f"BLEU Score: {bleu_score}")