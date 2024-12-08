###################################################################
# Patika.dev & NewMind AI Bootcamp Bitirme Projesi
###################################################################

# Gerekli kütüphanelerin import edilmesi
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import os
import joblib
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from evaluate import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Veriler
data_df = pd.read_csv('data/merged_data.csv')
summary_file = 'data/summary.csv' # kategorilere ait özetler
model_file = 'classifier_model.joblib'

#################################################################
# Classification için Modelleme
#################################################################
# Eğitim ve test verisini ayırma
X = data_df["cleaned content"]
y = data_df["category"]

# Label Encoding: Kategorik etiketleri sayısal hale getirme
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Etiketleri sayılara dönüştür

# Eğitim ve test setleri
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# Modeli yükleme , eğitim yapma
if os.path.exists(model_file):
    pipeline_lr = joblib.load(model_file)
else:
    pipeline_lr = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("classifier", xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"))
    ])
    pipeline_lr.fit(X_train, y_train)
    joblib.dump(pipeline_lr, model_file)

#################################################################
# Özet çıkarma ve metin üretme
#################################################################

# Hugging Face BART modelini ve tokenizer'ını yükleme (özetleme için)
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Hugging Face DistilGPT-2 modelini pipeline ile yükleme (metin genişletme için)
text_generator = pipeline("text-generation", model="distilgpt2")

# Kategoriler için promptlar
category_prompts = {
    'business': "Summarize the key points from contents related to business, including key trends in trade, companies, and economics.",
    'technology': "Summarize the key points from contents related to technology, including innovations in AI, robotics, and other technological developments.",
    'sports': "Summarize the key points from contents related to sports, including key events, teams, and athletes.",
    'education': "Summarize the key points contents related to education, including schools, policies, and students.",
    'entertainment': "Summarize the key points contents related to entertainment, including movies, concert, scenes, music, TV shows, and celebrity news."
}

def clean_redundancy(text, prompt): # Çıktılarda promptu temizleme
    return text.replace(prompt, "").strip()

def summarize_text(text, prompt, max_length=350, min_length=150): # Metin özetleme
    inputs = bart_tokenizer(f"{prompt} {text}", return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(
        inputs["input_ids"], max_length=max_length, min_length=min_length, num_beams=4, early_stopping=True
    )
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return clean_redundancy(summary, prompt)

def generate_text_expansion(text, max_length=1000): # Metin genişletme
    expansion_prompt = f"Expand on the following article by adding more details and background information. Make the text more informative and engaging. {text}"
    generated_text = text_generator(expansion_prompt, max_length=max_length, num_return_sequences=1)[0]['generated_text']
    return clean_redundancy(generated_text, expansion_prompt)

# Her kategoriye ait var olan verileri özetleme ve summary.csv dosyasına kaydetme
if not os.path.exists(summary_file):
    summaries = []
    for category in data_df['category'].unique():
        category_data = data_df[data_df['category'] == category]["content"].tolist()
        category_text = " ".join(category_data)
        prompt = category_prompts.get(category)
        summary = summarize_text(category_text, prompt)
        summaries.append({'category': category, 'summary': summary})
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(summary_file, index=False)
else:
    summary_df = pd.read_csv(summary_file)


#################################################################
# STREAMLIT Arayüzü
#################################################################

st.set_page_config(layout="wide")

# Streamlit başlığı
st.title("News Article Categorization, Summarization, Generation")

# İki sütunlu düzen oluşturma
col1, col2 = st.columns(2)

with col1:
    # Kullanıcıdan giriş alma
    user_input = st.text_area("Please enter a news article:",height=150)

    # Submit butonu
    if st.button('Submit'):
        if user_input:
            # Kategoriyi tahmin etme
            predicted_category_encoded = pipeline_lr.predict([user_input])[0]

            # Sayısal tahmini metin kategorisine dönüştürme
            predicted_category = label_encoder.inverse_transform([predicted_category_encoded])[0]

            # Kategorinin mevcut özetini al
            existing_summary = summary_df[summary_df['category'] == predicted_category]['summary'].values[0]

            # Yeni metnin özetini çıkar (BART kullanarak)
            prompt = category_prompts.get(predicted_category)
            new_summary = summarize_text(user_input, prompt, max_length=40, min_length=25)

            # Metni genişletme (DistilGPT-2 kullanarak)
            expanded_text = generate_text_expansion(user_input, max_length=500)

            # ROUGE ve BLEU metriklerini hesaplama
            rouge = load("rouge")
            bleu = load("bleu")

            rouge_score = rouge.compute(predictions=[new_summary], references=[user_input])
            bleu_score = bleu.compute(predictions=[new_summary], references=[[user_input]])

            # Model performansını hesaplama
            y_pred_lr = pipeline_lr.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_lr)
            precision = precision_score(y_test, y_pred_lr, average='weighted')
            recall = recall_score(y_test, y_pred_lr, average='weighted')
            f1 = f1_score(y_test, y_pred_lr, average='weighted')

            # Sol sütunda skorları göster
            with col1:
                st.write("### Model Performance Metrics:")
                st.write(f"**Accuracy**: {accuracy:.4f}")
                st.write(f"**Precision**: {precision:.4f}")
                st.write(f"**Recall**: {recall:.4f}")
                st.write(f"**F1 Score**: {f1:.4f}")

                st.write("### ROUGE Scores:")
                st.write(f"**ROUGE-1**: {rouge_score['rouge1']}")
                st.write(f"**ROUGE-2**: {rouge_score['rouge2']}")
                st.write(f"**ROUGE-L**: {rouge_score['rougeL']}")

                st.write("### BLEU Score:")
                st.write(f"**BLEU**: {bleu_score['bleu']:.4f}")

            # Sağ sütunda tahmin ve sonuçları göster
            with col2:
                st.write(f"### Predicted Category: {predicted_category}")
                st.write(f"### Input Summary:")
                st.write(new_summary)
                st.write(f"### Summary of other news in {predicted_category} Category:")
                st.write(existing_summary)
                st.write(f"### Expanded Text:")
                st.write(expanded_text)
        else:
            st.warning("Please enter a news article to proceed.")

