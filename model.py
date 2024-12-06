import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# 1. Küçük Boyutlu LLM Modelini Yükle
model_name = "google/flan-t5-small"  # Küçük bir T5 modeli
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. LLM Pipeline Oluştur
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# 3. Veriyi Yükle ve Hazırlık
data_df = pd.read_csv("data/all_tickets_processed_improved_v3.csv")


# Metinleri temizleyelim ve sınıflandırılmış metni yapılandıralım
def preprocess_reviews(text):
    import re
    from nltk.corpus import stopwords
    from textblob import Word

    # Lowercase, stopwords, lemmatization vb. işlemler
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    sw = stopwords.words('english')
    text = " ".join(x for x in text.split() if x not in sw)
    text = " ".join([Word(word).lemmatize() for word in text.split()])
    return text


data_df["cleaned Document"] = data_df["Document"].apply(preprocess_reviews)


# 4. LLM Giriş Metni Formatına Dönüştürme
def prepare_llm_input(batch):
    llm_input_text = ""
    for index, row in batch.iterrows():
        llm_input_text += f"Category: {row['Topic_group']}\n{row['cleaned Document']}\n\n"
    return llm_input_text


# 5. LLM Prompt Şablonu
prompt_template = """
Below are categorized user comments. Each category contains feedback that needs to be analyzed. 
Combine the insights from all categories into a concise, clear, and actionable summary.

{llm_input_text}

Generate a conclusion that addresses issues across all categories.
"""


# 6. LLM'e Giriş ve Çıktı Alma
def generate_conclusion_from_batch(batch):
    llm_input_text = prepare_llm_input(batch)

    # Promptu LLM'e gönder
    prompt = prompt_template.format(llm_input_text=llm_input_text)

    # LLM'e Giriş Sağla ve Sonucu Al
    result = llm_pipeline(prompt, max_length=500, num_beams=4, early_stopping=True)
    return result[0]['generated_text']


# 7. Veriyi Bölme: Toplu İşleme (Batched Processing)
batch_size = 1000  # Toplu işleme sayısı
batches = [data_df.iloc[i:i + batch_size] for i in range(0, len(data_df), batch_size)]

# 8. Sonuçları Birleştirme ve Kaydetme
full_conclusion = ""
for batch in batches:
    conclusion = generate_conclusion_from_batch(batch)
    full_conclusion += conclusion + "\n\n"

# 9. Sonuçları dosyaya yazma
with open("llm_generated_conclusion.txt", "w") as file:
    file.write(full_conclusion)

print("Inference ve sonuçlar başarıyla tamamlandı ve kaydedildi.")
