# New-Mind-AI-Bootcamp-Final-Case
Patika.dev &amp; New Mind
This project was developed as part of the Patika.dev & NewMind AI Bootcamp capstone. It focuses on leveraging machine learning and NLP models to analyze, categorize, summarize, and expand news articles efficiently.
## 🚀 Project Overview
This project addresses the challenge of processing large volumes of news articles by providing a streamlined solution for:

**Categorization**: Classifying articles into predefined categories like Business, Technology, Sports, Education, and Entertainment.

**Summarization**: Generating concise summaries for each article using BART (Bidirectional and Auto-Regressive Transformers).

**Text Generation**: Expanding on articles to provide more informative and engaging content using DistilGPT-2.

## 🧰 Key Features
**Category Prediction**: Using machine learning models like XGBoost, articles are classified into their respective categories.

**Summarization**: Leveraging Hugging Face's BART model for high-quality and context-aware article summaries.

**Text Expansion**: Using GPT-2 to expand articles with additional context and details.

**Performance Metrics**: The project evaluates model performance using metrics like accuracy, F1-score, ROUGE, and BLEU.

**Streamlit Application**: An interactive interface for users to input articles and view predictions, summaries, and expanded texts.

## 🛠️ Installation
 
### Clone the repository:
git clone https://github.com/mervtutar/LLM-Powered-News-Categorization-Summarization-and-Text-Generation.git

### Install dependencies:
pip install -r requirements.txt

### Download the necessary pre-trained models:
BART: facebook/bart-large-cnn
DistilGPT-2: distilgpt2

### Run the Streamlit app:
streamlit run app.py


## 🧪 How to Use
1. Open the Streamlit app in your browser.
2. Input a news article in the provided text box.
3. Click Submit to see:
   - Predicted category
   - Summary of the article
   - Expanded version of the article
   - Category-specific summaries for other articles

## 📊 Performance Metrics

### Classification:
- Accuracy: XX%
- F1-Score: XX%

### Summarization:
- ROUGE-1: XX
- BLEU: XX

## 🛠️ Technologies Used
- **Python**: Core programming language
- **Scikit-learn**: For model building and evaluation
- **XGBoost**: Advanced tree-based classifier
- **Hugging Face Transformers**: For BART and DistilGPT-2
- **Streamlit**: Web application framework for deployment
- **NLTK and TextBlob**: For text preprocessing and lemmatization
- **Pandas and Matplotlib**: For data handling and visualization

