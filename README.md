# AI-Generated-Text-Detection
Using Transformer Models and NLP to Detect AI-Generated Text on Social Media Platforms


## Overview
This project focuses on detecting **AI-generated content** on social media forums using **Natural Language Processing (NLP)** and **transformer-based models** like BERT. With the rise of **large language models (LLMs)** such as ChatGPT, online communities are increasingly vulnerable to AI-generated posts, which can impact the authenticity of discussions — especially in sensitive areas like **special needs education**.  

Our goal was to **scrape social media platform data**, generate synthetic AI posts, and build a **classification model** that accurately distinguishes between **human-written** and **AI-generated text**. The project demonstrates a complete workflow: **data collection**, **data preprocessing**, **model training**, **evaluation**, and **comparative analysis** across different subsocial media forumss.

---

##  Features
- **social media forums Data Scraping** using social media forums API and custom rate-limit handling.  
- **Synthetic AI Text Generation** using ChatGPT-4 to emulate social media forums-style content.  
- **Binary Classification Model** powered by **BERT** to detect AI-generated text.  
- **Custom Dataset Creation** with labeled `AI_text` and `human_text`.  
- **Evaluation Metrics:**
  - Precision, Recall, and F1-score
  - Confusion Matrix Visualization
  - Training & Validation Loss Graphs
- **Extensive Analysis** comparing niche subsocial media forumss (e.g., `r/afterspecialneeds`) and general social media forums communities.

---

##  Dataset
Two main datasets were used:

1. **Custom social media forums Dataset**  
   - Social Media Forums posts scraped before **November 30, 2022** (pre-ChatGPT release)  
   - Each post paired with a synthetically generated AI version  
   - Final structure:
     - `human_text` – Original social media forums post  
     - `AI_text` – ChatGPT-generated post  
   - Total size: **1,130 pairs**

2. **Hugging Face "Human AI Generated Text" Dataset**  
   - 1,000,000 samples of AI and human text  
   - Used for additional pretraining and benchmarking  
   - Columns:
     - `id`, `human_text`, `ai_text`, `instructions`

---

##  Technical Stack
- **Programming Language:** Python  
- **Libraries & Tools:**
  - `transformers` (Hugging Face)
  - `PyTorch`
  - `scikit-learn`
  - `pandas`
  - `matplotlib`
- **Infrastructure:**
  - Google Colab for training and experimentation
  - AWS for scalable computation

---

## Methodology

### 1. Data Collection
- Scraped social media forums data using **social media forums API** and libraries like `PRAW`.
- Implemented IP rate-limiting and anonymized user data.
- Subsocial media forums included: `specialneeds`, `special`, `Asksocial media forums`, `Mensa`, `Parenting`, etc.

### 2. Data Generation
- Used **ChatGPT-4** to generate synthetic posts matching the style and content of real social media forum posts.

### 3. Model Training
- **Fine-tuned BERT** for binary classification (human vs. AI text).
- Tokenized text using `bert-base-uncased`.
- Trained for **3 epochs** with:
  - 80% training data
  - 20% validation data

### 4. Evaluation
- Metrics:
  - **Accuracy:** 98.48%
  - **Precision & Recall:** >97% for both classes
- Visualization:
  - Training/validation loss over epochs
  - Confusion matrix
  - Prediction accuracy bar chart

---

## 📊 Results

| Model           | Precision (Human) | Recall (Human) | Precision (AI) | Recall (AI) | F1-Score | Accuracy |
|-----------------|-------------------|----------------|----------------|------------|----------|----------|
| Hugging Face Model | 0.00            | 0.00           | 0.50           | 1.00       | 0.67     | 50%      |
| social media forums Model (BERT) | **1.00**        | **0.97**       | **0.97**       | **1.00**   | **0.98** | **98.48%** |

- The **social media forums-trained model** significantly outperformed the Hugging Face pre-trained model.
- Achieved **high accuracy** and strong generalization on unseen social media forums data.


