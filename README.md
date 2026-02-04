
# 🎬 Production-Ready NLP Sentiment Analysis System

A **production-oriented NLP application** for movie review sentiment analysis using a **fine-tuned DistilBERT model**, deployed with **Streamlit** and built following **clean ML engineering practices**.

This project demonstrates **real-world ML inference**, **model caching**, **structured logging**, and **interactive visualization**, making it suitable for **ML Engineer / NLP Engineer roles**.

---

## 🚀 Demo Features

- ✅ Fine-tuned **DistilBERT** for binary sentiment classification
- ✅ Production-grade **Streamlit UI**
- ✅ Model caching for fast inference
- ✅ Confidence-aware predictions
- ✅ Probability visualization with Plotly
- ✅ Structured logging for debugging & monitoring
- ✅ Clean separation of UI and inference logic

---

## 🧠 Model Details

| Component | Description |
|---------|------------|
| Model | DistilBERT (fine-tuned) |
| Dataset | IMDB Movie Reviews |
| Task | Binary Sentiment Classification |
| Framework | PyTorch + HuggingFace |
| Accuracy | ~86.25% |
| Max Length | 512 tokens |

Model hosted on HuggingFace Hub:
```

asadullahshehbaz/my_text_classifier_model

```

---

## 🗂️ Project Structure

```

.
├── app.py                # Streamlit frontend (UI only)
├── main.py               # Model loading & inference logic
├── logg.py               # Centralized logging configuration
├── requirements.txt
└── README.md

````

**Design Principle:**  
> UI and ML logic are intentionally separated to reflect production ML systems.

---

## 🖥️ Application Flow

1. User enters a movie review
2. Text is tokenized using HuggingFace tokenizer
3. DistilBERT performs inference
4. Softmax probabilities are computed
5. Sentiment + confidence score returned
6. Results visualized in an interactive bar chart

---

## 📊 Output Example

- **Sentiment:** Positive 😊 / Negative 😞
- **Confidence:** Probability of predicted class
- **Visualization:** Probability distribution (Positive vs Negative)

Low-confidence predictions can be easily flagged in future iterations.

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/asadullahshehbaz/nlp-sentiment-analyzer.git
cd imdb-sentiment-analyzer
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 🧪 Inference Logic (Simplified)

```python
with torch.no_grad():
    outputs = model(**inputs)
    probs = softmax(outputs.logits)
```

Confidence is computed as:

```python
confidence = max(positive_prob, negative_prob)
```

---

## 📈 Engineering Highlights (Why This Is Production-Ready)

✔ Model caching using `@st.cache_resource`
✔ Clean inference abstraction
✔ Structured logging with severity levels
✔ Error-safe inference handling
✔ Production-friendly UI
✔ Extendable for API / batch inference

---

## ⚠️ Limitations

* Single-sentence inference only
* No explainability (SHAP / LIME) yet
* CPU inference (no GPU optimization)
* No batch processing in UI

---

## 🔮 Future Improvements

* 🔍 Add SHAP / LIME explainability
* 🚀 FastAPI backend for REST inference
* 📦 Dockerization
* 📊 Batch inference support
* 🧪 Unit tests for inference
* ☁️ Cloud deployment (AWS / GCP / Railway)

---

## 👨‍💻 Author

**Asadullah Shehbaz**
Machine Learning & NLP Engineer

* Kaggle Master
* PyTorch & HuggingFace Specialist
* Focused on production-grade AI systems

---

## ⭐ Why This Project Matters

This is **not a notebook demo**.
It reflects **real ML deployment thinking**, suitable for:

* NLP Engineer roles
* ML Engineer roles
* AI Engineer portfolios
* Freelance ML projects

---

## 📜 License

MIT License



