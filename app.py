import streamlit as st
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

st.set_page_config(page_title="Предиктор Сарказму", layout="centered")


def extract_linguistic_features(caption, ocr):
    caption = str(caption)
    ocr = str(ocr)

    len_caption = len(caption)
    len_ocr = len(ocr)

    caps_caption = sum(1 for c in caption if c.isupper())
    caps_ocr = sum(1 for c in ocr if c.isupper())

    punct_caption = len(re.findall(r'[!?]', caption))
    punct_ocr = len(re.findall(r'[!?]', ocr))

    sentiment_caption = TextBlob(caption).sentiment.polarity
    sentiment_ocr = TextBlob(ocr).sentiment.polarity

    sentiment_diff = abs(sentiment_caption - sentiment_ocr)

    sarcasm_markers = r'\b(wow|sure|great|thanks|yeah|right|literally|totally)\b'
    markers_caption = len(re.findall(sarcasm_markers, caption.lower()))
    markers_ocr = len(re.findall(sarcasm_markers, ocr.lower()))

    return [
        len_caption, len_ocr, caps_caption, caps_ocr,
        punct_caption, punct_ocr, sentiment_caption,
        sentiment_ocr, sentiment_diff, markers_caption, markers_ocr
    ]


@st.cache_resource
def load_and_train_model():
    df = pd.read_excel(r'sarcastic-classifier.xlsx')
    df['caption_text'] = df['caption_text'].fillna('')
    df['image_text'] = df['image_text'].fillna('')

    features_list = df.apply(lambda row: extract_linguistic_features(row['caption_text'], row['image_text']), axis=1)
    features_df = pd.DataFrame(features_list.tolist())

    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(features_df)

    df['combined_text'] = df['caption_text'] + " " + df['image_text']
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    text_features = tfidf.fit_transform(df['combined_text'])

    X = hstack([text_features, numerical_features])
    y = df['is_sarcastic']


    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X, y)

    return model, tfidf, scaler


model, tfidf, scaler = load_and_train_model()

st.title("Предиктор сарказму")

caption_input = st.text_area("Текст 1:",)
ocr_input = st.text_area("Текст 2: ")

if st.button("Predict", type="primary"):
    if not caption_input and not ocr_input:
        st.warning("Текст потрібен хоча б в одному з полів")
    else:
        ling_features = extract_linguistic_features(caption_input, ocr_input)
        scaled_features = scaler.transform([ling_features])

        combined_text = f"{caption_input} {ocr_input}"
        text_vec = tfidf.transform([combined_text])

        X_input = hstack([text_vec, scaled_features])

        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]

        confidence = probabilities[prediction] * 100

        st.divider()
        if prediction == 1:
            st.error("**Результат: Це сарказм!**")
        else:
            st.success("**Результат: Це НЕ сарказм.**")

        st.metric(label="Упевненість моделі", value=f"{confidence:.1f}%")
        st.progress(int(confidence))

        with st.expander("Переглянути параметри цього тексту"):
            st.write(f"- Тональність тексту 1: {ling_features[6]:.2f}")
            st.write(f"- Тональність тексту 2: {ling_features[7]:.2f}")
            st.write(f"- Різниця тональностей: **{ling_features[8]:.2f}**")
            st.write(f"Модель використовує не тільки сам конфлікт тональностей тому ці показники іноді можуть не сходитися з результатом")
