import streamlit as st
import pandas as pd
import re
import easyocr
import joblib
from textblob import TextBlob

st.set_page_config(page_title="Предиктор Сарказму", layout="centered")


@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['en'], gpu=False)


@st.cache_resource
def load_ml_pipeline():
    return joblib.load('sarcasm_model_pipeline.pkl')


def extract_linguistic_features(caption, ocr):
    caption = str(caption)
    ocr = str(ocr)

    sentiment_caption = TextBlob(caption).sentiment.polarity
    sentiment_ocr = TextBlob(ocr).sentiment.polarity

    return {
        'len_caption': len(caption),
        'len_ocr': len(ocr),
        'caps_caption': len(re.findall('[A-Z]', caption)),
        'caps_ocr': len(re.findall('[A-Z]', ocr)),
        'punct_caption': len(re.findall(r'[!?]', caption)),
        'punct_ocr': len(re.findall(r'[!?]', ocr)),
        'sentiment_caption': sentiment_caption,
        'sentiment_ocr': sentiment_ocr,
        'sentiment_diff': abs(sentiment_caption - sentiment_ocr),
        'markers_caption': len(re.findall(r'\b(wow|sure|great|thanks|yeah|right|literally|totally)\b', caption.lower())),
        'markers_ocr': len(re.findall(r'\b(wow|sure|great|thanks|yeah|right|literally|totally)\b', ocr.lower()))
    }


pipeline = load_ml_pipeline()
ocr_reader = load_ocr_model()

if "text_1" not in st.session_state:
    st.session_state["text_1"] = ""
if "text_2" not in st.session_state:
    st.session_state["text_2"] = ""
if "last_processed_file" not in st.session_state:
    st.session_state["last_processed_file"] = None

st.title("Предиктор сарказму")

uploaded_file = st.file_uploader("Завантажте картинку для автоматичного розпізнавання", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and uploaded_file.name != st.session_state["last_processed_file"]:
    with st.spinner("Розпізнавання тексту..."):
        image_bytes = uploaded_file.read()
        results = ocr_reader.readtext(image_bytes)

        top_texts, bottom_texts = [], []
        if results:
            max_y = max([bbox[2][1] for bbox, text, prob in results])
            split_y = max_y * 0.4

            for bbox, text, prob in results:
                center_y = (bbox[0][1] + bbox[2][1]) / 2
                if center_y <= split_y:
                    top_texts.append(text)
                else:
                    bottom_texts.append(text)

        st.session_state["text_1"] = " ".join(top_texts)
        st.session_state["text_2"] = " ".join(bottom_texts)
        st.session_state["last_processed_file"] = uploaded_file.name
        st.rerun()

caption_input = st.text_area("Текст 1 (Підпис/Верх):", key="text_1")
ocr_input = st.text_area("Текст 2 (Мем/Низ):", key="text_2")

if st.button("Predict", type="primary"):
    if not caption_input and not ocr_input:
        st.warning("Текст потрібен хоча б в одному з полів")
    else:
        features_dict = extract_linguistic_features(caption_input, ocr_input)
        features_dict['combined_text'] = f"{caption_input} {ocr_input}"

        X_input = pd.DataFrame([features_dict])

        prediction = pipeline.predict(X_input)[0]
        probabilities = pipeline.predict_proba(X_input)[0]
        confidence = probabilities[1] * 100 if prediction == 1 else probabilities[0] * 100

        st.divider()
        if prediction == 1:
            st.error("**Результат: Це сарказм**")
        else:
            st.success("**Результат: Це НЕ сарказм**")

        st.metric(label="Упевненість моделі", value=f"{confidence:.1f}%")
        st.progress(int(confidence))

        with st.expander("Переглянути параметри цього тексту"):
            st.write(f"- Тональність тексту 1: {features_dict['sentiment_caption']:.2f}")
            st.write(f"- Тональність тексту 2: {features_dict['sentiment_ocr']:.2f}")
            st.write(f"- Різниця тональностей: **{features_dict['sentiment_diff']:.2f}**")
            st.write(
                "Модель використовує не тільки сам конфлікт тональностей, тому ці показники іноді можуть не сходитися з результатом")
