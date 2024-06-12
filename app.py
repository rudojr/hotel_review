import streamlit as st
import pickle
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    text = text.lower()
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    text = [word for word in text if not any(c.isdigit() for c in word)]
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    text = [t for t in text if len(t) > 0]
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    text = [t for t in text if len(t) > 1]
    text = " ".join(text)
    return text

tfidf = pickle.load(open('tfidf_vectorizer.pkl','rb'))
rf_model = pickle.load(open('random_forest_model.pkl','rb'))
features = pickle.load(open('features.pkl', 'rb'))

st.title("Hotel Review Sentiment Classifier")

input_review = st.text_area("Enter the hotel review")

if st.button('Predict'):
    # 1. preprocess
    cleaned_review = clean_text(input_review)
    
    # 2. vectorize
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(cleaned_review.split(" "))]

    # train a Doc2Vec model with our text data
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    doc2vec_vectors = []
    for text in cleaned_review:
        vector = model.infer_vector(text.split(" "))
        doc2vec_vectors.append(vector)
    
    doc2vec_df = pd.DataFrame(doc2vec_vectors, columns=[f"doc2vec_vector_{i}" for i in range(len(vector))])
    st.write(doc2vec_df)
    # # Vector hóa bằng TF-IDF
    # tfidf_vectorizer = TfidfVectorizer(min_df=10, max_df=1.0)

    # # Sử dụng fit_transform để chuyển đổi chuỗi thành vector sử dụng TF-IDF
    # tfidf_matrix = tfidf_vectorizer.fit_transform([cleaned_review])

    # # Chuyển đổi ma trận TF-IDF thành mảng NumPy
    # tfidf_array = tfidf_matrix.toarray()
    # columns = tfidf_vectorizer.get_feature_names_out()

    # # Tạo DataFrame từ tfidf_array
    # tfidf_df = pd.DataFrame(tfidf_array, columns=columns)
    # st.write(tfidf_df)
    # Kết hợp các đặc trưng
    vector_input_df = doc2vec_df
    
    # Đảm bảo rằng tất cả các cột đặc trưng có mặt
    for feature in features:
        if feature not in vector_input_df.columns:
            vector_input_df[feature] = 0

    vector_input_df = vector_input_df[features]
    
    # Debug: In ra dữ liệu đầu vào để kiểm tra
    st.write("Vectorized input:")
    st.write(vector_input_df)

    # 3. predict
    result = rf_model.predict(vector_input_df)[0]
    
    # 4. Display
    if result == 1:
        st.header("Negative Review")
    else:
        st.header("Positive Review")