# Import libraries yang diperlukan
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# --- FUNGSI-FUNGSI PEMROSESAN (Tidak diubah) ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(pattern=r'[^a-zA-Z\s]', repl='', string=text, flags=re.I|re.A)
    text = text.lower()
    text = text.strip()
    tokens = word_tokenize(text)
    list_stopwords = set(stopwords.words('indonesian'))
    list_stopwords.update(['yg', 'tdk', 'ga', 'gak', 'nya', 'sih', 'ya', 'aja', 'saja', 'juga', 'untuk'])
    filtered_tokens = [word for word in tokens if word not in list_stopwords and len(word) > 2]
    return " ".join(filtered_tokens)

def analyze_sentiment(text):
    positive_keywords = ['bagus', 'baik', 'puas', 'senang', 'keren', 'hebat', 'meningkat', 'apresiasi', 'terbantu', 'nyaman', 'suportif', 'kompak', 'bangga', 'relevan', 'efisien', 'inovasi']
    negative_keywords = ['buruk', 'kurang', 'kecewa', 'sulit', 'beban', 'berat', 'tidak', 'lambat', 'masalah', 'komplain', 'susah', 'hang', 'birokratis', 'sakit', 'terbatas']
    if not isinstance(text, str):
        return "Netral"
    text_lower = text.lower()
    score = 0
    for word in positive_keywords:
        if word in text_lower:
            score += 1
    for word in negative_keywords:
        if word in text_lower:
            score -= 1
    if score > 0:
        return 'Positif'
    elif score < 0:
        return 'Negatif'
    else:
        return 'Netral'

def perform_topic_modeling(corpus, n_topics=3):
    if not corpus or all(s.isspace() or not s for s in corpus):
        return None, None
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))
    try:
        X = vectorizer.fit_transform(corpus)
    except ValueError:
        return None, None
    if X.shape[1] == 0:
        return None, None
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out() 
    return lda, feature_names

def display_topics(model, feature_names, n_top_words=10):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics[f"Topik {topic_idx + 1}"] = ", ".join(top_words)
    return topics

def create_sample_data():
    data = {
        'Umpan Balik': [
            "Gaji dan tunjangan sudah cukup baik, tapi beban kerja terasa sangat berat belakangan ini.",
            "Saya sangat senang dengan lingkungan kerja yang kolaboratif dan manajer yang suportif.",
            "Fasilitas kantor seperti kursi dan meja kurang nyaman, membuat punggung sakit.",
            "Program training yang diberikan sangat membantu pengembangan karir saya. Keren!",
            "Proses approval cuti terkadang lambat dan tidak jelas prosedurnya.",
            "Apresiasi dari atasan membuat saya merasa dihargai dan termotivasi.",
            "Sistem IT internal sering mengalami masalah, sangat mengganggu pekerjaan.",
            "Saya netral saja, pekerjaan seperti biasa.",
            "Tolong perhatikan masalah parkir yang semakin susah.",
            "Beban pekerjaan tidak seimbang antar tim, ada yang terlalu santai, ada yang sangat berat.",
            "Komunikasi antar departemen harus ditingkatkan agar tidak terjadi miskoordinasi.",
            "Manajer saya sangat baik dalam memberikan arahan dan feedback yang membangun."
        ],
        'Departemen': [
            'Teknologi', 'Human Capital', 'Operasional', 'Human Capital', 'Operasional',
            'Marketing', 'Teknologi', 'Keuangan', 'Operasional', 'Teknologi', 'Marketing', 'Marketing'
        ]
    }
    return pd.DataFrame(data)


# --- ANTARMUKA STREAMLIT ---

st.set_page_config(layout="wide")
st.title("Dashboard Analisis Sentimen Feedback Karyawan")
st.markdown("Unggah file Excel atau CSV (pemisah koma/titik koma) berisi umpan balik karyawan untuk memulai analisis.")

# Sidebar (Tidak diubah)
with st.sidebar:
    st.title("Tentang Aplikasi")
    st.info(
        "Aplikasi dapat membantu sebuah perusahaan dalam menganalisis Feedback karyawan"
        " dengan mengklasifikasikannya menjadi sentimen Positif, Negatif, atau Netral menggunakan AI secara cepat dan tepat. \n\n"
        "**Fitur Utama:**\n"
        "- **Analisis Sentimen:** Mengklasifikasikan feedback ke dalam kategori Positif, Negatif, atau Netral.\n"
        "- **Topic Modeling:** Mengekstrak topik-topik utama yang paling sering dibicarakan oleh karyawan."
    )

# Bagian Kontrol Utama (Tidak diubah)
st.subheader("Mulai Analisis")
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "1. Unggah Data Anda di Sini",
        type=["xlsx", "csv"],
        help="Unggah file Excel atau CSV yang berisi kolom 'Umpan Balik' dan (opsional) 'Departemen'."
    )
with col2:
    use_sample_data = st.checkbox(
        "2. Atau, Gunakan Data Contoh",
        value=True if uploaded_file is None else False,
        help="Centang untuk menggunakan data internal jika Anda tidak punya file sendiri."
    )
st.divider()

# --- BLOK LOGIKA UTAMA (TELAH DIPERBAIKI) ---

# Inisialisasi session_state jika belum ada
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
    st.session_state.analyzed_df = None
    st.session_state.current_data_id = None

# TAHAP 1: MEMUAT DATA
# Bagian ini hanya mempersiapkan 'df_raw' dari sumber yang dipilih.
df_raw = None
current_data_id = None

if uploaded_file is not None:
    current_data_id = uploaded_file.name
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(uploaded_file)
        else:
            # PERBAIKAN BOM: Menggunakan encoding 'utf-8-sig' untuk menghapus karakter \ufeff
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8-sig')
        df_raw.columns = df_raw.columns.str.strip() # Membersihkan spasi pada nama kolom
    except Exception as e:
        st.error(f"Gagal membaca file. Pastikan formatnya valid. Error: {e}")
        st.stop()

elif use_sample_data:
    current_data_id = "sample_data"
    df_raw = create_sample_data()

# Reset state jika sumber data berubah
if st.session_state.current_data_id != current_data_id:
    st.session_state.analysis_run = False
    st.session_state.analyzed_df = None
    st.session_state.current_data_id = current_data_id

# TAHAP 2: MENJALANKAN ANALISIS (Hanya jika data berhasil dimuat)
if df_raw is not None:
    # Cek kolom sebelum menampilkan Preview
    if 'Umpan Balik' not in df_raw.columns:
        st.error(f"Kolom 'Umpan Balik' tidak ditemukan di file Anda.")
        st.warning(f"Kolom yang berhasil terdeteksi adalah: {list(df_raw.columns)}")
        st.stop()
    
    # Hanya tampilkan Preview jika analisis belum dijalankan
    if not st.session_state.analysis_run:
        st.subheader("Preview Data yang Dimuat")
        st.dataframe(df_raw.head())

    # Tombol analisis hanya bertugas memproses dan menyimpan ke state
    if st.button("Mulai Analisis Sekarang", type="primary"):
        with st.spinner('Sedang menganalisis data... Mohon tunggu.'):
            # Melakukan analisis pada salinan data mentah
            df_processed = df_raw.copy()
            df_processed['processed_feedback'] = df_processed['Umpan Balik'].apply(preprocess_text)
            df_processed['sentimen'] = df_processed['Umpan Balik'].apply(analyze_sentiment)
            # Menyimpan hasil ke session state
            st.session_state.analyzed_df = df_processed
            st.session_state.analysis_run = True
            st.rerun() # Refresh halaman untuk menampilkan hasil

# TAHAP 3: MENAMPILKAN HASIL (Jika analisis sudah dijalankan)
if st.session_state.analysis_run:
    st.divider()
    st.header("Hasil Analisis")

    analyzed_data = st.session_state.analyzed_df
    selected_dept = "Semua"
    
    # Filter ditempatkan di sini, sebelum tampilan hasil
    if 'Departemen' in analyzed_data.columns:
        st.subheader("Filter Tampilan Hasil")
        all_depts = analyzed_data['Departemen'].dropna().unique().tolist()
        depts = ['Semua'] + sorted(all_depts)
        
        # 'key' memastikan widget ini unik dan state-nya terjaga
        selected_dept = st.selectbox("Tampilkan data untuk departemen:", depts, key="filter_selectbox")
    
    # Menyiapkan data untuk ditampilkan berdasarkan filter
    if selected_dept != 'Semua' and 'Departemen' in analyzed_data.columns:
        display_df = analyzed_data[analyzed_data['Departemen'] == selected_dept].copy()
    else:
        display_df = analyzed_data.copy()

    # Tampilkan hasil dalam kolom
    res_col1, res_col2 = st.columns((1, 2))
    with res_col1:
        st.subheader(f"Persentase Distribusi Sentimen \n (Filter: {selected_dept})")
        sentiment_counts = display_df['sentimen'].value_counts()
        if not sentiment_counts.empty:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999', '#99ff99'])
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.warning("Tidak ada data untuk filter yang dipilih.")
            
    with res_col2:
        st.subheader(f"Word Cloud dari Topik Utama\n (Filter: {selected_dept})")
        if not display_df.empty:
            corpus = display_df['processed_feedback'].dropna().tolist()
            lda_model, feature_names = perform_topic_modeling(corpus, n_topics=3)
            if lda_model:
                topics = display_topics(lda_model, feature_names)
                for topic_name, words in topics.items():
                    st.markdown(f"**{topic_name}:** `{words}`")
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words.replace(",", " "))
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
            else:
                st.warning("Tidak cukup data unik untuk melakukan Topic Modeling pada filter ini.")
        else:
            st.warning("Tidak ada data untuk ditampilkan pada filter ini.")

    st.subheader(f"Detail Hasil Analisis\n (Filter: {selected_dept})")
    cols_to_show = ['Umpan Balik', 'sentimen']
    if 'Departemen' in display_df.columns:
        cols_to_show.append('Departemen')
    st.dataframe(display_df[cols_to_show])
