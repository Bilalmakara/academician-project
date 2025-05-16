import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import Counter
import json
import re

# Sayfa yapılandırması
st.set_page_config(
    page_title="Akademik Proje Eşleştirme Sistemi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS
st.markdown("""
<style>
    /* Ana tema renkleri */
    :root {
        --primary-color: #00C49A;
        --secondary-color: #00C49A;
        --accent-color: #00C49A;
        --background-color: #0E1117;
        --card-background: #262730;
        --text-color: #FAFAFA;
        --text-secondary: #FAFAFA;
        --border-color: #262730;
        --hover-color: #262730;
        --select-background: #0E1117;
        --select-text: #FAFAFA;
        --select-hover: #262730;
    }

    /* Genel stil */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    .stApp {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }

    /* Başlık stilleri */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-color);
        text-align: center;
        margin-bottom: 1rem;
    }

    .subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Kart stilleri */
    .card {
        background-color: var(--card-background);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        transition: transform 0.2s;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.3);
        background-color: var(--hover-color);
    }

    /* Metrik kartı */
    .metric-card {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: var(--text-color);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Sidebar stilleri */
    .css-1d391kg {
        background-color: var(--card-background);
        padding: 2rem 1rem;
        border-right: 1px solid var(--border-color);
    }

    .sidebar-title {
        color: var(--text-color);
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
    }

    /* Form elemanları */
    .stSelectbox, .stRadio > div {
        background-color: var(--card-background);
        border-radius: 8px;
        padding: 0.5rem;
    }

    .stRadio > div[role="radiogroup"] > label {
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        background-color: var(--card-background);
        margin: 0.5rem 0;
        transition: all 0.2s;
        color: var(--text-color);
    }

    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {
        background-color: var(--primary-color);
        color: var(--text-color);
    }

    /* Tablo stilleri */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        background-color: var(--card-background) !important;
    }

    .dataframe th {
        background-color: var(--primary-color) !important;
        color: var(--text-color) !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }

    .dataframe td {
        padding: 0.75rem !important;
        border-bottom: 1px solid var(--border-color) !important;
        color: var(--text-color) !important;
        background-color: var(--card-background) !important;
    }

    /* Grafik stilleri */
    .plot-container {
        background-color: var(--card-background);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }

    /* Buton stilleri */
    .stButton > button {
        background-color: var(--primary-color);
        color: var(--text-color);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background-color: var(--secondary-color);
        transform: translateY(-1px);
    }

    /* Tooltip stilleri */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: var(--primary-color);
        color: var(--text-color);
        text-align: center;
        padding: 0.5rem;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Metrik kartları */
    .stMetric {
        background-color: var(--card-background);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }

    .stMetric > div {
        color: var(--text-color) !important;
    }

    /* Slider stilleri */
    .stSlider > div > div > div {
        background-color: var(--primary-color) !important;
    }

    .stSlider > div > div > div > div {
        background-color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# Başlık ve açıklama
st.markdown("""
<div class='main-title'>🎓 Akademik Proje Eşleştirme Sistemi</div>
<div class='subtitle'>
    Projeleriniz için en uygun akademisyenleri bulun ve en iyi eşleşmeleri görüntüleyin.
    <br>
    <small>Son güncelleme: """ + datetime.now().strftime("%d.%m.%Y %H:%M") + """</small>
</div>
""", unsafe_allow_html=True)

# CSV dosyasını yükle
@st.cache_data
def load_data():
    try:
        script_path = Path(__file__).resolve()
        csv_path = script_path.parent / "sbert_test_results.csv"
        
        if not csv_path.exists():
            st.error(f"Dosya bulunamadı: {csv_path}")
            return None
            
        df = pd.read_csv(str(csv_path))
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {str(e)}")
        return None

def normalize_name(name):
    if pd.isna(name):
        return name
    
    name = str(name).upper()
    
    # Türkçe karakterleri değiştir
    tr_chars = {
        'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U',
        'ç': 'C', 'ğ': 'G', 'ı': 'I', 'ö': 'O', 'ş': 'S', 'ü': 'U'
    }
    for tr_char, eng_char in tr_chars.items():
        name = name.replace(tr_char, eng_char)
    
    # Noktalama işaretlerini kaldır
    name = re.sub(r'[^\w\s]', '', name)
    
    # Fazla boşlukları temizle
    name = ' '.join(name.split())
    
    # CSV'deki "Soyad, Ad" formatını "Ad Soyad" formatına çevir
    if ',' in name:
        parts = name.split(',')
        if len(parts) == 2:
            surname = parts[0].strip()
            firstname = parts[1].strip()
            name = f"{firstname} {surname}"
    
    return name

def load_academician_info():
    json_path = Path(__file__).parent / "academicians.json"
    if not json_path.exists():
        return pd.DataFrame()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # JSON'dan DataFrame oluştur
    df = pd.DataFrame(data)
    # İsimleri normalize et
    df['normalized_name'] = df['Fullname'].apply(normalize_name)
    return df

df = load_data()
acad_df = load_academician_info()

if df is not None:
    # Akademisyen bilgilerini ana tabloya ekle
    if not acad_df.empty:
        # CSV'deki isimleri normalize et
        df['normalized_name'] = df['Akademisyen'].apply(normalize_name)
        
        # Sadece JSON'da bulunan akademisyenleri filtrele
        json_names = set(acad_df['normalized_name'])
        df = df[df['normalized_name'].isin(json_names)]
        
        # Eşleşen isimleri bul
        matched_names = set(df['normalized_name']).intersection(set(acad_df['normalized_name']))
        
        # Eşleşen isimler için JSON'dan bilgileri al
        for idx, row in df.iterrows():
            if row['normalized_name'] in matched_names:
                # JSON'dan eşleşen kaydı bul
                json_row = acad_df[acad_df['normalized_name'] == row['normalized_name']].iloc[0]
                # Bilgileri ekle
                df.at[idx, 'Email'] = json_row['Email']
                df.at[idx, 'Field'] = json_row['Field']
        
        # Geçici sütunu kaldır
        df = df.drop('normalized_name', axis=1)
        
        # NaN değerleri düzelt
        df['Email'] = df['Email'].fillna('E-posta bilgisi yok')
        df['Field'] = df['Field'].fillna('Uzmanlık alanı yok')
        
        # Eşleşme durumunu göster
        total_academics = len(json_names)  # JSON'daki toplam akademisyen sayısı
        matched_academics = len(matched_names)
        st.sidebar.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{matched_academics}/{total_academics}</div>
            <div class='metric-label'>Eşleşen Akademisyen</div>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class='sidebar-title'>🔍 Filtreler</div>
        """, unsafe_allow_html=True)
        
        # Proje seçim tipi
        selection_type = st.radio(
            "Proje Seçim Tipi:",
            ["Proje Adı", "Proje ID"],
            horizontal=True
        )
        
        if selection_type == "Proje Adı":
            unique_projects = df["EU Projesi"].dropna().unique()
            selected_project = st.selectbox("Proje Seçin:", unique_projects)
            filtered = df[df["EU Projesi"] == selected_project].copy()
        else:
            unique_ids = df["EU ID"].dropna().unique()
            selected_id = st.selectbox("Proje ID Seçin:", unique_ids)
            filtered = df[df["EU ID"] == selected_id].copy()

    # Filtreleme ve sıralama
    filtered = filtered.sort_values(by="Benzerlik Oranı", ascending=False)
    filtered = filtered.drop_duplicates(subset=["Akademisyen"], keep="first")
    filtered = filtered.sort_values(by="Benzerlik Oranı", ascending=False)

    # Ana içerik
    # Metrik kartları
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{len(filtered)}</div>
            <div class='metric-label'>Benzersiz Akademisyen</div>
        </div>
        """, unsafe_allow_html=True)

    # Grafikler ve en iyi eşleşmeler
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Benzerlik Analizi")
        
        # Detaylı benzerlik dağılımı grafiği
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=filtered["Benzerlik Oranı"],
            name="Dağılım",
            nbinsx=20,
            marker_color='#3B82F6',
            opacity=0.7
        ))
        
        # KDE (Kernel Density Estimation)
        kde = filtered["Benzerlik Oranı"].value_counts().sort_index()
        fig.add_trace(go.Scatter(
            x=kde.index,
            y=kde.values,
            name="Yoğunluk",
            line=dict(color='#60A5FA', width=2),
            fill='tozeroy',
            opacity=0.3
        ))
        
        # Ortalama çizgisi
        mean_value = filtered["Benzerlik Oranı"].mean()
        fig.add_vline(
            x=mean_value,
            line_dash="dash",
            line_color="#F1F5F9",
            annotation_text=f"Ortalama: {mean_value:.2f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title="Benzerlik Skorları Dağılımı ve Yoğunluk Analizi",
            xaxis_title="Benzerlik Skoru",
            yaxis_title="Frekans",
            plot_bgcolor='#334155',
            paper_bgcolor='#334155',
            font=dict(color='#F1F5F9'),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0)"
            ),
            margin=dict(t=50, l=50, r=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🏆 En İyi Eşleşmeler")
        for i, row in filtered.head(3).iterrows():
            # NaN kontrolü ekle
            email = row.get('Email')
            field = row.get('Field')
            email_info = f"📧 {email if pd.notna(email) else 'E-posta bilgisi yok'}"
            field_info = f"🎓 {field if pd.notna(field) else 'Uzmanlık alanı yok'}"
            
            # Eğer JSON'da eşleşme varsa, kartı vurgula
            card_style = "background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));" if pd.notna(email) else ""
            
            st.markdown(f"""
            <div class='card' style='{card_style}'>
                <h4 style='color: var(--text-color); margin-bottom: 0.5rem;'>👤 {row['Akademisyen']}</h4>
                <p style='color: #F1F5F9; font-weight: 600; background-color: #1E3A8A; padding: 5px; border-radius: 5px;'>Benzerlik: {row['Benzerlik Oranı']:.2f}</p>
                <p style='color: var(--text-secondary);'>Tip: {row['Eşleşme Tipi']}</p>
                <p style='color: var(--text-secondary);'>{email_info}</p>
                <p style='color: var(--text-secondary);'>{field_info}</p>
            </div>
            """, unsafe_allow_html=True)

    # Tüm eşleşmeleri tablo olarak göster
    st.markdown("### 📋 Tüm Eşleşmeler")
    possible_columns = ["Akademisyen", "Benzerlik Oranı", "Eşleşme Tipi", "E-posta", "Uzmanlık Alanı"]
    if selection_type == "Proje ID":
        possible_columns.insert(0, "EU Projesi")
    display_columns = [col for col in possible_columns if col in filtered.columns]
    
    # Tablo stilini özelleştir
    styled_df = filtered[display_columns].style.background_gradient(
        subset=[col for col in ["Benzerlik Oranı"] if col in display_columns],
        cmap="RdYlGn"
    ).set_properties(**{
        'background-color': '#334155',
        'color': '#F1F5F9',
        'border-color': '#475569',
        'padding': '0.75rem'
    }).set_table_styles([
        {'selector': 'th',
         'props': [('background-color', '#1E3A8A'),
                  ('color', '#F1F5F9'),
                  ('font-weight', 'bold'),
                  ('padding', '0.75rem'),
                  ('border', '1px solid #475569')]},
        {'selector': 'td',
         'props': [('background-color', '#334155'),
                  ('color', '#F1F5F9'),
                  ('padding', '0.75rem'),
                  ('border', '1px solid #475569')]},
        {'selector': 'tr:hover td',
         'props': [('background-color', '#475569')]}
    ])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )

    # İstatistikler
    st.markdown("### 📈 İstatistikler")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Ortalama Benzerlik",
            f"{filtered['Benzerlik Oranı'].mean():.2f}",
            f"{filtered['Benzerlik Oranı'].std():.2f}"
        )
    
    with col2:
        st.metric(
            "En Yüksek Benzerlik",
            f"{filtered['Benzerlik Oranı'].max():.2f}"
        )
    
    with col3:
        st.metric(
            "En Düşük Benzerlik",
            f"{filtered['Benzerlik Oranı'].min():.2f}"
        )
