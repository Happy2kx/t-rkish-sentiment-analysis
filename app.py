import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import clean_text, remove_stopwords

# Sayfa yapılandırmasını ayarla
st.set_page_config(
    page_title="Türkçe Duygu Analizi",
    page_icon="📊",
    layout="wide"
)

# Sabitler
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR_RAW = os.path.join(BASE_DIR, 'models_raw')
MODEL_DIR_BALANCED = os.path.join(BASE_DIR, 'models_balanced')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

@st.cache_resource
def load_models_from_dir(model_dir):
    """Bir dizinden tüm eğitilmiş modelleri ve vectörizörleri yükle"""
    models = {}
    vectorizers = {}
    
    if not os.path.exists(model_dir):
        return {}, {}
        
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl') and not filename.endswith('_vectorizer.pkl'):
            model_name = filename.replace('.pkl', '')
            model_path = os.path.join(model_dir, filename)
            vec_path = os.path.join(model_dir, f"{model_name}_vectorizer.pkl")
            
            if os.path.exists(vec_path):
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                with open(vec_path, 'rb') as f:
                    vectorizers[model_name] = pickle.load(f)
                    
    return models, vectorizers


def get_confidence_score(model, vectorized_input, model_name):
    """Eğer varsa tahmin için güven skorunu al"""
    try:
        # Destekleyen modeller için predict_proba'yı dene (Naive Bayes, Logistic Regression, Ensemble)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vectorized_input)[0]
            confidence = np.max(proba) * 100
            return confidence
        # decision_function olan SVM için
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(vectorized_input)[0]
            
            if isinstance(decision, np.ndarray) and decision.ndim > 0:
                # Çok sınıflı: Karar skorlarını softmax kullanarak pseudo-olasılıklara çevir
                # Sayısal stabilite için max(decision)'u çıkarıyoruz
                exp_decision = np.exp(decision - np.max(decision))
                proba = exp_decision / np.sum(exp_decision)
                confidence = np.max(proba) * 100
            else:
                # İkili: Skaler kararı (mesafe) sigmoid kullanarak olasılığa çevir
                confidence = (1 / (1 + np.exp(-np.abs(decision)))) * 100
            return confidence
        else:
            return None
    except Exception:
        return None


def analyze_single_text(text, model, vectorizer, model_name):
    """Tek bir metni analiz et ve güvenle tahmin döndür"""
    cleaned = clean_text(text)
    processed = remove_stopwords(cleaned)
    vectorized = vectorizer.transform([processed])
    
    prediction = model.predict(vectorized)[0]
    confidence = get_confidence_score(model, vectorized, model_name)
    
    return prediction, confidence


def render_header(title, dataset_type=None, show_badge=True):
    """Veri seti göstergesi ile tutarlı modern bir başlık oluşturur"""
    if show_badge:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<h1 style='margin-bottom: 0; padding-top: 0;'>{title}</h1>", unsafe_allow_html=True)
            st.caption("🚀 TR Metin Tabanlı Duygu Analizi Projesi")
        
        with col2:
            if dataset_type == "balanced":
                st.markdown("""
                <div style='background: rgba(46, 204, 113, 0.1); border: 1px solid #2ecc71;
                            padding: 8px 16px; border-radius: 12px; text-align: center;
                            color: #2ecc71; font-weight: 600; font-size: 0.9em; margin-top: 10px;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    ⚖️ Dengelenmiş Veri
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: rgba(231, 76, 60, 0.1); border: 1px solid #e74c3c;
                            padding: 8px 16px; border-radius: 12px; text-align: center;
                            color: #e74c3c; font-weight: 600; font-size: 0.9em; margin-top: 10px;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    📊 Ham Veri
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='margin-bottom: 0; padding-top: 0;'>{title}</h1>", unsafe_allow_html=True)
        st.caption("🚀 TR Metin Tabanlı Duygu Analizi Projesi")
        
    st.markdown("---")

def main():
    # Özel başlık düzeni için st.title kaldırıldı
    
    # Navigasyon için kenar çubuğu
    st.sidebar.title("Navigasyon")
    
    # Kenar çubuğunun üstünde veri seti seçimi
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Model Seçimi")
    
    dataset_choice = st.sidebar.radio(
        "Eğitim Verisi:",
        ["⚖️ Dengelenmiş", "📊 Ham Veri"],
        help="Dengelenmiş: Tüm sınıflar eşit. Ham: Orijinal dağılım."
    )
    
    # Hangi model dizininin kullanılacağını belirle
    if dataset_choice == "⚖️ Dengelenmiş":
        MODEL_DIR = MODEL_DIR_BALANCED
        dataset_type = "balanced"
        st.sidebar.success("Dengeli veri modelleri aktif")
    else:
        MODEL_DIR = MODEL_DIR_RAW
        dataset_type = "raw"
        st.sidebar.warning("Ham veri modelleri aktif")
    
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Sayfa Seçiniz", ["Tahmin", "Toplu Analiz", "Model Karşılaştırma", "Veri İstatistikleri"])
    
    # Modelleri sadece mevcut sayfa için gerekli olduğunda yükle
    models = {}
    vectorizers = {}
    
    if page in ["Tahmin", "Toplu Analiz"]:
        with st.spinner("Modeller yükleniyor..."):
            models, vectorizers = load_models_from_dir(MODEL_DIR)
        
        if not models:
            st.error(f"Modeller bulunamadı! ({MODEL_DIR})\n\nLütfen önce modelleri eğitin: `python run_all.py`")
            return

    if page == "Tahmin":
        # Mevcut model setiyle ilgili dinamik başlık
        render_header("🔮 Gerçek Zamanlı Duygu Analizi", dataset_type, show_badge=True)
        
        # Mevcut model setiyle ilgili bilgi kutusu
        if dataset_type == "balanced":
            st.info("🎯 **Dengelenmiş veri modeli kullanılıyor.** Tüm sınıflar eşit temsil ediliyor, özellikle azınlık sınıflarını daha iyi tanır.")
        else:
            st.warning("📊 **Ham veri modeli kullanılıyor.** Orijinal veri dağılımı ile eğitildi, yüksek accuracy ama dengesiz sınıf dağılımı.")
        
        # Modern kart benzeri düzen
        st.markdown("""
        <style>
        .analysis-card {
            background: linear-gradient(145deg, #1e1e2e, #2d2d3d);
            border-radius: 15px;
            padding: 25px;
            margin: 10px 0;
            border: 1px solid #3d3d4d;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Giriş için iki sütunlu düzen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Metin Girişi")
            user_input = st.text_area(
                "Analiz edilecek metni giriniz:",
                height=180,
                placeholder="Örnek: Bu ürün gerçekten harika, çok memnun kaldım!"
            )
        
        with col2:
            st.markdown("### ⚙️ Model Ayarları")
            selected_model = st.selectbox(
                "Algoritma:",
                list(models.keys()),
                help="Tahmin için kullanılacak makine öğrenmesi algoritması"
            )
            
            # Model bilgisi
            model_info = {
                "Logistic_Regression": "Hızlı ve güvenilir",
                "SVM": "Yüksek doğruluk",
                "Naive_Bayes": "En hızlı",
                "Random_Forest": "Ensemble yöntem",
                "Voting_Ensemble": "🏆 4 model birleşimi - En yüksek doğruluk"
            }
            st.caption(f"💡 {model_info.get(selected_model, '')}")
        
        # Analiz düğmesi - ortalanmış ve stilize edilmiş
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button("🚀 Analiz Et", use_container_width=True, type="primary")
        
        if analyze_btn:
            if user_input:
                model = models[selected_model]
                vec = vectorizers[selected_model]
                
                with st.spinner("Analiz ediliyor..."):
                    prediction, confidence = analyze_single_text(user_input, model, vec, selected_model)
                
                # Sonuçları modern kartlarla göster
                st.markdown("---")
                st.markdown("### 📊 Analiz Sonucu")
                
                sentiment = prediction
                
                # Sonuç kartları (Güven Skoru kaldırıldığı için 2 sütuna değişti)
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    if sentiment == 1 or sentiment == 'Positive' or sentiment == 'pozitif':
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #2ecc71, #27ae60); 
                                    padding: 30px; border-radius: 15px; text-align: center;'>
                            <h1 style='color: white; margin: 0;'>😊</h1>
                            <h3 style='color: white; margin: 10px 0 0 0;'>POZİTİF</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    elif sentiment == 0 or sentiment == 'Negative' or sentiment == 'negatif':
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #e74c3c, #c0392b); 
                                    padding: 30px; border-radius: 15px; text-align: center;'>
                            <h1 style='color: white; margin: 0;'>😠</h1>
                            <h3 style='color: white; margin: 10px 0 0 0;'>NEGATİF</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    elif sentiment == 'Notr' or sentiment == 2:
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #f39c12, #e67e22); 
                                    padding: 30px; border-radius: 15px; text-align: center;'>
                            <h1 style='color: white; margin: 0;'>😐</h1>
                            <h3 style='color: white; margin: 10px 0 0 0;'>NÖTR</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"Duygu: {sentiment}")
                
                with result_col2:
                    st.markdown(f"""
                    <div style='background: linear-gradient(145deg, #1e1e2e, #2d2d3d); 
                                padding: 20px; border-radius: 15px; text-align: center;
                                border: 1px solid #3d3d4d; height: 100%; display: flex; flex-direction: column; justify-content: center;'>
                        <h4 style='color: #888; margin: 0;'>Kullanılan Model</h4>
                        <h3 style='color: #3498db; margin: 10px 0;'>{selected_model.replace('_', ' ')}</h3>
                        <p style='color: #666; margin: 0; font-size: 14px;'>
                            {'✅ Dengelenmiş Veri' if dataset_type == 'balanced' else '⚠️ Ham Veri'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            else:
                st.warning("⚠️ Lütfen analiz edilecek bir metin giriniz.")
    
    elif page == "Toplu Analiz":
        render_header("📁 Toplu Metin Analizi", show_badge=False)
        st.write("CSV dosyası yükleyerek birden fazla metni aynı anda analiz edin.")
        
        # Model seçimi
        selected_model = st.selectbox("Model Seçiniz", list(models.keys()), key="batch_model")
        
        # Dosya yükleme
        uploaded_file = st.file_uploader("CSV dosyası yükleyin", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write(f"**Yüklenen dosya:** {len(df_upload)} satır")
                
                # Sütunları göster
                st.write("**Mevcut sütunlar:**", list(df_upload.columns))
                
                # Metin sütununu seç
                text_column = st.selectbox("Metin sütununu seçin", df_upload.columns)
                
                if st.button("Analizi Başlat"):
                    model = models[selected_model]
                    vec = vectorizers[selected_model]
                    
                    predictions = []
                    confidences = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, text in enumerate(df_upload[text_column]):
                        if pd.notna(text):
                            pred, conf = analyze_single_text(str(text), model, vec, selected_model)
                            predictions.append(pred)
                            confidences.append(conf if conf else 0)
                        else:
                            predictions.append(None)
                            confidences.append(0)
                        
                        # İlerlemeyi güncelle
                        progress = (idx + 1) / len(df_upload)
                        progress_bar.progress(progress)
                        status_text.text(f"Analiz ediliyor: {idx + 1}/{len(df_upload)}")
                    
                    status_text.text("Analiz tamamlandı!")
                    
                    # Sonucu dataframe'e ekle
                    df_upload['Tahmin'] = predictions
                    df_upload['Güven_Skoru'] = confidences
                    
                    # Tahminleri okunabilir etiketlere eşle
                    def map_sentiment(pred):
                        if pred == 1 or pred == 'Positive' or pred == 'pozitif':
                            return 'POZİTİF'
                        elif pred == 0 or pred == 'Negative' or pred == 'negatif':
                            return 'NEGATİF'
                        return str(pred) if pred else 'Bilinmiyor'
                    
                    df_upload['Duygu'] = df_upload['Tahmin'].apply(map_sentiment)
                    
                    # Sonuçları göster
                    st.subheader("Sonuçlar")
                    st.dataframe(df_upload)
                    
                    # Özet istatistikler
                    st.subheader("Özet İstatistikler")
                    col1, col2, col3 = st.columns(3)
                    
                    sentiment_counts = df_upload['Duygu'].value_counts()
                    
                    with col1:
                        pozitif_count = sentiment_counts.get('POZİTİF', 0)
                        st.metric("Pozitif", pozitif_count, f"%{100*pozitif_count/len(df_upload):.1f}")
                    
                    with col2:
                        negatif_count = sentiment_counts.get('NEGATİF', 0)
                        st.metric("Negatif", negatif_count, f"%{100*negatif_count/len(df_upload):.1f}")
                    
                    with col3:
                        avg_conf = df_upload['Güven_Skoru'].mean()
                        st.metric("Ort. Güven", f"%{avg_conf:.1f}")
                    
                    # İndirme düğmesi
                    csv = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Sonuçları CSV olarak indir",
                        data=csv,
                        file_name="analiz_sonuclari.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Dosya okuma hatası: {e}")
                
    elif page == "Model Karşılaştırma":
        render_header("🤖 Algoritma Performansı", dataset_type, show_badge=True)
        
        # results.csv varsa, yükle
        results_path = os.path.join(MODEL_DIR, 'comparison_results.csv')
        if os.path.exists(results_path):
            # Özel Tablo Stil
            st.markdown("""
            <style>
            .premium-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-family: 'Inter', sans-serif;
                color: #e0e0e0;
                background: #1e1e2e;
                border-radius: 12px;
                overflow: hidden;
            }
            .premium-table thead tr {
                background: linear-gradient(90deg, #3498db, #8e44ad);
                color: #ffffff;
                text-align: left;
                font-weight: bold;
            }
            .premium-table th, .premium-table td {
                padding: 12px 15px;
                border-bottom: 1px solid #3d3d4d;
            }
            .premium-table tbody tr:hover {
                background-color: #2d2d3d;
                transition: 0.3s;
            }
            .premium-table tr:last-of-type {
                border-bottom: 2px solid #3498db;
            }
            .best-tag {
                background: linear-gradient(135deg, #f1c40f, #f39c12);
                color: #000;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                font-weight: bold;
                margin-left: 5px;
            }
            .result-badge {
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: 500;
            }
            </style>
            """, unsafe_allow_html=True)

            # Görüntü için verileri işle
            df_results = pd.read_csv(results_path)
            df_display = df_results.copy()
            
            # Sütun adlarını Türkçe'ye eşle
            column_mapping = {
                'Model': 'Algoritma',
                'Accuracy': 'Doğruluk (Acc)',
                'F1 Score': 'F1 Skoru',
                'Precision': 'Hassasiyet',
                'Recall': 'Duyarlılık',
                'Prediction Time (ms/sample)': 'Hız (ms/örnek)',
                'Training Time (s)': 'Eğitim (sn)'
            }
            
            # Seç ve sütunları yeniden sırala
            cols_to_keep = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'Prediction Time (ms/sample)', 'Training Time (s)']
            df_display = df_display[cols_to_keep]
            
            # Vurgulamak için en iyi modeli bul
            best_acc_idx = df_display['Accuracy'].idxmax()
            best_f1_idx = df_display['F1 Score'].idxmax()
            
            # HTML tablo oluştur
            html = '<table class="premium-table"><thead><tr>'
            for col in cols_to_keep:
                html += f'<th>{column_mapping.get(col, col)}</th>'
            html += '</tr></thead><tbody>'
            
            for idx, row in df_display.iterrows():
                html += '<tr>'
                for col in cols_to_keep:
                    val = row[col]
                    formatted_val = val
                    
                    # Değerleri biçimlendir
                    if col in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
                        formatted_val = f'%{val*100:.2f}'
                    elif col == 'Prediction Time (ms/sample)':
                        formatted_val = f'{val:.4f}'
                    elif col == 'Training Time (s)':
                        formatted_val = f'{val:.2f}s'
                    elif col == 'Model':
                        formatted_val = val.replace('_', ' ')
                        if idx == best_acc_idx:
                            formatted_val += ' <span class="best-tag">🏆 EN İYİ ACC</span>'
                        elif idx == best_f1_idx:
                            formatted_val += ' <span class="best-tag">⭐ EN İYİ F1</span>'
                    
                    html += f'<td>{formatted_val}</td>'
                html += '</tr>'
            
            html += '</tbody></table>'
            st.markdown(html, unsafe_allow_html=True)
            
            # İnteraktif grafikler
            st.subheader("📈 Metrik Karşılaştırması")
            
            # Farklı metrikler için sekmeler oluştur
            tab1, tab2, tab3 = st.tabs(["Doğruluk Metrikleri", "Hız Karşılaştırması", "Confusion Matrix"])
            
            with tab1:
                metric = st.selectbox("Metrik Seçiniz", ['Accuracy', 'F1 Score', 'Precision', 'Recall'])
                
                # Koyu tema minimalist grafik
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                # Modern renkler
                colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
                bar_colors = colors[:len(df_results)]
                
                bars = ax.bar(df_results['Model'], df_results[metric], color=bar_colors, alpha=0.9, edgecolor='none')
                
                # Stilleme
                ax.tick_params(axis='x', colors='#e0e0e0', rotation=45)
                ax.tick_params(axis='y', colors='#e0e0e0')
                ax.spines['bottom'].set_color('#e0e0e0')
                ax.spines['left'].set_color('#e0e0e0')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', linestyle='--', alpha=0.1, color='#e0e0e0')
                
                ax.set_ylabel(metric, color='#e0e0e0', fontsize=10)
                ax.set_title(f'Model Karşılaştırması - {metric}', color='#ffffff', fontsize=12, pad=20)
                ax.set_ylim(0, 1)
                
                # Değer etiketlerini ekle
                for bar, val in zip(bars, df_results[metric]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{val:.2%}', ha='center', va='bottom', fontsize=9, color='#e0e0e0', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with tab2:
                st.write("**Tahmin Süresi (ms/örnek)**")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
                bar_colors = colors[:len(df_results)]
                
                bars = ax.bar(df_results['Model'], df_results['Prediction Time (ms/sample)'], color=bar_colors, alpha=0.9)
                
                # Stilleme
                ax.tick_params(axis='x', colors='#e0e0e0', rotation=45)
                ax.tick_params(axis='y', colors='#e0e0e0')
                ax.spines['bottom'].set_color('#e0e0e0')
                ax.spines['left'].set_color('#e0e0e0')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', linestyle='--', alpha=0.1, color='#e0e0e0')
                
                ax.set_ylabel('Tahmin Süresi (ms)', color='#e0e0e0')
                ax.set_title('Model Hız Karşılaştırması', color='#ffffff', pad=20)
                
                for bar, val in zip(bars, df_results['Prediction Time (ms/sample)']):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{val:.4f}', ha='center', va='bottom', fontsize=9, color='#e0e0e0')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Eğitim Süresi sütununu kontrol et
                if 'Training Time (s)' in df_results.columns:
                    st.write("**Eğitim Süresi (saniye)**")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bar_colors = colors[:len(df_results)]
                    bars = ax.bar(df_results['Model'], df_results['Training Time (s)'], color=bar_colors)
                    ax.set_ylabel('Eğitim Süresi (s)')
                    ax.set_title('Model Eğitim Süresi Karşılaştırması')
                    
                    for bar, val in zip(bars, df_results['Training Time (s)']):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                               f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
            with tab3:
                st.write("**Confusion Matrix Görselleri**")
                
                # Confusion matrix görselleri ara
                cm_files = [f for f in os.listdir(MODEL_DIR) if 'confusion_matrix' in f and f.endswith('.png')]
                
                if cm_files:
                    # Confusion matrix'leri bir grid'de göster
                    cols = st.columns(2)
                    for idx, cm_file in enumerate(cm_files):
                        with cols[idx % 2]:
                            model_name = cm_file.replace('_confusion_matrix.png', '')
                            st.write(f"**{model_name}**")
                            cm_path = os.path.join(MODEL_DIR, cm_file)
                            st.image(cm_path)
                else:
                    st.info("Confusion matrix görselleri bulunamadı. Modelleri yeniden değerlendirmek için run_all.py çalıştırın.")
            
        else:
            st.info("Karşılaştırma sonuçları bulunamadı.")

    elif page == "Veri İstatistikleri":
        render_header("📊 Veri Seti İstatistikleri", show_badge=False)
        
        # Hem ham hem de işlenmiş veriyi yükle
        raw_data_path = os.path.join(DATA_DIR, '..', 'raw', 'turkish_sentiment_data.csv')
        processed_data_path = os.path.join(DATA_DIR, 'processed_data.csv')
        
        # Semantik renkleri tanımla
        label_color_map = {
            'Positive': '#2ecc71',
            'pozitif': '#2ecc71',
            'Negative': '#e74c3c',
            'negatif': '#e74c3c',
            'Notr': '#f39c12',
            'notr': '#f39c12',
            'Neutral': '#f39c12',
            'neutral': '#f39c12',
            1: '#2ecc71',
            0: '#e74c3c',
            2: '#f39c12'
        }
        
        # Ham ve işlenmiş veri için sekmeler oluştur
        tab1, tab2, tab3 = st.tabs(["📁 Ham Veri", "⚖️ Dengelenmiş Veri", "🔍 Karşılaştırma"])
        
        with tab1:
            if os.path.exists(raw_data_path):
                df_raw = pd.read_csv(raw_data_path)
                
                st.markdown("### 📈 Ham Veri Özeti")
                st.info("Bu, orijinal veri setidir. Sınıf dağılımı dengesizdir.")
                
                # Stilize edilmiş sütunlarda metrikler
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📝 Toplam Yorum", f"{len(df_raw):,}")
                
                with col2:
                    if 'label' in df_raw.columns:
                        st.metric("🏷️ Etiket Sayısı", df_raw['label'].nunique())
                
                with col3:
                    if 'text' in df_raw.columns:
                        avg_len = df_raw['text'].str.len().mean()
                        st.metric("📏 Ort. Uzunluk", f"{avg_len:.0f} karakter")
                
                with col4:
                    if 'label' in df_raw.columns:
                        majority = df_raw['label'].value_counts().index[0]
                        st.metric("👑 Çoğunluk Sınıf", majority)
                
                if 'label' in df_raw.columns:
                    st.markdown("### 📊 Sınıf Dağılımı (Ham Veri)")
                    
                    label_counts = df_raw['label'].value_counts()
                    chart_colors = [label_color_map.get(label, '#3498db') for label in label_counts.index]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        fig.patch.set_facecolor('none')
                        ax.set_facecolor('none')
                        
                        bars = ax.bar(label_counts.index.astype(str), label_counts.values, color=chart_colors, width=0.6)
                        
                        ax.set_title('Ham Veri Dağılımı', fontsize=14, fontweight='bold', color='white', pad=20)
                        ax.tick_params(colors='#e0e0e0')
                        ax.spines['bottom'].set_color('#e0e0e0')
                        ax.spines['left'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.yaxis.set_visible(False)  # Minimal look
                        
                        # Add value labels
                        for bar, val in zip(bars, label_counts.values):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                                   f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e0e0e0')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        fig.patch.set_facecolor('none')
                        
                        wedges, texts, autotexts = ax.pie(
                            label_counts.values, 
                            labels=label_counts.index, 
                            autopct='%1.1f%%', 
                            colors=chart_colors,
                            explode=[0.02] * len(label_counts),
                            shadow=False, # Removed shadow for flat design
                            startangle=90,
                            textprops={'color': '#e0e0e0'}
                        )
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('bold')
                            
                        ax.set_title('Dağılım Oranları', fontsize=14, fontweight='bold', color='white')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                
                st.markdown("### 📋 Örnek Veriler")
                st.dataframe(df_raw.head(10), use_container_width=True)
            else:
                st.warning("Ham veri dosyası bulunamadı.")
        
        with tab2:
            if os.path.exists(processed_data_path):
                df_processed = pd.read_csv(processed_data_path)
                
                st.markdown("### ⚖️ Dengelenmiş Veri Özeti")
                st.success("Veri dengeli hale getirildi. Tüm sınıflar eşit sayıda örnek içeriyor.")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📝 Toplam Yorum", f"{len(df_processed):,}")
                
                with col2:
                    if 'label' in df_processed.columns:
                        st.metric("🏷️ Etiket Sayısı", df_processed['label'].nunique())
                
                with col3:
                    if 'label' in df_processed.columns:
                        per_class = len(df_processed) // df_processed['label'].nunique()
                        st.metric("📊 Sınıf Başına", f"{per_class:,}")
                
                with col4:
                    if 'processed_text' in df_processed.columns:
                        avg_len = df_processed['processed_text'].str.len().mean()
                        st.metric("📏 Ort. Uzunluk", f"{avg_len:.0f} karakter")
                
                if 'label' in df_processed.columns:
                    st.markdown("### 📊 Sınıf Dağılımı (Dengelenmiş)")
                    
                    label_counts = df_processed['label'].value_counts()
                    chart_colors = [label_color_map.get(label, '#3498db') for label in label_counts.index]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        fig.patch.set_facecolor('none')
                        ax.set_facecolor('none')
                        
                        bars = ax.bar(label_counts.index.astype(str), label_counts.values, color=chart_colors, width=0.6)
                        
                        ax.set_title('Dengelenmiş Veri Dağılımı', fontsize=14, fontweight='bold', color='white', pad=20)
                        ax.tick_params(colors='#e0e0e0')
                        ax.spines['bottom'].set_color('#e0e0e0')
                        ax.spines['left'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.yaxis.set_visible(False)
                        
                        for bar, val in zip(bars, label_counts.values):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                                   f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e0e0e0')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        fig.patch.set_facecolor('none')
                        
                        wedges, texts, autotexts = ax.pie(
                            label_counts.values, 
                            labels=label_counts.index, 
                            autopct='%1.1f%%', 
                            colors=chart_colors,
                            explode=[0.02] * len(label_counts),
                            shadow=False,
                            startangle=90,
                            textprops={'color': '#e0e0e0'}
                        )
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('bold')
                            
                        ax.set_title('Eşit Dağılım', fontsize=14, fontweight='bold', color='white')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                
                st.markdown("### 📋 Örnek İşlenmiş Veriler")
                display_cols = ['text', 'label', 'processed_text'] if 'processed_text' in df_processed.columns else df_processed.columns[:3]
                st.dataframe(df_processed[display_cols].head(10), use_container_width=True)
            else:
                st.warning("İşlenmiş veri dosyası bulunamadı. Modelleri eğitmek için `run_all.py` çalıştırın.")
        
        with tab3:
            st.markdown("### 📈 Ham vs Dengelenmiş Veri Karşılaştırması")
            
            if os.path.exists(raw_data_path) and os.path.exists(processed_data_path):
                df_raw = pd.read_csv(raw_data_path)
                df_processed = pd.read_csv(processed_data_path)
                
                # Karşılaştırma metrikleri
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📁 Ham Veri")
                    st.metric("Toplam Örnek", f"{len(df_raw):,}")
                    if 'label' in df_raw.columns:
                        for label in df_raw['label'].unique():
                            count = len(df_raw[df_raw['label'] == label])
                            pct = 100 * count / len(df_raw)
                            color = label_color_map.get(label, '#3498db')
                            st.markdown(f"<span style='color:{color}; font-weight:bold;'>● {label}:</span> {count:,} (%{pct:.1f})", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### ⚖️ Dengelenmiş Veri")
                    st.metric("Toplam Örnek", f"{len(df_processed):,}")
                    if 'label' in df_processed.columns:
                        for label in df_processed['label'].unique():
                            count = len(df_processed[df_processed['label'] == label])
                            pct = 100 * count / len(df_processed)
                            color = label_color_map.get(label, '#3498db')
                            st.markdown(f"<span style='color:{color}; font-weight:bold;'>● {label}:</span> {count:,} (%{pct:.1f})", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Görsel karşılaştırma
                st.markdown("#### 📊 Görsel Karşılaştırma")
                
                raw_counts = df_raw['label'].value_counts()
                proc_counts = df_processed['label'].value_counts()
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                fig.patch.set_facecolor('none')
                
                # Ham veri çubuğu
                colors1 = [label_color_map.get(label, '#3498db') for label in raw_counts.index]
                axes[0].set_facecolor('none')
                axes[0].bar(raw_counts.index.astype(str), raw_counts.values, color=colors1, alpha=0.8)
                axes[0].set_title('Ham Veri (Dengesiz)', fontsize=14, fontweight='bold', color='white')
                axes[0].tick_params(colors='#e0e0e0')
                axes[0].spines['bottom'].set_color='#e0e0e0'
                axes[0].spines['left'].set_visible(False)
                axes[0].spines['top'].set_visible(False)
                axes[0].spines['right'].set_visible(False)
                axes[0].yaxis.set_visible(False)
                
                # Değerleri ekle
                for i, v in enumerate(raw_counts.values):
                     axes[0].text(i, v + 50, str(v), color='white', ha='center', fontweight='bold')
                
                # İşlenmiş veri çubuğu
                colors2 = [label_color_map.get(label, '#3498db') for label in proc_counts.index]
                axes[1].set_facecolor('none')
                axes[1].bar(proc_counts.index.astype(str), proc_counts.values, color=colors2, alpha=0.8)
                axes[1].set_title('Dengelenmiş Veri (Eşit)', fontsize=14, fontweight='bold', color='white')
                axes[1].tick_params(colors='#e0e0e0')
                axes[1].spines['bottom'].set_color='#e0e0e0'
                axes[1].spines['left'].set_visible(False)
                axes[1].spines['top'].set_visible(False)
                axes[1].spines['right'].set_visible(False)
                axes[1].yaxis.set_visible(False)
                
                # Değerleri ekle
                for i, v in enumerate(proc_counts.values):
                     axes[1].text(i, v + 50, str(v), color='white', ha='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Bilgi kutusu
                st.info(f"""
                **Veri Dengeleme Özeti:**
                - Ham veri: {len(df_raw):,} örnek (dengesiz dağılım)
                - Dengelenmiş veri: {len(df_processed):,} örnek (eşit dağılım)
                - Azaltılan örnek: {len(df_raw) - len(df_processed):,} ({100*(len(df_raw) - len(df_processed))/len(df_raw):.1f}%)
                
                Dengeleme, modelin tüm sınıfları eşit şekilde öğrenmesini sağlar.
                """)
            else:
                st.warning("Karşılaştırma için her iki veri dosyası da gereklidir.")

if __name__ == "__main__":
    main()
