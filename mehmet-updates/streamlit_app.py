"""
Streamlit UI için: 
streamlit run mehmet-updates/streamlit_app.py
Api için: 
uvicorn api.app:app --reload

"""



import streamlit as st
import requests
import json
from typing import Optional
import time

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Türkçe Haber Chatbotu",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL'si
API_BASE_URL = "http://localhost:8000"

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        min-height: 150px;
    }
    .summary-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #000000 !important;
    }
    .summary-box strong {
        color: #1f77b4 !important;
    }
    .answer-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #32cd32;
        margin: 1rem 0;
        color: #000000 !important;
    }
    .answer-box strong {
        color: #32cd32 !important;
    }
    .stats-box {
        background-color: #fff8dc;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        color: #000000 !important;
    }
    /* Dark mode için özel stil */
    [data-theme="dark"] .summary-box {
        background-color: #1e3a5f !important;
        color: #ffffff !important;
        border-left: 4px solid #4da6ff;
    }
    [data-theme="dark"] .summary-box strong {
        color: #4da6ff !important;
    }
    [data-theme="dark"] .answer-box {
        background-color: #1e4a1e !important;
        color: #ffffff !important;
        border-left: 4px solid #66ff66;
    }
    [data-theme="dark"] .answer-box strong {
        color: #66ff66 !important;
    }
    /* Streamlit dark mode sınıfı */
    .stApp[data-theme="dark"] .summary-box {
        background-color: #2d4a63 !important;
        color: #ffffff !important;
    }
    .stApp[data-theme="dark"] .summary-box span {
        color: #ffffff !important;
    }
    .stApp[data-theme="dark"] .answer-box {
        background-color: #2d4a2d !important;
        color: #ffffff !important;
    }
    .stApp[data-theme="dark"] .answer-box span {
        color: #ffffff !important;
    }
    /* Force text color for all themes */
    .summary-box span, .answer-box span {
        color: #333333 !important;
    }
    /* Override for specific dark mode detection */
    @media (prefers-color-scheme: dark) {
        .summary-box {
            background-color: #2d4a63 !important;
        }
        .summary-box span {
            color: #ffffff !important;
        }
        .answer-box {
            background-color: #2d4a2d !important;
        }
        .answer-box span {
            color: #ffffff !important;
        }
    }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """API'nin çalışıp çalışmadığını kontrol eder"""
    try:
        # Query the models status endpoint which returns loading info
        response = requests.get(f"{API_BASE_URL}/models/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            # API considered healthy when at least one model is loaded
            return bool(data.get("total_loaded", 0) > 0)

        # Fallback: check root endpoint and treat HTTP 200 as partial success
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def summarize_text(text: str, max_length: int = 128, model: str = "multitask-lora-fast") -> Optional[dict]:
    """Metni özetler; hangi modelin kullanılacağını `model` ile API'ya bildirir.
    Daha iyi hata mesajları (503 detayları) döner.
    """
    try:
        payload = {"text": text, "max_length": max_length, "model": model}
        response = requests.post(
            f"{API_BASE_URL}/summarize",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            # Service unavailable, likely model missing/not loaded
            try:
                err = response.json()
                detail = err.get('detail') or err
            except Exception:
                detail = response.text
            st.error(f"Özetleme servisi şu anda kullanılamıyor (503): {detail}\nLütfen /models/status adresinden model durumunu kontrol edin.")
            return None
        else:
            try:
                err = response.json()
            except Exception:
                err = response.text
            st.error(f"Özetleme hatası ({response.status_code}): {err}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API bağlantı hatası: {e}")
        return None

def answer_question(context: str, question: str, max_length: int = 128) -> Optional[dict]:
    """Soruya cevap verir"""
    try:
        # API's QA endpoint is /qa (enhanced_multi_model_api.py)
        response = requests.post(
            f"{API_BASE_URL}/qa",
            json={"context": context, "question": question, "max_length": max_length},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Soru-cevap hatası: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API bağlantı hatası: {e}")
        return None

def main():
    """Ana uygulama fonksiyonu"""
    
    # Başlık
    st.markdown('<h1 class="main-header">🤖 Türkçe Haber Chatbotu</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        # API durumu
        api_status = check_api_health()
        if api_status:
            st.success("✅ API Bağlantısı Başarılı")
        else:
            st.error("❌ API Bağlantısı Başarısız")
            st.warning("API'nin çalıştığından emin olun: `uvicorn api.app:app --reload`")
        
        st.markdown("---")
        
        # Model ayarları
        st.subheader("Model Parametreleri")
        summary_max_length = st.slider("Özet Maksimum Uzunluk", 50, 256, 128)
        answer_max_length = st.slider("Cevap Maksimum Uzunluk", 32, 256, 128)
        # Allow user to select which summarization model to call
        model_options = ["multitask-lora-fast", "multitask-lora", "mt5"]
        selected_model = st.selectbox("Özetleme modeli seçin:", model_options, index=0)
        
        st.markdown("---")
        
        # Örnek metinler
        st.subheader("📰 Örnek Haber Metinleri")
        example_texts = {
            "Teknoloji Haberi": """
            Türkiye'deki teknoloji şirketleri son yıllarda hızla büyüyor. Özellikle fintech, 
            e-ticaret ve oyun sektörlerinde önemli gelişmeler yaşanıyor. İstanbul ve Ankara'daki 
            teknoloji merkezleri yeni startup'lara ev sahipliği yapıyor. Uzmanlar, bu trendin 
            gelecek yıllarda da devam edeceğini öngörüyor. Yapay zeka ve makine öğrenmesi 
            alanlarında da Türk şirketleri önemli projeler geliştiriyor.
            """,
            "Sağlık Haberi": """
            Sağlık Bakanlığı, hastanelerde dijital dönüşüm projelerini hızlandırıyor. 
            Elektronik reçete sistemi yaygınlaştırılırken, telemedicine uygulamaları da artıyor. 
            Hastaların randevu alma süreçleri dijitalleşiyor ve bekleme süreleri kısalıyor. 
            Bu gelişmeler sağlık hizmetlerinin kalitesini artırıyor. Özellikle kırsal bölgelerde 
            yaşayan vatandaşlar artık uzaktan sağlık hizmeti alabiliyorlar.
            """,
            "Eğitim Haberi": """
            Milli Eğitim Bakanlığı, okullarda yapay zeka destekli eğitim sistemlerini pilot 
            uygulamaya aldı. Öğrencilerin bireysel öğrenme hızlarına göre uyarlanabilen sistemler 
            test ediliyor. Öğretmenler de AI araçlarını kullanarak ders planları hazırlıyor. 
            Bu teknolojiler eğitimde kişiselleştirme imkanı sunuyor. Dijital okuryazarlık 
            müfredatı da güncelleniyor.
            """
        }
        
        selected_example = st.selectbox("Örnek seçin:", [""] + list(example_texts.keys()))
        
        if selected_example and st.button("Örneği Kullan"):
            st.session_state["example_text"] = example_texts[selected_example].strip()
    
    # Ana içerik
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Haber Metni")
        
        # Metin girişi
        default_text = st.session_state.get("example_text", "")
        news_text = st.text_area(
            "Özetlemek istediğiniz haber metnini buraya yapıştırın:",
            value=default_text,
            height=200,
            placeholder="Haber metnini buraya yazın veya yapıştırın..."
        )
        
        # Özet butonu
        if st.button("📋 Özet Oluştur", disabled=not api_status or not news_text.strip()):
            if news_text.strip():
                with st.spinner("Özet oluşturuluyor..."):
                    summary_result = summarize_text(news_text.strip(), summary_max_length, selected_model)
                    
                    if summary_result:
                        st.session_state["summary"] = summary_result
                        st.session_state["current_text"] = news_text.strip()
        
        # Özet sonucu
        if "summary" in st.session_state:
            st.markdown("### 📋 Özet")
            summary_data = st.session_state["summary"]
            
            st.markdown(f"""
            <div class="summary-box">
                <strong>📝 Özet:</strong><br><br>
                <span style="color: #333333; font-size: 16px; line-height: 1.5;">{summary_data['summary']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # İstatistikler
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                # Prefer API-provided input_length, fallback to session text
                original_len = summary_data.get('input_length') or summary_data.get('original_length')
                if original_len is None:
                    original_text = st.session_state.get('current_text', '')
                    original_len = len(original_text.split()) if original_text else 0
                st.metric("📊 Orijinal Kelime", original_len)
            with col_stat2:
                summary_len = summary_data.get('summary_length')
                if summary_len is None:
                    summary_text = summary_data.get('summary', '')
                    summary_len = len(summary_text.split()) if summary_text else 0
                st.metric("📋 Özet Kelime", summary_len)
            with col_stat3:
                # Sıkıştırma oranı hesaplamasında sıfıra bölmeyi önle
                if original_len and summary_len is not None:
                    compression_ratio = (1 - summary_len / original_len) * 100
                    st.metric("📉 Sıkıştırma Oranı", f"{compression_ratio:.1f}%")
                else:
                    st.metric("📉 Sıkıştırma Oranı", "N/A")
    
    with col2:
        st.subheader("❓ Soru-Cevap")
        
        # Soru girişi
        if "current_text" in st.session_state:
            question = st.text_input(
                "Haber hakkında sormak istediğiniz soruyu yazın:",
                placeholder="Örn: Hangi sektörlerde gelişmeler yaşanıyor?"
            )
            
            # Cevap butonu
            if st.button("💬 Cevap Al", disabled=not api_status or not question.strip()):
                if question.strip():
                    with st.spinner("Cevap aranıyor..."):
                        answer_result = answer_question(
                            st.session_state["current_text"], 
                            question.strip(), 
                            answer_max_length
                        )
                        
                        if answer_result:
                            st.markdown("### 💬 Cevap")
                            st.markdown(f"""
                            <div class="answer-box">
                                <strong>❓ Soru:</strong> <span style="color: #333333; font-size: 15px;">{answer_result['question']}</span><br><br>
                                <strong>💡 Cevap:</strong> <span style="color: #333333; font-size: 16px; line-height: 1.5;">{answer_result['answer']}</span>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Örnek sorular
            st.markdown("#### 💡 Örnek Sorular")
            example_questions = [
                "Ana konular nelerdir?",
                "Hangi gelişmeler yaşanıyor?",
                "Kim bu gelişmelerden sorumlu?",
                "Ne zaman uygulanacak?",
                "Hangi faydalar sağlanıyor?"
            ]
            
            for eq in example_questions:
                if st.button(f"💭 {eq}", key=f"eq_{hash(eq)}"):
                    with st.spinner("Cevap aranıyor..."):
                        answer_result = answer_question(
                            st.session_state["current_text"], 
                            eq, 
                            answer_max_length
                        )
                        
                        if answer_result:
                            st.markdown("### 💬 Cevap")
                            st.markdown(f"""
                            <div class="answer-box">
                                <strong>❓ Soru:</strong> <span style="color: #333333; font-size: 15px;">{answer_result['question']}</span><br><br>
                                <strong>💡 Cevap:</strong> <span style="color: #333333; font-size: 16px; line-height: 1.5;">{answer_result['answer']}</span>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("💡 Önce bir haber metni girin ve özetleyin, sonra soru sorabilirsiniz.")
    
    # Alt bilgi
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Bu chatbot Türkçe haber metinlerini özetler ve haber içeriği hakkında sorulara cevap verir.</p>
        <p></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
