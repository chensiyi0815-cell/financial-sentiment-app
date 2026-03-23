import streamlit as st
from transformers import pipeline
import torch
import pandas as pd

# ==================== Global Page Setup ====================
st.set_page_config(page_title="Financial News Smart Analysis Engine", page_icon="📈", layout="wide")

st.title("📈 Financial News Smart Analysis Engine")

# Introduction and supported categories
st.markdown("Enter a financial news statement, and the system will automatically perform 'Sentiment Diagnosis' and 'Topic Classification'.")
st.markdown("""
**Supported 6 Major Topic Categories:**
* `M&A | Investments`
* `Company | Product News`
* `Stock`
* `Macro`
* `Financials`
* `Others`
""")

# ==================== Core Business Mapping Function ====================
def nickmuchi_to_6(nickmuchi_label):
    mapping = {
        "Analyst Update": "Others",
        "Fed | Central Banks": "Macro",
        "Company | Product News": "Company | Product News",
        "Treasuries | Corporate Debt": "Financials",
        "Dividend": "Stock",
        "Earnings": "Financials",
        "Energy | Oil": "Others",
        "Financials": "Financials",
        "Currencies": "Macro",
        "General News | Opinion": "Others",
        "Gold | Metals | Materials": "Macro",
        "IPO": "Stock",
        "Legal | Regulation": "Company | Product News",
        "M&A | Investments": "M&A | Investments",
        "Macro": "Macro",
        "Markets": "Macro",
        "Politics": "Others",
        "Personnel Change": "Company | Product News",
        "Stock Commentary": "Stock",
        "Stock Movement": "Stock",
    }
    return mapping.get(nickmuchi_label, 'Others')

# ==================== Initialize Session State ====================
if 'history' not in st.session_state:
    st.session_state.history = []

# ==================== Dual Engine Loading ====================
@st.cache_resource(show_spinner="Loading AI analysis engines, please wait...")
def load_models():
    hf_token = st.secrets["HF_TOKEN"]
    
    # 1. Sentiment Analysis Model
    sentiment_model_id = "ychenqz/financial-sentiment-model" 
    sentiment_pipe = pipeline("text-classification", model=sentiment_model_id, device=-1)
    
    # 2. Topic Classification Model
    topic_model_id = "nickmuchi/finbert-tone-finetuned-finance-topic-classification"
    topic_pipe = pipeline("text-classification", model=topic_model_id, device=-1, token=hf_token)
    
    return sentiment_pipe, topic_pipe

try:
    # Initialize both engines
    sentiment_classifier, topic_classifier = load_models()
    
    # ==================== User Interaction Section ====================
    st.markdown("### 🔍 Real-time News Analysis")
    
    # Text Input Area
    user_input = st.text_area(
        "Enter English financial news here (💡 Tip: Inputs with < 300 words yield more accurate results):", 
        value="",               
        height=100,
        placeholder="Please enter text here"  
    )

    # Dynamic word count logic
    word_count = len(user_input.split()) if user_input.strip() else 0
    
    # Word count display
    if word_count > 300:
        st.error(f"📊 Current word count: **{word_count} / 300** (Exceeded limit, please shorten!)")
    else:
        st.caption(f"📊 Current word count: **{word_count} / 300** words")

    # Execution button and Guardrail logic
    if st.button("🚀 Start Multi-dimensional Analysis", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Text cannot be empty, please enter news content.")
        elif word_count > 300:
            st.error("🚨 Blocked: Input exceeds the 300-word limit! To ensure inference accuracy and speed, please shorten the text and try again.")
        else:
            with st.spinner("Running dual engines..."):
                # ---------- Execute Pipeline 1 (Sentiment) ----------
                sent_result = sentiment_classifier(user_input)[0]
                sent_label = sent_result['label']
                sent_score = sent_result['score']
                
                # ---------- Execute Pipeline 2 (Topic) ----------
                topic_result = topic_classifier(user_input)[0]
                raw_topic = topic_result['label']
                topic_score = topic_result['score']
                mapped_topic = nickmuchi_to_6(raw_topic)
                
            # ==================== Results Display Section ====================
            st.markdown("#### 🎯 Comprehensive Diagnosis Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if "POSITIVE" in sent_label.upper():
                    st.success(f"**Sentiment**: 😊 Positive (Confidence: {sent_score:.1%})")
                    display_sent = "😊 Positive"
                elif "NEGATIVE" in sent_label.upper():
                    st.error(f"**Sentiment**: 😡 Negative (Confidence: {sent_score:.1%})")
                    display_sent = "😡 Negative"
                else:
                    st.info(f"**Sentiment**: 😐 Neutral (Confidence: {sent_score:.1%})")
                    display_sent = "😐 Neutral"
            
            with col2:
                if mapped_topic == "Others":
                    st.warning(f"**Core Topic**: 📦 {mapped_topic}")
                else:
                    st.info(f"**Core Topic**: 🏷️ {mapped_topic}")
            
            # 🌟 FIXED: Strictly enforcing the 5 requested English columns
            st.session_state.history.insert(0, {
                "Original News": user_input,
                "Sentiment": display_sent,
                "Topic": mapped_topic,
                "Sentiment Confidence": f"{sent_score:.1%}",
                "Topic Confidence": f"{topic_score:.1%}"
            })
            
            st.session_state.history = st.session_state.history[:50]

    # ==================== History Dashboard Section ====================
    if st.session_state.history:
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("### 📝 Historical Analysis Records")
        
        history_df = pd.DataFrame(st.session_state.history)
        
        col_search, col_download, col_clear = st.columns([2, 1, 1])
        
        with col_search:
            search_query = st.text_input("🔍 Search history (Enter news keywords):", placeholder="e.g., Microsoft")
            
        with col_download:
            st.write("") 
            st.write("")
            csv_data = history_df.to_csv(index=False).encode('utf-8-sig') 
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="finance_analysis_history.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with col_clear:
            st.write("") 
            st.write("")
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        if search_query:
            # Match the exact English column name "Original News"
            display_df = history_df[history_df["Original News"].str.contains(search_query, case=False, na=False)]
            st.caption(f"Found {len(display_df)} records containing '{search_query}' (Max 50 saved, showing top 5):")
        else:
            display_df = history_df
            st.caption(f"Currently saved {len(history_df)} records (Max 50 saved, showing top 5):")
            
        st.dataframe(display_df.head(5), use_container_width=True)

except Exception as e:
    st.error(f"Engine initialization failed, please check network or model config. Error log: {e}")

st.markdown("---")
st.caption("Architecture: Sentiment Engine (Custom Fine-tuned) + Topic Engine (finbert-tone-finetuned) | Powered by Streamlit")
