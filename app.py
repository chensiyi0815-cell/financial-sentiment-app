import streamlit as st
from transformers import pipeline
import torch
import pandas as pd

# ==================== 页面全局设置 ====================
st.set_page_config(page_title="金融新闻智能分析引擎", page_icon="📈", layout="wide")

st.title("📈 金融新闻智能分析引擎")
st.write("输入一条金融新闻语句，系统将自动进行「情感诊断」与「主题归类」。")

# ==================== 核心业务映射函数 ====================
# 使用经过验证的 nickmuchi 模型 6 大主题映射函数
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

# ==================== 初始化 Session State ====================
if 'history' not in st.session_state:
    st.session_state.history = []

# ==================== 双引擎加载 ====================
@st.cache_resource(show_spinner="正在加载 AI 分析引擎，请稍候...")
def load_models():
    # 1. 情感分析模型 (Pipeline 1)
    sentiment_model_id = "ychenqz/financial-sentiment-model" 
    sentiment_pipe = pipeline("text-classification", model=sentiment_model_id, device=-1)
    
    # 2. 主题分类模型 (Pipeline 2) - 盲测冠军模型
    topic_model_id = "nickmuchi/finbert-tone-finetuned-finance-topic-classification"
    topic_pipe = pipeline("text-classification", model=topic_model_id, device=-1)
    
    return sentiment_pipe, topic_pipe

try:
    # 唤醒双引擎
    sentiment_classifier, topic_classifier = load_models()
    
    # ==================== 用户交互区 ====================
    st.markdown("### 🔍 实时新闻解析")
    
    # 预设一条能够触发 M&A 或 Company 标签的测试新闻
    default_news = "Microsoft has officially completed its $68.7 billion acquisition of Activision Blizzard, leading to a major personnel change."
    user_input = st.text_area("在此输入一段英文金融新闻：", default_news, height=100)

    if st.button("🚀 开始多维分析", type="primary"):
        if user_input.strip():
            with st.spinner("双引擎运算中..."):
                # ---------- 执行 Pipeline 1: 情感推断 ----------
                sent_result = sentiment_classifier(user_input)[0]
                sent_label = sent_result['label']
                sent_score = sent_result['score']
                
                # ---------- 执行 Pipeline 2: 主题推断 ----------
                topic_result = topic_classifier(user_input)[0]
                raw_topic = topic_result['label']
                topic_score = topic_result['score']
                
                # 🌟 调用你的自定义函数进行主题翻译
                mapped_topic = nickmuchi_to_6(raw_topic)
                
            # ==================== 结果展示区 ====================
            st.markdown("#### 🎯 综合诊断结果")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 情感卡片
                if "POSITIVE" in sent_label.upper():
                    st.success(f"**情感倾向**: 😊 正面 (置信度: {sent_score:.1%})")
                    display_sent = "😊 正面"
                elif "NEGATIVE" in sent_label.upper():
                    st.error(f"**情感倾向**: 😡 负面 (置信度: {sent_score:.1%})")
                    display_sent = "😡 负面"
                else:
                    st.info(f"**情感倾向**: 😐 中性 (置信度: {sent_score:.1%})")
                    display_sent = "😐 中性"
            
            with col2:
                # 主题卡片：如果归类到了 Others，给一个灰色的视觉提示
                if mapped_topic == "Others":
                    st.secondary(f"**核心主题**: 📦 {mapped_topic} \n\n(底层原生标签: *{raw_topic}*, 置信度: {topic_score:.1%})")
                else:
                    st.info(f"**核心主题**: 🏷️ {mapped_topic} \n\n(底层原生标签: *{raw_topic}*, 置信度: {topic_score:.1%})")
            
            # 记录历史数据
            st.session_state.history.insert(0, {
                "新闻原文": user_input,
                "情感倾向": display_sent,
                "所属主题": mapped_topic,
                "情感置信度": f"{sent_score:.1%}",
                "主题置信度": f"{topic_score:.1%}"
            })
            
        else:
            st.warning("⚠️ 文本不能为空，请输入新闻内容。")

    # ==================== 历史记录看板 ====================
    if st.session_state.history:
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("### 📝 分析流水台账")
        
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("🗑️ 清空流水账"):
            st.session_state.history = []
            st.rerun()

except Exception as e:
    st.error(f"引擎初始化失败，请检查网络或模型配置。错误日志: {e}")

st.markdown("---")
st.caption("Architecture: Sentiment Engine (Custom Fine-tuned) + Topic Engine (finbert-tone-finetuned) | Powered by Streamlit")
