import streamlit as st
from transformers import pipeline
import torch
import pandas as pd

# ==================== 页面全局设置 ====================
st.set_page_config(page_title="金融新闻智能分析引擎", page_icon="📈", layout="wide")

st.title("📈 金融新闻智能分析引擎")

st.markdown("输入一条金融新闻语句，系统将自动进行「情感诊断」与「主题归类」。")

st.markdown("""
**支持识别的 6 大主题分类：**
* `M&A | Investments (投资并购)`
* `Company | Product News (公司/产品)`
* `Stock (股票市场)`
* `Macro (宏观经济)`
* `Financials (财务数据)`
* `Others (其他)`
""")

# ==================== 核心业务映射函数 ====================
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
    hf_token = st.secrets["HF_TOKEN"]
    
    # 1. 情感分析模型 
    sentiment_model_id = "ychenqz/financial-sentiment-model" 
    sentiment_pipe = pipeline("text-classification", model=sentiment_model_id, device=-1)
    
    # 2. 主题分类模型
    topic_model_id = "nickmuchi/finbert-tone-finetuned-finance-topic-classification"
    topic_pipe = pipeline("text-classification", model=topic_model_id, device=-1, token=hf_token)
    
    return sentiment_pipe, topic_pipe

try:
    # 唤醒双引擎
    sentiment_classifier, topic_classifier = load_models()
    
    # ==================== 用户交互区 ====================
    st.markdown("### 🔍 实时新闻解析")
    
    default_news = "Microsoft has officially completed its $68.7 billion acquisition of Activision Blizzard, leading to a major personnel change."
    # 优化 2：加入字数限制提示
    user_input = st.text_area("在此输入一段英文金融新闻（💡 提示：输入 <100 字的内容，判断更准确）：", default_news, height=100)

    if st.button("🚀 开始多维分析", type="primary"):
        if user_input.strip():
            with st.spinner("双引擎运算中..."):
                # ---------- 执行 Pipeline 1 ----------
                sent_result = sentiment_classifier(user_input)[0]
                sent_label = sent_result['label']
                sent_score = sent_result['score']
                
                # ---------- 执行 Pipeline 2 ----------
                topic_result = topic_classifier(user_input)[0]
                raw_topic = topic_result['label']
                topic_score = topic_result['score']
                mapped_topic = nickmuchi_to_6(raw_topic)
                
            # ==================== 结果展示区 ====================
            st.markdown("#### 🎯 综合诊断结果")
            col1, col2 = st.columns(2)
            
            with col1:
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
                if mapped_topic == "Others":
                    st.warning(f"**核心主题**: 📦 {mapped_topic}")
                else:
                    st.info(f"**核心主题**: 🏷️ {mapped_topic}")
            
            # 记录历史数据
            st.session_state.history.insert(0, {
                "新闻原文": user_input,
                "情感倾向": display_sent,
                "所属主题": mapped_topic,
                "情感置信度": f"{sent_score:.1%}",
                "主题置信度": f"{topic_score:.1%}"
            })
            
            # 优化 4：严格控制历史记录在 50 条以内，防止撑爆内存
            st.session_state.history = st.session_state.history[:50]
            
        else:
            st.warning("⚠️ 文本不能为空，请输入新闻内容。")

    # ==================== 历史记录看板 ====================
    if st.session_state.history:
        st.markdown("<br><hr>", unsafe_allow_html=True)
        # 优化 3：标题修改为“历史分析记录”
        st.markdown("### 📝 历史分析记录")
        
        history_df = pd.DataFrame(st.session_state.history)
        
        # 构建并排的控制面板：搜索框 (宽) + 下载按钮 (窄) + 清空按钮 (窄)
        col_search, col_download, col_clear = st.columns([2, 1, 1])
        
        with col_search:
            # 优化 4：关键词搜索框
            search_query = st.text_input("🔍 搜索历史记录（输入新闻关键词）：", placeholder="例如：Microsoft")
            
        with col_download:
            # 使用空行对齐输入框的高度
            st.write("") 
            st.write("")
            # 优化 4：生成可供下载的 CSV 文件数据
            csv_data = history_df.to_csv(index=False).encode('utf-8-sig') # utf-8-sig 保证中文导出在 Excel 不乱码
            st.download_button(
                label="📥 下载 CSV",
                data=csv_data,
                file_name="finance_analysis_history.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with col_clear:
            st.write("") 
            st.write("")
            # 优化 5：“清空历史记录”按钮
            if st.button("🗑️ 清空历史记录", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        # 根据搜索词过滤数据
        if search_query:
            # case=False 忽略大小写，na=False 防止空值报错
            display_df = history_df[history_df["新闻原文"].str.contains(search_query, case=False, na=False)]
            st.caption(f"为您找到 {len(display_df)} 条包含 '{search_query}' 的记录（最大保存 50 条，仅展示前 5 条）：")
        else:
            display_df = history_df
            st.caption(f"当前共保存 {len(history_df)} 条记录（最大保存 50 条，仅展示前 5 条）：")
            
        # 优化 4：前端仅展示 .head(5) 前 5 条数据
        st.dataframe(display_df.head(5), use_container_width=True)

except Exception as e:
    st.error(f"引擎初始化失败，请检查网络或模型配置。错误日志: {e}")

st.markdown("---")
st.caption("Architecture: Sentiment Engine (Custom Fine-tuned) + Topic Engine (finbert-tone-finetuned) | Powered by Streamlit")
