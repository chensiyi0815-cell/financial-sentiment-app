import streamlit as st
from transformers import pipeline
import torch
import pandas as pd

# 页面设置
st.set_page_config(page_title="金融情感分析", page_icon="📈", layout="wide")

st.title("📈 金融新闻情感分析系统")
st.write("输入一条金融新闻语句，模型将自动识别其情绪倾向。")

# ==================== 初始化 Session State ====================
# 如果当前会话还没有 history 变量，就初始化一个空列表
if 'history' not in st.session_state:
    st.session_state.history = []

# ==================== 加载模型 ====================
@st.cache_resource
def load_model():
    model_id = "Norarara/financial-sentiment-model"
    # 强制指定使用 CPU 运行
    return pipeline("text-classification", model=model_id, device=-1)

try:
    classifier = load_model()
    
    # 用户输入区
    user_input = st.text_area("在此输入英文语句：", "The company's revenue increased by 20% this quarter.")

    if st.button("🚀 开始分析"):
        if user_input:
            # 运行预测
            result = classifier(user_input)[0]
            label = result['label']
            score = result['score']
            
            # 结果展示
            st.subheader("当前分析结果：")
            if "POSITIVE" in label.upper():
                st.success(f"😊 正面情绪 (置信度: {score:.2%})")
                display_label = "😊 正面 (Positive)"
            elif "NEGATIVE" in label.upper():
                st.error(f"😡 负面情绪 (置信度: {score:.2%})")
                display_label = "😡 负面 (Negative)"
            else:
                st.info(f"😐 中性情绪 (置信度: {score:.2%})")
                display_label = "😐 中性 (Neutral)"
            
            # 将本次结果插入到历史记录的开头 (最上面)
            st.session_state.history.insert(0, {
                "新闻语句": user_input,
                "情感倾向": display_label,
                "置信度": f"{score:.2%}"
            })
            
        else:
            st.warning("⚠️ 请输入文字后再点击分析。")

    # ==================== 历史记录展示区 ====================
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📝 历史预测记录")
        
        # 将字典列表转换为 DataFrame 以便更美观地展示
        history_df = pd.DataFrame(st.session_state.history)
        
        # 使用 st.dataframe 展示表格，并设置宽度自适应
        st.dataframe(history_df, use_container_width=True)
        
        # 添加清空记录按钮
        if st.button("🗑️ 清空历史记录"):
            st.session_state.history = []
            st.rerun() # 重新运行应用以刷新页面

except Exception as e:
    st.error(f"模型加载出错，请检查模型是否已公开。错误详情: {e}")

st.markdown("---")
st.caption("Model: Norarara/financial-sentiment-model | Powered by Streamlit")
