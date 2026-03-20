import streamlit as st
from transformers import pipeline
import torch

# 页面设置
st.set_page_config(page_title="金融情感分析", page_icon="📈")

st.title("📈 金融新闻情感分析系统")
st.write("输入一条金融新闻语句，模型将自动识别其情绪倾向。")

# 加载模型 (增加缓存，避免重复加载)
@st.cache_resource
def load_model():
    model_id = "Norarara/financial-sentiment-model"
    # 强制指定使用 CPU 运行，因为 Streamlit Cloud 免费版没有 GPU
    return pipeline("text-classification", model=model_id, device=-1)

try:
    classifier = load_model()
    
    # 用户输入
    user_input = st.text_area("在此输入英文语句：", "The company's revenue increased by 20% this quarter.")

    if st.button("开始分析"):
        if user_input:
            result = classifier(user_input)[0]
            label = result['label']
            score = result['score']
            
            # 结果展示
            st.subheader("分析结果：")
            if "POSITIVE" in label.upper():
                st.success(f"😊 正面情绪 (置信度: {score:.2%})")
            elif "NEGATIVE" in label.upper():
                st.error(f"😡 负面情绪 (置信度: {score:.2%})")
            else:
                st.info(f"😐 中性情绪 (置信度: {score:.2%})")
        else:
            st.warning("请输入文字后再点击分析。")

except Exception as e:
    st.error(f"模型加载出错，请检查模型是否已公开。错误详情: {e}")

st.markdown("---")
st.caption("Model: Norarara/financial-sentiment-model | Powered by Streamlit")
