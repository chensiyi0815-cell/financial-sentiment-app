import streamlit as st
from transformers import pipeline

# ==================== 1. 页面配置与标题 ====================
# 设置页面标题、图标和布局
st.set_page_config(
    page_title="📈 金融新闻情感分析系统",
    page_icon="https://huggingface.co/front/assets/huggingface_hub/logo-color.svg",
    layout="wide"
)

# 显示一个 illustrative image（可选，增加美观度）
st.image("https://images.pexels.com/photos/590022/pexels-photo-590022.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1", caption="通过情感分析洞察市场情绪", use_column_width=True)

st.title("📈 金融新闻情感分析系统")
st.markdown("""
### 💡 简介
这是一个利用你的微调模型构建的情感分析系统。只需在下方输入一条金融相关的语句（英文），系统就能自动帮您识别其情绪。

该系统适用于分析金融新闻、财报、社交媒体评论等。

* **基座模型:** Norarara/financial-sentiment-model
* **训练数据:** Financial Phrasebank Dataset
* **模型功能:** 将语句分类为 **正面 (Positive)**、**负面 (Negative)** 或 **中性 (Neutral)**。

---
""")

# ==================== 2. 加载模型 (使用了缓存以提升性能) ====================
@st.cache_resource() # 将模型加载结果缓存，避免每次刷新页面都重新加载
def load_sentiment_pipeline():
    # 使用 pipeline 轻松加载用户上传的模型
    model_id = "Norarara/financial-sentiment-model"
    st.info(f"正在加载模型 {model_id} ...")
    try:
        # 加载分类 pipeline
        sentiment_pipe = pipeline("text-classification", model=model_id, return_all_scores=True)
        st.success("模型加载完成！")
        return sentiment_pipe
    except Exception as e:
        st.error(f"模型加载失败。请检查模型路径或 Hugging Face 状态。\n错误信息: {e}")
        return None

# 初始化模型
sentiment_analyzer = load_sentiment_pipeline()

# ==================== 3. 用户输入界面 ====================
st.subheader("✍️ 请输入待分析的金融语句 (英文)：")

# 提供一个默认的示例，让用户更容易上手
example_text = "The company reported stronger-than-expected quarterly earnings, leading to a significant stock price increase."
user_input = st.text_area("在此处输入文本：", value=example_text, height=150)

# ==================== 4. 情绪分析核心逻辑 ====================
# 定义标签到用户友好文本和 emoji 的映射
label_map = {
    "POSITIVE": "🟢 正面 (Positive)",
    "NEGATIVE": "🔴 负面 (Negative)",
    "NEUTRAL": "⚪ 中性 (Neutral)"
}

# 用于在结果展示中使用的不同颜色的容器
result_colors = {
    "POSITIVE": st.success,
    "NEGATIVE": st.error,
    "NEUTRAL": st.info
}

# 点击按钮触发分析
if st.button("🚀 开始分析情感"):
    if sentiment_analyzer and user_input:
        with st.spinner("🔄 系统正在紧张分析中，请稍候..."):
            # 运行模型进行预测
            # pipeline 返回一个包含所有得分的列表，我们取第一个结果
            result = sentiment_analyzer(user_input)[0]
            
            # 找出得分最高的情绪标签
            top_prediction = max(result, key=lambda x: x['score'])
            label = top_prediction['label']
            score = top_prediction['score']

            st.markdown("---")
            st.subheader("📊 分析结果展示：")

            # 用于展示结果的 UX 优化
            # 1. 显示情感标签
            friendly_label = label_map.get(label, label) # 如果标签不在映射中，显示原标签
            
            # 使用特定颜色的 st.success/st.error/st.info 展示情感标签
            result_colors.get(label, st.write)(f"### 该语句的情绪倾向为：{friendly_label}")
            
            # 2. 显示置信度
            confidence_percentage = f"{score:.2%}" # 格式化为百分比
            st.metric("模型置信度 (Confidence Score)", confidence_percentage)
            
            # 3. 显示详细的情绪概率分布情况（可选，增加可解释性）
            with st.expander("👉 查看详细的情绪概率分布"):
                st.write("各情绪类别的概率：")
                for item in result:
                    st.write(f"- {label_map.get(item['label'], item['label'])}: `{item['score']:.4%}`")

    elif not user_input:
        st.warning("⚠️ 请先输入一些待分析的文本。")
    else:
        st.error("⚠️ 模型未能成功加载，无法进行分析。")

# ==================== 5. 侧边栏/底部信息 ====================
st.markdown("---")
st.markdown("#### About")
st.markdown("""
此项目展示了基于 Hugging Face 模型在 Streamlit 上的部署。

由 **Norarara** 训练并维护。

* [模型主页 (Hugging Face)](https://huggingface.co/Norarara/financial-sentiment-model)
* [训练代码 (可选)]()
""")
