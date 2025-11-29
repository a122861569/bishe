import streamlit as st
from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# -------------------------
# 配置路径
# -------------------------
MODEL_PATH = "models/best.pt"
UPLOAD_DIR = "uploads"
HISTORY_DIR = "history"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# -------------------------
# 加载模型（缓存）
# -------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------
# 页面配置
# -------------------------
st.set_page_config(
    page_title="饮料瓶智能监控系统",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# 页面风格（高科技主题）
# -------------------------
st.markdown("""
<style>
body { background-color: #0B0E13; color: #FFFFFF; }
h1, h2, h3, h4 { color: #00FFFF; }
.stButton>button { background-color:#1E90FF; color:white; border-radius:5px; }
.card {
    background-color:#1C1F26;
    padding:15px;
    border-radius:12px;
    margin-bottom:10px;
    box-shadow:0 0 10px rgba(0,255,255,0.3);
}
</style>
""", unsafe_allow_html=True)

st.title("🥤 饮料瓶智能监控系统")

# -------------------------
# 初始化统计数据
# -------------------------
if "total_images" not in st.session_state:
    st.session_state.total_images = 0
if "confidences" not in st.session_state:
    st.session_state.confidences = []
if "brand_counts" not in st.session_state:
    st.session_state.brand_counts = {}
if "history_records" not in st.session_state:
    st.session_state.history_records = []
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None  # 用于标记新上传的文件

# -------------------------
# 页面布局
# -------------------------
tab1, tab2 = st.tabs(["实时摄像头", "上传图片检测"])

# ==========================================================
# 1️⃣ 实时摄像头监控
# ==========================================================
with tab1:
    st.header("摄像头实时监控")
    run_camera = st.checkbox("开启摄像头监控")
    FRAME_WINDOW = st.image([])

    if run_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ 无法打开摄像头")
        else:
            st.write("按 Ctrl+C 或关闭浏览器停止摄像头")
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(annotated_frame, channels="RGB")

                # 更新统计数据
                boxes = results[0].boxes
                confidences = [float(b.conf) for b in boxes]
                st.session_state.total_images += 1
                st.session_state.confidences.extend(confidences)

                for b in boxes:
                    brand = results[0].names[int(b.cls)]
                    st.session_state.brand_counts[brand] = st.session_state.brand_counts.get(brand, 0) + 1

            cap.release()

# ==========================================================
# 2️⃣ 上传图片检测（立即增加总次数 + 置信度）
# ==========================================================
with tab2:
    st.header("上传图片进行检测")

    col_upload, col_stats = st.columns([3, 2])

    with col_upload:
        uploaded_file = st.file_uploader("选择图片上传", type=["jpg","jpeg","png"], key="uploader")

    with col_stats:
        avg_conf = np.mean(st.session_state.confidences) if st.session_state.confidences else 0
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <h3>📊 本站检测总次数</h3>
            <p style="font-size:28px;color:white;">{st.session_state.total_images}</p>
            <h4>平均置信度 / 准确率</h4>
            <p style="font-size:22px;color:white;">{avg_conf:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # ----------- 立即处理新上传文件 -----------
    if uploaded_file is not None:
        # 判断是不是新文件，避免重复检测
        if st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.last_uploaded_file = uploaded_file.name

            # 读取图片
            file_bytes = np.asarray(bytearray(uploaded_file.getbuffer()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # YOLO 检测
            results = model(image)
            annotated_image = results[0].plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # 保存图片
            file_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            upload_path = os.path.join(UPLOAD_DIR, f"{file_timestamp}_{uploaded_file.name}")
            cv2.imwrite(upload_path, image)
            save_path = os.path.join(HISTORY_DIR, f"{file_timestamp}_result.jpg")
            cv2.imwrite(save_path, annotated_image)

            # ----------- 更新统计 -----------
            st.session_state.total_images += 1
            boxes = results[0].boxes
            confidences = [float(b.conf) for b in boxes]
            st.session_state.confidences.extend(confidences)
            for b in boxes:
                brand = results[0].names[int(b.cls)]
                st.session_state.brand_counts[brand] = st.session_state.brand_counts.get(brand, 0) + 1

            # ----------- 保存历史记录 -----------
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image": annotated_image_rgb,
                "results": [
                    {"类别": results[0].names[int(b.cls)], "置信度": f"{float(b.conf):.2f}"}
                    for b in boxes
                ]
            }
            st.session_state.history_records.append(record)

    # 上传框为空时清除 last_uploaded_file
    if uploaded_file is None:
        st.session_state.last_uploaded_file = None

    # ----------- 历史图片缩略图 + 点击查看大图 -----------
    if st.session_state.history_records:
        st.subheader("历史图片检测记录（点击按钮查看大图）")
        for idx, rec in enumerate(reversed(st.session_state.history_records)):
            st.markdown(f"**时间：{rec['timestamp']}**")
            st.image(rec["image"], width=250)
            if st.button("🔍 查看大图", key=f"view_{idx}"):
                st.image(rec["image"], use_container_width=True)
            if rec["results"]:
                st.table(rec["results"])
            st.markdown("---")

