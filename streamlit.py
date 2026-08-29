import cv2
import numpy as np
import streamlit as st
from skimage.feature import hog
from skimage import exposure

# 網頁標題與介紹
st.set_page_config(layout="wide") # 設定網頁為寬螢幕佈局
st.title("🖼️ 電腦視覺特徵提取練習場")
st.markdown("上傳一張照片，對比電腦在 **Pixel 原始像素** 與 **HOG 邊緣梯度** 下所看到的影像特徵。")

# 側邊欄：使用者可以即時調整 HOG 的參數
st.sidebar.header("🛠️ HOG 參數微調")
orientations = st.sidebar.slider("方向梯度數量 (orientations)", min_value=4, max_value=12, value=9)
pixels_per_cell = st.sidebar.slider("Cell 像素大小 (pixels_per_cell)", min_value=4, max_value=32, value=8, step=4)

# 檔案上傳元件
uploaded_file = st.file_uploader("請選擇並上傳一張電腦中的照片 (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. 讀取上傳的影像
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. 提取 Pixel 特徵 (轉換為灰階)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. 提取 HOG 特徵並視覺化
    with st.spinner("正在計算 HOG 特徵中..."):
        features, hog_image = hog(
            gray_image, 
            orientations=orientations, 
            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
            cells_per_block=(2, 2), 
            visualize=True, 
            channel_axis=None
        )
        # 增強 HOG 影像對比度方便網頁顯示
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # 4. 網頁版面配置：並排呈現三個畫面
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. 原始彩色照片")
        st.image(image_rgb, use_column_width=True, caption="人類眼中的彩色世界")
        
    with col2:
        st.subheader("2. Pixel 特徵 (灰階)")
        st.image(gray_image, use_column_width=True, caption="電腦看見的亮度絕對值 (每個點 0~255)")
        st.write(f"📊 矩陣維度：{gray_image.shape}")
        
    with col3:
        st.subheader("3. HOG 特徵 (梯度輪廓)")
        st.image(hog_image_rescaled, clamp=True, channels="GRAY", use_column_width=True, caption="電腦看見的邊緣形狀與方向")
        st.write(f"📊 特徵向量長度：{len(features)} 維")

else:
    st.info("💡 請在上方上傳照片以開始進行特徵對比練習。")
