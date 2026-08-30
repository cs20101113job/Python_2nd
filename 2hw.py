import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os
from datetime import datetime
st.title("影像操作技術")
save_folder = "image_operations_saved"
os.makedirs(save_folder, exist_ok=True)
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    original_img = img.copy()   
    st.sidebar.header("操作控制項目")
    # 調整亮暗與對比
    # contrast_brightness = st.sidebar.slider("調整亮暗對比", 10, 200, 100)
    contrast = st.sidebar.slider("對比度", 0, 300, 100)
    brightness = st.sidebar.slider("亮度", -100, 100, 0)
    contrast = contrast / 100.0
    brightness = brightness
    display_img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    # 調整閥值Threshold
    threshold_mode = st.sidebar.selectbox("閥值模式", ["保留原圖", "物亮背暗", "物暗背亮", "高亮削留灰階", 
                    "背景黑亮留原亮","留暗亮清掉"])
    if threshold_mode == "保留原圖":
        pass  # display_img 維持 convertScaleAbs 處理後的樣子
    elif threshold_mode == "物亮背暗":
        retval, display_img = cv2.threshold(display_img, 127, 255, cv2.THRESH_BINARY)
    elif threshold_mode == "物暗背亮":
        retval, display_img = cv2.threshold(display_img, 127, 255, cv2.THRESH_BINARY_INV)
    elif threshold_mode == "高亮削留灰階":
        retval, display_img = cv2.threshold(display_img, 127, 255, cv2.THRESH_TRUNC)
    elif threshold_mode == "背景黑亮留原亮":
        retval, display_img = cv2.threshold(display_img, 127, 255, cv2.THRESH_TOZERO)
    elif threshold_mode == "留暗亮清掉":
        retval, display_img = cv2.threshold(display_img, 127, 255, cv2.THRESH_TOZERO_INV)
    # 邊緣偵測
    edge_mode = st.sidebar.selectbox("邊緣偵測", ["無", "Sobel", "Laplacian", "Canny"])
    if edge_mode == "Sobel":
        gray = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_x = cv2.convertScaleAbs(sobelx)
        sobel_y = cv2.convertScaleAbs(sobely)
        sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        display_img = cv2.convertScaleAbs(sobel_combined)
    elif edge_mode == "Laplacian":
        gray = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        display_img = cv2.convertScaleAbs(laplacian)
    elif edge_mode == "Canny":
        gray = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blurred, 80, 160)
        display_img = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    #操作前圖像
    st.subheader("操作前的圖像")
    st.image(original_img, use_container_width=True)
    #調整亮暗/對比/閥值/邊緣偵測後的圖像
    st.subheader("操作後的圖像")
    st.image(display_img, use_container_width=True)
    #自動儲存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_operation_filename = os.path.join(save_folder, f"image_operation_{timestamp}.png")
    image_operation_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)  
    cv2.imwrite(image_operation_filename, image_operation_bgr)
    st.success(f"影像操作技術已經儲存 {image_operation_filename}")

