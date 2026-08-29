# 分別呈現Pixel和HOG電腦看的照片
import cv2
import numpy as np
import streamlit as st
import os
from skimage import exposure
from PIL import Image
from datetime import datetime
from skimage.feature import hog

st.title("Pixel & HOG 電腦看的圖像")
save_folder = "Pixel_HOG_extractors_saved"
os.makedirs(save_folder, exist_ok=True)
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])
# 將所有圖像處理邏輯，放在 if 裡面
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # 解碼成 NumPy 陣列 (BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 轉換成 RGB 格式 
    original_img = img.copy() # 複製給 original_img，位於記憶體中的 RGB 格式 NumPy 陣列
    
    #原始上傳圖像
    st.subheader("原始上傳圖像")
    st.image(original_img, use_column_width=True)
    #像素特徵提取
    def extract_pixel(original_img): # 定義函數：建立名為 extract_pixel 的函數，接收圖片的 NumPy 陣列（original_img）作為輸入參數。
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY) # 轉換為灰階：將 RGB 彩色圖轉為單通道灰階圖，排除顏色干擾並減少資料量。
        display_img = cv2.resize(gray_img, (256, 256)) # 2D 矩陣，256 × 256，縮放統一尺寸：將圖片強制調整為 256 × 256 像素，確保不論原圖多大，輸出維度都固定。
        pixel_vector = display_img.flatten().astype(float) # 1D 陣列，長度 65536，將灰階圖展平並轉型：將 2D 灰階圖轉換為 1D 向量（flatten），並將資料型態轉為 float，以便後續機器學習模型使用。
        return display_img, pixel_vector  # 加上 return 回傳兩個變數
    
    # display_img（2D 矩陣，256 × 256）：用來顯示視覺圖像與儲存圖片
    # pixel_vector（1D 陣列，長度 65536）：用來儲存特徵向量，提供給機器學習模型
    display_img, pixel_vector = extract_pixel(original_img) # 【關鍵】呼叫函數，取得 display_img 與 pixel_vector
    #像素特徵電腦看的圖像
    st.subheader("像素特徵電腦看的圖像") 
    st.image(display_img, use_column_width=True) # 顯示灰階圖像，使用 display_img 變數
    #HOG特徵提取
    def extract_hog(original_img): # 定義函數：建立名為 extract_hog 的函數，接收圖片的 NumPy 陣列（original_img）作為輸入參數。
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY) # 轉換為灰階：將 RGB 彩色圖轉為單通道灰階圖，排除顏色干擾並減少資料量。
        display_img = cv2.resize(gray_img, (256, 256)) # 2D 矩陣，256 × 256，縮放統一尺寸：將圖片強制調整為 256 × 256 像素，確保不論原圖多大，輸出維度都固定。
        fd, hog_image = hog(display_img, 
                                    orientations=8, 
                                    pixels_per_cell=(4, 4), # 全圖被切割成64*64＝4096 個格子
                                    cells_per_block=(2, 2), 
                                    visualize=True,    # visualize=True 想要拿圖片顯示，就必須寫 fd, hog_image =
                                    channel_axis=None) # 計算 HOG 特徵向量，fd 為 1D 陣列，長度 3780，包含了圖像的梯度方向與強度資訊
        return fd, hog_image  # 回傳特徵向量與視覺化圖片
    fd, hog_image = extract_hog(original_img) # 【關鍵】呼叫函數，取得 fd 與 hog_image
    # 將 HOG 特徵圖像的像素值重新縮放到 0~255 的範圍，並轉換為 uint8 型態，以便後續顯示與儲存
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range="image", out_range=(0, 255)).astype(np.uint8) 
    #HOG特徵電腦看的圖像
    st.subheader("HOG特徵電腦看的圖像")
    st.image(hog_image_rescaled, use_column_width=True)


    #Pixel自動儲存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Pixel_filename = os.path.join(save_folder, f"Pixel_{timestamp}.png")
    Pixel_bgr = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)  
    cv2.imwrite(Pixel_filename, Pixel_bgr)
    st.success(f"像素特徵向量已經儲存 {Pixel_filename}")

    #HOG自動儲存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    HOG_filename = os.path.join(save_folder, f"HOG_{timestamp}.png")
    # 先將 float64 強制轉型為 uint8，再進行色彩轉換
    hog_image_uint8 = hog_image_rescaled.astype(np.uint8) # 將 float64 轉型為 uint8，避免 cv2.imwrite() 出現錯誤
    HOG_bgr = cv2.cvtColor(hog_image_uint8, cv2.COLOR_GRAY2BGR) # 將灰階圖轉換為 BGR 格式，方便儲存成彩色圖片
    cv2.imwrite(HOG_filename, HOG_bgr)
    st.success(f"HOG特徵向量已經儲存 {HOG_filename}")