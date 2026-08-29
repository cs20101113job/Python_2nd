import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os
from datetime import datetime
# import tensorflow as tf #針對VGG16模型
from tensorflow.keras.preprocessing import image #針對VGG16模型的圖片預處理
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.applications.vgg16 as vgg16 #VGG16模型載入
st.title("VGG16模型預測")
save_folder = "vgg16model_operations_saved"
os.makedirs(save_folder, exist_ok=True)
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    original_img = img.copy()   

    # VGG16模型訓練
    model=vgg16.VGG16(weights='imagenet',include_top=True,input_shape=(224,224,3))
    # 使用 OpenCV 調整尺寸為 (224, 224)
    display_img = cv2.resize(img, (224, 224))
    img_array = display_img.copy()  # 直接複製 NumPy 陣列即可
    
    # 擴充維度並轉換型態為 float32
    img_array = np.expand_dims(display_img, axis=0).astype(np.float32)
    
    # 使用 VGG16 標準預處理 (會處理 Channel 順序與減去均值)
    # tensorflow格式轉換為VGG16格式，進行標準化
    img_preprocessed = vgg16.preprocess_input(img_array.copy())
   
    prediction = model.predict(img_preprocessed)
    # 模型預測
    # 一千個結果，我們想要查看最大的可能性是甚麼
    # 將tuple型態資料做個處理
    # [0] 的作用是取出批次（Batch）中「第一張圖片」的預測結果列表，加了 [0] 後，變數才能順利被後續的 for i 迴圈解構並印出內容。
    decoded_results = vgg16.decode_predictions(prediction ,top=3)[0]
    
    # :.2f 格式化控制符，代表強制四捨五入保留至小數點後第 2 位，**：代表粗體。
    for i, (imagenet_id, label, score) in enumerate(decoded_results):st.write(f"**Top {i+1}:** {label} (信心值: {score * 100:.2f}%)") # 
     #操作前圖像
    st.subheader("操作前的圖像")
    st.image(original_img, use_container_width=True)
    #VGG16模型判斷
    st.subheader("預測結果")
    st.image(display_img, use_container_width=True)
    #自動儲存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vgg16_operation_filename = os.path.join(save_folder, f"vgg16_operation_{timestamp}.png")
    image_operation_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)  
    cv2.imwrite(vgg16_operation_filename, image_operation_bgr)
    st.success(f"VGG16模型預測已經儲存 {vgg16_operation_filename}")

