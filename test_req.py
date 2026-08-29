import requests

print("正在發送網路請求給 Google...")

try:
    # 改用絕對穩定的 Google 首頁進行測試
    url = "https://google.com"
    response = requests.get(url, timeout=5)

    if response.status_code == 200:
        print("\n🎉 連線完全成功！")
        print("你的 uv 虛擬環境與 requests 套件一切正常！")
        print(f"回應狀態碼：{response.status_code} (OK)")
    else:
        print(f"❌ 連線失敗，狀態碼：{response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"❌ 依然發生網路連線錯誤: {e}")

