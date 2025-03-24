import os
from dotenv import load_dotenv
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi
)

# 載入環境變數
load_dotenv()

# 導出 CHANNEL_SECRET
CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

def get_line_bot_api():
    """獲取 LINE Bot API 設定"""
    channel_secret = os.getenv('LINE_CHANNEL_SECRET')
    channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
    
    if not channel_secret or not channel_access_token:
        raise ValueError("未設置 LINE Bot 的 Channel Secret 或 Channel Access Token")
    
    # 創建配置
    configuration = Configuration(
        access_token=channel_access_token
    )
    
    # 創建 API 客戶端
    api_client = ApiClient(configuration)
    
    # 返回 MessagingApi 實例
    return MessagingApi(api_client) 