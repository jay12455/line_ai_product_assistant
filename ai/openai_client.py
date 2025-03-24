import os
from openai import OpenAI

class OpenAIClient:
    _instance = None

    @staticmethod
    def get_instance():
        if OpenAIClient._instance is None:
            api_key = os.getenv("OPENAI_API_KEY")  # 從環境變數讀取 API 金鑰
            if not api_key:
                raise ValueError("未設定 OPENAI_API_KEY 環境變數")
            OpenAIClient._instance = OpenAI(api_key=api_key)
        return OpenAIClient._instance
