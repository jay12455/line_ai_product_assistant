from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging
import torch
from database.db_handler import DatabaseHandler
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import pickle

class SentimentAnalyzer:
    def __init__(self):
        try:
            # 載入 OpenAI API Key
            load_dotenv()
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # 使用正確的中文情感分析模型
            self.model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
            self.db = DatabaseHandler()
            logging.info("情感分析模型載入成功")
            
        except Exception as e:
            logging.error(f"初始化情感分析模型時發生錯誤: {str(e)}")
            raise

    def get_embedding(self, text):
        """獲取文本的向量嵌入"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"生成向量嵌入時發生錯誤: {str(e)}")
            return None

    def analyze_sentiment(self, text, user_id):
        """分析文本情感并返回结果，但不写入数据库"""
        try:
            # 1. 使用 Hugging Face 进行初步情感分析
            hf_result = self.sentiment_pipeline(text)[0]
            
            # 计算基础情感分数
            raw_score = hf_result['score']
            if hf_result['label'] == "positive":
                base_sentiment_score = raw_score
            else:
                base_sentiment_score = 1 - raw_score

            # 2. 使用 OpenAI 进行深入情感分析
            try:
                openai_response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": """你是一個情感分析專家。請分析用戶訊息的情感，
                         並返回以下 JSON 格式：
                         {
                             "sentiment_score": float (0-1),
                             "sentiment_label": "非常滿意|滿意|中性|不滿意|非常不滿意",
                             "emotion_analysis": "詳細的情緒分析",
                             "response_suggestion": "建議的回應方式"
                         }"""},
                        {"role": "user", "content": f"基礎情感分數: {base_sentiment_score}\n用戶訊息: {text}"}
                    ],
                    response_format={ "type": "json_object" }
                )
                ai_analysis = eval(openai_response.choices[0].message.content)
            except Exception as e:
                logging.warning(f"OpenAI API 調用失敗，使用基礎情感分析結果: {str(e)}")
                ai_analysis = {
                    'sentiment_score': base_sentiment_score,
                    'sentiment_label': "滿意" if base_sentiment_score > 0.6 else "不滿意" if base_sentiment_score < 0.4 else "中性",
                    'emotion_analysis': "無法進行深入情緒分析",
                    'response_suggestion': "請以一般方式回應"
                }

            # 3. 生成向量嵌入
            try:
                embedding = self.get_embedding(text)
            except Exception as e:
                logging.warning(f"生成向量嵌入失敗: {str(e)}")
                embedding = None
            
            # 4. 结合两种分析结果
            final_score = (base_sentiment_score + ai_analysis['sentiment_score']) / 2
            
            # 5. 根据最终分数确定情感标签
            if final_score >= 0.8:
                final_label = "非常滿意"
            elif final_score >= 0.6:
                final_label = "滿意"
            elif final_score >= 0.45:
                final_label = "中性"
            elif final_score >= 0.3:
                final_label = "不滿意"
            else:
                final_label = "非常不滿意"

            # 6. 写入数据库
            try:
                # 首先写入聊天记录
                sql_chat = """
                    INSERT INTO chat_history 
                    (line_user_id, message_text, sentiment_score, sentiment_label) 
                    VALUES (%s, %s, %s, %s)
                """
                self.db.execute_update(sql_chat, (
                    user_id,
                    text,
                    round(final_score, 3),
                    final_label
                ))
                
                # 获取新插入记录的ID
                chat_id = self.db.execute_query("SELECT LAST_INSERT_ID() as id")[0]['id']
                
                # 如果有向量嵌入，写入chat_embeddings表
                if embedding is not None:
                    sql_embedding = """
                        INSERT INTO chat_embeddings 
                        (chat_id, embedding) 
                        VALUES (%s, %s)
                    """
                    self.db.execute_update(sql_embedding, (
                        chat_id,
                        pickle.dumps(embedding)
                    ))
                    
            except Exception as e:
                logging.error(f"寫入聊天記錄時發生錯誤: {str(e)}")
                # 不中斷程序，繼續返回分析結果

            return {
                'score': round(final_score, 3),
                'label': final_label,
                'emotion_analysis': ai_analysis['emotion_analysis'],
                'response_suggestion': ai_analysis['response_suggestion'],
                'embedding': embedding
            }

        except Exception as e:
            logging.error(f"情感分析發生錯誤: {str(e)}")
            return {
                'score': 0.5,
                'label': '中性',
                'emotion_analysis': '無法進行情緒分析',
                'response_suggestion': '請以一般方式回應',
                'embedding': None
            }

    def analyze_sentiment_only(self, text):
        """只進行情感分析，不寫入資料庫"""
        try:
            # 1. 使用 Hugging Face 進行初步情感分析
            hf_result = self.sentiment_pipeline(text)[0]
            
            # 計算基礎情感分數
            raw_score = hf_result['score']
            if hf_result['label'] == "positive":
                base_sentiment_score = raw_score
            else:
                base_sentiment_score = 1 - raw_score

            # 2. 嘗試使用 OpenAI 進行深入情感分析
            try:
                openai_response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": """你是一個情感分析專家。請分析用戶訊息的情感，
                         並返回以下 JSON 格式：
                         {
                             "sentiment_score": float (0-1),
                             "sentiment_label": "非常滿意|滿意|中性|不滿意|非常不滿意",
                             "emotion_analysis": "詳細的情緒分析",
                             "response_suggestion": "建議的回應方式"
                         }"""},
                        {"role": "user", "content": f"基礎情感分數: {base_sentiment_score}\n用戶訊息: {text}"}
                    ],
                    response_format={ "type": "json_object" }
                )
                ai_analysis = eval(openai_response.choices[0].message.content)
            except Exception as e:
                logging.warning(f"OpenAI API 調用失敗，使用基礎情感分析結果: {str(e)}")
                ai_analysis = {
                    'sentiment_score': base_sentiment_score,
                    'sentiment_label': "滿意" if base_sentiment_score > 0.6 else "不滿意" if base_sentiment_score < 0.4 else "中性",
                    'emotion_analysis': "使用基礎情感分析",
                    'response_suggestion': "請以一般方式回應"
                }

            # 3. 嘗試生成向量嵌入
            try:
                embedding = self.get_embedding(text)
            except Exception as e:
                logging.warning(f"生成向量嵌入失敗: {str(e)}")
                embedding = None
            
            # 4. 結合兩種分析結果
            final_score = (base_sentiment_score + ai_analysis['sentiment_score']) / 2
            
            # 5. 根據最終分數確定情感標籤
            if final_score >= 0.8:
                final_label = "非常滿意"
            elif final_score >= 0.6:
                final_label = "滿意"
            elif final_score >= 0.45:
                final_label = "中性"
            elif final_score >= 0.3:
                final_label = "不滿意"
            else:
                final_label = "非常不滿意"

            return {
                'score': round(final_score, 3),
                'label': final_label,
                'emotion_analysis': ai_analysis['emotion_analysis'],
                'response_suggestion': ai_analysis['response_suggestion'],
                'embedding': embedding
            }

        except Exception as e:
            logging.error(f"情感分析發生錯誤: {str(e)}")
            # 返回中性結果，確保系統可以繼續運行
            return {
                'score': 0.5,
                'label': '中性',
                'emotion_analysis': '無法進行情緒分析',
                'response_suggestion': '請以一般方式回應',
                'embedding': None
            }

    def __del__(self):
        """確保資源正確釋放"""
        try:
            if hasattr(self, 'model'):
                self.model.cpu()
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(self, 'db'):
                del self.db
        except Exception as e:
            logging.error(f"清理情感分析模型資源時發生錯誤: {str(e)}") 
