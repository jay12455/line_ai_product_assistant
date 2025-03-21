from .openai_client import OpenAIClient
import numpy as np
from datetime import datetime
from database.db_handler import DatabaseHandler
import faiss
import pickle
from openai import OpenAI
import logging
import json
import os
from dotenv import load_dotenv
from .sentiment_analyzer import SentimentAnalyzer
import re
from qdrant_client import QdrantClient

class AIAssistant:
    def __init__(self):
        # 加載 .env 文件中的環境變量
        load_dotenv()
        
        # 從環境變量獲取 API 密鑰
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("未設置 OPENAI_API_KEY 環境變量")
            
        self.client = OpenAI(api_key=self.api_key)
        self.db = DatabaseHandler()
        
        # 初始化 FAISS 索引
        self.index = faiss.IndexFlatL2(512)  # 假設嵌入向量的維度是 512
        self.embeddings = []  # 用於存儲嵌入向量的列表
        self.ids = []  # 用於存儲對應的用戶 ID
        self.conversation_history = {}  # 用於存儲對話歷史
        self.sentiment_analyzer = SentimentAnalyzer()
        self.qdrant_client = QdrantClient()

    def get_embedding(self, text):
        """獲取文本的嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"獲取嵌入向量時發生錯誤: {str(e)}")
            return None

    def update_vector_db(self, line_user_id, message_text, sentiment_score, sentiment_label):
        """更新向量資料庫"""
        embedding = self.get_embedding(message_text)  # 獲取嵌入向量
        self.index.add(np.array([embedding]))  # 將嵌入向量添加到 FAISS 索引
        self.embeddings.append(embedding)  # 存儲嵌入向量
        self.ids.append(line_user_id)  # 存儲用戶 ID

        # 將嵌入向量和情感分析結果存儲到 chat_history 表中
        self.db.add_chat_history(line_user_id, message_text, sentiment_score, sentiment_label, embedding)

    def search_similar(self, query_text, k=5):
        """查詢相似的嵌入向量"""
        query_embedding = self.get_embedding(query_text)
        distances, indices = self.index.search(np.array([query_embedding]), k)  # 查詢 k 個最近鄰
        return [(self.ids[i], distances[0][j]) for j, i in enumerate(indices[0])]

    def get_relevant_history(self, user_id, query, limit=5):
        """獲取相關的歷史對話"""
        query_embedding = self.get_embedding(query)
        results = self.search_similar(query)
        return results[:limit]

    def get_product_by_no(self, product_no):
        """根據產品編號獲取產品信息"""
        try:
            # 格式化產品編號為三位數（例如：'20' -> '020'）
            formatted_no = str(product_no).zfill(3)
            logging.info(f"正在查詢產品編號: {formatted_no}")
            
            sql = """
                SELECT product_no, product_name, price_original, product_url, product_description 
                FROM product_details 
                WHERE product_no = %s AND is_active = TRUE
                LIMIT 1
            """
            result = self.db.execute_query(sql, (formatted_no,))
            
            if result:
                product = result[0]
                logging.info(f"找到產品: {product['product_name']}")
                
                # 處理可能為 None 的欄位
                product_name = product['product_name'] or '未提供名稱'
                price = product['price_original'] or '請私訊詢問'
                product_url = product['product_url'] or '#'
                product_description = product['product_description'] or '暫無描述'
                
                # 使用處理過的值構建回應
                response = f"""📱 產品資訊 #{formatted_no}
💠 產品名稱：{product_name}
💰 價格：{price}
🔗 商品連結：{product_url}
📝 產品描述：{product_description}"""
                
                logging.info(f"成功生成產品資訊回應: {response[:100]}...")
                return response
            else:
                logging.warning(f"找不到產品編號 {formatted_no}")
                return f"抱歉，找不到編號為 {formatted_no} 的產品。"
            
        except Exception as e:
            logging.error(f"查詢產品信息時發生錯誤: {str(e)}")
            return "抱歉，查詢產品信息時發生錯誤。請稍後再試。"

    def search_products_by_context(self, user_message):
        """根據用戶消息上下文搜索相關產品"""
        try:
            # 生成消息的向量嵌入
            message_embedding = self.get_embedding(user_message)
            if not message_embedding:
                logging.error("無法生成消息的向量嵌入")
                return []
            
            # 使用 Qdrant 搜索相似產品
            search_result = self.qdrant_client.search(
                collection_name="product_embeddings",
                query_vector=message_embedding,
                limit=5,
                with_payload=True
            )
            
            if not search_result:
                logging.info("未找到相似產品")
                return []
            
            # 提取產品信息
            products = []
            for hit in search_result:
                product_no = hit.payload.get('product_no')
                if not product_no:
                    continue
                    
                # 從 MariaDB 獲取完整的產品信息
                sql = """
                    SELECT product_no, product_name, price_original, product_url, product_description
                    FROM product_details
                    WHERE product_no = %s AND is_active = TRUE
                    LIMIT 1
                """
                result = self.db.execute_query(sql, (product_no,))
                
                if result:
                    product = result[0]
                    product['similarity_score'] = hit.score
                    products.append(product)
            
            # 按相似度排序
            products.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            logging.info(f"找到 {len(products)} 個相似產品")
            return products
            
        except Exception as e:
            logging.error(f"搜索相關產品時發生錯誤: {str(e)}")
            return []

    def search_products_by_keyword(self, query):
        """根據關鍵字搜索產品"""
        try:
            # 清理查詢字串
            original_query = query
            query = query.replace("你們", "").replace("有沒有", "").replace("有賣", "")
            query = query.replace("嗎", "").replace("?", "").replace("？", "").strip()
            
            # 提取核心關鍵字
            core_keyword = query.replace("有", "").replace("賣", "").replace("想買", "").strip()
            
            logging.info(f"原始查詢: {original_query}")
            logging.info(f"清理後查詢: {query}")
            logging.info(f"核心關鍵字: {core_keyword}")
            
            # 簡化的 SQL 查詢，直接搜索產品名稱和描述
            sql = """
                SELECT 
                    p.product_no,
                    p.product_name,
                    p.price_original,
                    p.product_url,
                    p.product_description,
                    CASE 
                        WHEN p.product_name ILIKE %s THEN 100
                        WHEN p.product_name ILIKE %s THEN 80
                        WHEN p.product_description ILIKE %s THEN 60
                        WHEN p.product_description ILIKE %s THEN 40
                        ELSE 0
                    END as relevance
                FROM product_details p
                WHERE p.is_active = TRUE
                AND (
                    p.product_name ILIKE %s
                    OR p.product_name ILIKE %s
                    OR p.product_description ILIKE %s
                    OR p.product_description ILIKE %s
                )
                ORDER BY relevance DESC, product_no
                LIMIT 2
            """
            
            # 準備搜索參數
            exact_match = f"%{core_keyword}%"
            partial_match = f"%{core_keyword}%"
            
            params = [
                exact_match,          # 產品名稱完全匹配
                partial_match,        # 產品名稱部分匹配
                exact_match,          # 產品描述完全匹配
                partial_match,        # 產品描述部分匹配
                exact_match,          # WHERE 條件的匹配
                partial_match,
                exact_match,
                partial_match
            ]
            
            logging.info(f"執行產品搜索 - 關鍵字: {core_keyword}")
            logging.info(f"SQL 查詢: {sql}")
            logging.info(f"查詢參數: {params}")
            
            # 執行查詢
            results = self.db.execute_query(sql, tuple(params))
            
            if not results:
                logging.info("未找到相關產品，嘗試模糊搜索")
                # 如果沒有直接匹配，嘗試更寬鬆的搜索
                fuzzy_sql = """
                    SELECT 
                        p.product_no,
                        p.product_name,
                        p.price_original,
                        p.product_url,
                        p.product_description
                    FROM product_details p
                    WHERE p.is_active = TRUE
                    AND (
                        p.product_name ILIKE %s
                        OR p.product_description ILIKE %s
                    )
                    LIMIT 2
                """
                # 使用更寬鬆的匹配模式
                fuzzy_params = [f"%{core_keyword}%", f"%{core_keyword}%"]
                results = self.db.execute_query(fuzzy_sql, tuple(fuzzy_params))
            
            if results:
                logging.info(f"找到 {len(results)} 個相關產品")
                for result in results:
                    logging.info(f"產品編號: {result['product_no']}, 名稱: {result['product_name']}")
                return results
            else:
                logging.info("未找到任何相關產品")
                return None
            
        except Exception as e:
            logging.error(f"搜索產品時發生錯誤: {str(e)}")
            return None

    def get_product_details(self, product_no):
        """獲取產品詳細信息"""
        try:
            sql = """
                SELECT 
                    product_no,
                    product_name,
                    price_original,
                    product_url,
                    product_description
                FROM product_details
                WHERE product_no = %s AND is_active = TRUE
                LIMIT 1
            """
            
            result = self.db.execute_query(sql, (product_no,))
            return result[0] if result else None
            
        except Exception as e:
            logging.error(f"獲取產品詳情時發生錯誤: {str(e)}")
            return None

    def format_product_response(self, products):
        """格式化產品推薦回應"""
        if not products:
            return "抱歉，目前沒有找到相關的產品。"
        
        response = "為您推薦以下產品：\n\n"
        for product in products:
            response += f"""📦 商品編號：{product['product_no']}
🏷️ 商品名稱：{product['product_name']}
💰 價格：{product['price_original'] if product['price_original'] else '請私訊詢問'}
🔗 商品連結：{product['product_url']}
📝 商品描述：{product['product_description'][:200]}...

{'─' * 30}\n"""
        
        return response

    def get_response(self, user_id, message):
        try:
            # 檢查是否是產品查詢
            query_keywords = ["有沒有", "有什麼", "推薦", "介紹", "有賣", "有", "想買", "賣", "找"]
            
            # 檢查是否是產品查詢
            if any(keyword in message for keyword in query_keywords):
                logging.info(f"檢測到產品查詢請求: {message}")
                
                # 優先使用向量搜索
                context_products = self.search_products_by_context(message)
                if context_products:
                    response = self.format_product_response(context_products)
                    logging.info(f"生成向量搜索推薦回應: {response[:100]}...")
                    return response
                
                # 如果向量搜索沒有結果，嘗試關鍵字搜索
                products = self.search_products_by_keyword(message)
                if products:
                    response = self.format_product_response(products)
                    logging.info(f"生成關鍵字搜索推薦回應: {response[:100]}...")
                    return response
                
                return "抱歉，目前沒有找到相關的產品。您可以試試：\n1. 使用不同的關鍵字\n2. 描述您想要的產品特點\n3. 告訴我產品的用途"
            
            # 檢查是否是產品編號查詢（支援多種格式）
            product_no_match = re.search(r'(?:no\.?|編號|商品編號)?(\d+)', message.lower())
            
            if product_no_match:
                # 提取數字
                product_no = product_no_match.group(1)
                logging.info(f"檢測到產品編號查詢: {product_no}")
                
                # 格式化產品編號為三位數
                formatted_no = str(product_no).zfill(3)
                logging.info(f"正在查詢產品編號: {formatted_no}")
                
                # 直接查詢產品信息
                sql = """
                    SELECT product_no, product_name, price_original, product_url, product_description 
                    FROM product_details 
                    WHERE product_no = %s AND is_active = TRUE
                    LIMIT 1
                """
                result = self.db.execute_query(sql, (formatted_no,))
                
                if result:
                    product = result[0]
                    logging.info(f"找到產品: {product['product_name']}")
                    
                    # 構建回應
                    response = f"""📱 產品資訊 #{formatted_no}
💠 產品名稱：{product['product_name']}
💰 價格：{product['price_original'] if product['price_original'] else '請私訊詢問'}
🔗 商品連結：{product['product_url']}
📝 產品描述：{product['product_description']}"""
                    
                    logging.info(f"成功生成產品資訊回應")
                    return response
                else:
                    return f"抱歉，找不到編號為 {formatted_no} 的產品。"
            
            # 檢查是否需要產品推薦
            recommend_keywords = ["推薦", "介紹", "建議", "有什麼", "推薦一下"]
            need_recommendation = any(keyword in message for keyword in recommend_keywords)
            
            if need_recommendation:
                # 搜索相關產品
                products = self.search_products_by_context(message)
                
                if products:
                    # 使用 OpenAI 生成推薦回應
                    prompt = {
                        "role": "system",
                        "content": """你是一個專業的產品推薦專家。請根據用戶的需求和提供的產品列表，生成合適的推薦內容。
                        注意：
                        1. 根據用戶的具體需求推薦產品
                        2. 突出每個產品的特點
                        3. 包含價格信息
                        4. 使用繁體中文
                        5. 保持專業且親切的語氣"""
                    }
                    
                    context = {
                        "user_message": message,
                        "products": [
                            {
                                "product_no": p['product_no'],
                                "name": p['product_name'],
                                "price": p['price_original'],
                                "description": p['product_description']
                            }
                            for p in products
                        ]
                    }
                    
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            prompt,
                            {"role": "user", "content": str(context)}
                        ]
                    )
                    
                    return response.choices[0].message.content
            
            # 一般對話處理
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = []
            
            # 分析用戶訊息的情感
            sentiment_result = self.sentiment_analyzer.analyze_sentiment_only(message)
            
            # 添加用戶的新消息到歷史記錄
            self.conversation_history[user_id].append({
                "role": "user",
                "content": message
            })
            
            # 準備系統提示詞
            system_prompt = {
                "role": "system",
                "content": f"""你是一個專業的客服助手。
                當前用戶的情感狀態：
                - 情感分數：{sentiment_result['score']}
                - 情感標籤：{sentiment_result['label']}
                
                請根據用戶的情感狀態提供適當的回應。
                注意：
                - 使用繁體中文
                - 保持專業、友善的態度
                - 如果用戶提到產品相關的問題，可以建議他們使用產品編號查詢或請求推薦"""
            }
            
            # 準備完整的對話歷史
            messages = [system_prompt] + self.conversation_history[user_id][-5:]
            
            # 調用 OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            # 獲取 AI 的回應
            ai_response = response.choices[0].message.content
            
            # 將 AI 的回應添加到對話歷史
            self.conversation_history[user_id].append({
                "role": "assistant",
                "content": ai_response
            })
            
            return ai_response
            
        except Exception as e:
            logging.error(f"生成回應時發生錯誤: {str(e)}")
            return "抱歉，我現在無法正確處理您的請求。請稍後再試。"

    def _format_history(self, history):
        """格式化歷史對話"""
        formatted = []
        for i, (doc, metadata) in enumerate(zip(history['documents'], history['metadatas'])):
            sentiment = float(metadata['sentiment_score'])
            sentiment_text = "正面" if sentiment > 0 else "負面" if sentiment < 0 else "中性"
            formatted.append(f"對話 {i+1}:\n內容: {doc}\n情感: {sentiment_text}\n")
        return "\n".join(formatted) 