from database.db_handler import DatabaseHandler
import logging
from datetime import datetime, timedelta
from collections import Counter
from openai import OpenAI
from ai.sentiment_analyzer import SentimentAnalyzer
import os
from dotenv import load_dotenv
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

class AIRecommender:
    def __init__(self):
        self.db = DatabaseHandler()
        self.sentiment_analyzer = SentimentAnalyzer()
        load_dotenv()
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # 使用與 sentiment_analyzer 相同的模型
        self.model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # 產品關鍵字類別
        self.keyword_categories = {
            "食物": {
                "複合產品": [
                    "蘋果麵包", "起司麵包", "巧克力餅乾", "水果茶", 
                    "牛肉水餃", "蝦仁水餃", "冷凍蝦仁", "起司餅乾",
                    "冷凍水餃", "百事可樂"
                ],
                "單一產品": [
                    "麵包", "餅乾", "起司", "水餃", "可樂", "飲料",
                    "點心", "零食", "水果", "茶葉", "蝦仁"
                ],
                "通用類別": ["食物", "美食", "小吃", "甜點", "冷凍食品"]
            },
            "生活用品": {
                "複合產品": [
                    "抗菌牙刷", "天然沐浴乳", "有機洗髮精",
                    "環保衛生紙", "抗敏毛巾"
                ],
                "單一產品": [
                    "牙刷", "毛巾", "沐浴乳", "洗髮精", "衛生紙"
                ],
                "通用類別": ["日用品", "居家", "生活用品", "清潔用品"]
            },
            "電子產品": {
                "複合產品": [
                    "無線藍芽耳機", "快充行動電源", "機械式鍵盤",
                    "無線滑鼠", "Type-C充電線", "藍芽耳機",
                    "USB充電線", "行動電源", "無線充電器", "智能手環",
                    "藍牙音箱", "無線鍵盤", "電競滑鼠", "機械鍵盤"
                ],
                "單一產品": [
                    "耳機", "電源", "充電線", "滑鼠", "鍵盤",
                    "手機", "平板", "電腦", "音箱", "手環",
                    "充電器", "數據線", "配件", "音響"
                ],
                "通用類別": ["3C", "科技", "電子", "配件", "電器", "數位", "智能"]
            }
        }

        # 類別特定評價詞
        self.category_specific_keywords = {
            "食物": ["好吃", "美味", "可口", "美食", "香", "Q彈", "口感", "新鮮", "美味可口", "香氣"],
            "生活用品": ["好用", "實用", "方便", "舒適", "柔軟", "乾淨", "清潔", "品質好", "品質佳", "品質很好", "很好用", "非常好用"],
            "電子產品": [
                "耐用", "好用", "方便", "智慧", "快速", "穩定", "效能好", "品質佳", "品質好", "品質很好", 
                "容量大", "很耐用", "非常好用", "反應快", "音質好", "音質佳", "音質清晰", "續航力強",
                "充電快", "操作順暢", "靈敏", "精準", "舒適", "便攜", "高效", "省電", "輕巧"
            ]
        }
        
        # 通用評價詞
        self.common_keywords = [
            "喜歡", "推薦", "讚", "回購", "值得", "棒", "超棒", "不錯", "很好", "很讚", 
            "很推薦", "很喜歡", "很棒", "真的很好", "真的不錯", "好評", "優質", "完美", 
            "超讚", "超級棒", "物超所值", "CP值高", "推", "強推"
        ]

    def check_evaluation_word(self, word, text):
        """更靈活地匹配評價詞"""
        logging.info(f"\n檢查評價詞: {word}")
        logging.info(f"評論內容: {text}")
        
        # 基本匹配
        if word in text:
            logging.info(f"✓ 直接匹配成功: {word}")
            return True
        
        # 清理評價詞（移除可能已有的前綴）
        clean_word = word
        prefixes = ["很", "非常", "真的", "超", "特別", "十分", "太", "好"]
        for prefix in prefixes:
            if word.startswith(prefix):
                clean_word = word[len(prefix):]
                break
        
        # 生成所有可能的變體
        variations = [
            clean_word,                    # 基本形式
            f"很{clean_word}",            # 很 + 詞
            f"非常{clean_word}",          # 非常 + 詞
            f"真的{clean_word}",          # 真的 + 詞
            f"超{clean_word}",            # 超 + 詞
            f"特別{clean_word}",          # 特別 + 詞
            f"十分{clean_word}",          # 十分 + 詞
            f"太{clean_word}",            # 太 + 詞
            f"好{clean_word}",            # 好 + 詞
            f"{clean_word}的",            # 詞 + 的
            f"很{clean_word}的",          # 很 + 詞 + 的
            f"非常{clean_word}的",        # 非常 + 詞 + 的
            f"真的很{clean_word}",        # 真的很 + 詞
            f"真的非常{clean_word}",      # 真的非常 + 詞
            f"超級{clean_word}",          # 超級 + 詞
            f"真是{clean_word}"           # 真是 + 詞
        ]
        
        # 檢查所有變體
        for variant in variations:
            if variant in text:
                logging.info(f"✓ 變體匹配成功: {variant}")
                return True
        
        logging.info("✗ 未找到匹配的評價詞變體")
        return False

    def analyze_user_interests(self, user_id, days=7, mention_threshold=3, sentiment_threshold=0.7):
        """分析用戶興趣和情感傾向"""
        try:
            # 獲取用戶的聊天記錄和情感分析
            sql = """
                SELECT DISTINCT
                    id,
                    message_text,
                    sentiment_score,
                    created_at
                FROM chat_history
                WHERE line_user_id = %s
                AND created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                AND message_text != '指令'
                ORDER BY created_at DESC
            """
            
            logging.info(f"\n=== 執行資料庫查詢 ===")
            logging.info(f"SQL查詢: {sql}")
            logging.info(f"用戶ID: {user_id}")
            
            # 確保資料庫連接
            if not self.db.conn.is_connected():
                logging.info("重新連接資料庫...")
                self.db.reconnect()
                logging.info("資料庫重新連接成功")
            
            # 執行查詢
            self.db.cursor.execute(sql, (user_id,))
            chat_records = self.db.cursor.fetchall()
            
            # 添加日志显示查询到的记录
            logging.info(f"\n=== 查詢結果 ===")
            logging.info(f"總共找到 {len(chat_records)} 條評論")
            for record in chat_records:
                logging.info(f"\n評論詳情:")
                logging.info(f"ID: {record['id']}")
                logging.info(f"內容: {record['message_text']}")
                logging.info(f"情感分數: {record['sentiment_score']}")
                logging.info(f"時間: {record['created_at']}")
            
            # 分析每個類別的關鍵字和情感
            category_analysis = {
                "食物": {'mentions': 0, 'sentiment_scores': [], 'embeddings': [], 'messages': []},
                "生活用品": {'mentions': 0, 'sentiment_scores': [], 'embeddings': [], 'messages': []},
                "電子產品": {'mentions': 0, 'sentiment_scores': [], 'embeddings': [], 'messages': []}
            }
            
            for record in chat_records:
                message = record['message_text']  # 不轉換大小寫
                sentiment_score = float(record['sentiment_score'])
                embedding = record['embedding']
                
                # 添加详细日志
                logging.info(f"\n=== 处理评论 ===")
                logging.info(f"评论内容: {message}")
                logging.info(f"情感分数: {sentiment_score}")
                
                # 检查每个类别
                for category in category_analysis.keys():
                    # 分别检查复合产品和单一产品
                    complex_products = self.keyword_categories[category]["複合產品"]
                    single_products = self.keyword_categories[category]["單一產品"]
                    
                    # 先检查复合产品
                    matched_complex = [kw for kw in complex_products if kw in message]
                    
                    # 如果没有匹配到复合产品，再检查单一产品
                    matched_single = [kw for kw in single_products if kw in message]
                    
                    # 记录匹配结果
                    matched_products = matched_complex + matched_single
                    
                    logging.info(f"\n类别: {category}")
                    logging.info(f"匹配到的复合产品: {matched_complex}")
                    logging.info(f"匹配到的单一产品: {matched_single}")
                    
                    # 检查评价词
                    specific_words = self.category_specific_keywords[category]
                    common_words = self.common_keywords
                    
                    # 检查评价词
                    matched_specific = []
                    matched_common = []
                    
                    # 检查特定评价词
                    for word in specific_words:
                        if self.check_evaluation_word(word, message):
                            matched_specific.append(word)
                    
                    # 检查通用评价词
                    for word in common_words:
                        if self.check_evaluation_word(word, message):
                            matched_common.append(word)
                    
                    logging.info(f"匹配到的特定评价词: {matched_specific}")
                    logging.info(f"匹配到的通用评价词: {matched_common}")
                    
                    # 如果同时匹配到产品和评价词，将整条评论作为一个匹配单位
                    if matched_products and (matched_specific or matched_common):
                        matched_comment = {
                            'id': record['id'],
                            'message': message,
                            'sentiment_score': sentiment_score,
                            'products': matched_products,
                            'evaluation_words': matched_specific + matched_common,
                            'created_at': record['created_at']
                        }
                        category_analysis[category]['mentions'] += 1
                        category_analysis[category]['sentiment_scores'].append(sentiment_score)
                        if embedding:
                            category_analysis[category]['embeddings'].append(embedding)
                        category_analysis[category]['messages'].append(message)
                        logging.info(f"✓ 评论匹配成功")
                        logging.info(f"✓ 产品: {matched_products}")
                        logging.info(f"✓ 评价词: {matched_specific + matched_common}")
                        logging.info(f"✓ 情感分数: {sentiment_score}")
                        logging.info(f"✓ 当前匹配评论数: {category_analysis[category]['mentions']}")
            
            # 準備 OpenAI 分析的內容
            categories_info = []
            for category, data in category_analysis.items():
                if data['mentions'] >= mention_threshold:
                    avg_sentiment = np.mean(data['sentiment_scores']) if data['sentiment_scores'] else 0
                    if avg_sentiment >= sentiment_threshold:
                        categories_info.append({
                            'category': category,
                            'mentions': data['mentions'],
                            'avg_sentiment': round(float(avg_sentiment), 3),
                            'messages': data['messages'][-5:]  # 最近5條相關消息
                        })
            
            if not categories_info:
                return []
            
            # 使用 OpenAI 分析用戶偏好
            analysis_prompt = {
                "role": "system",
                "content": """分析用戶的對話記錄和情感數據，找出：
                1. 用戶對各類產品的具體興趣點
                2. 最適合的推薦時機
                3. 用戶可能感興趣的具體商品特徵
                
                請以 JSON 格式返回：
                {
                    "interests": [
                        {
                            "category": "類別",
                            "specific_interests": ["具體興趣點"],
                            "best_timing": "最佳推薦時機",
                            "preferred_features": ["偏好的商品特徵"]
                        }
                    ]
                }"""
            }
            
            # 將分析數據轉換為 OpenAI 可處理的格式
            analysis_content = str({
                'categories': categories_info,
                'sentiment_threshold': sentiment_threshold,
                'mention_threshold': mention_threshold
            })
            
            ai_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    analysis_prompt,
                    {"role": "user", "content": analysis_content}
                ],
                response_format={ "type": "json_object" }
            )
            
            ai_analysis = eval(ai_response.choices[0].message.content)
            
            # 整合所有分析結果
            final_recommendations = []
            for interest in ai_analysis['interests']:
                category_data = next(
                    (data for data in categories_info 
                     if data['category'] == interest['category']), 
                    None
                )
                
                if category_data:
                    final_recommendations.append({
                        'category': interest['category'],
                        'mentions': category_data['mentions'],
                        'avg_sentiment': category_data['avg_sentiment'],
                        'specific_interests': interest['specific_interests'],
                        'best_timing': interest['best_timing'],
                        'preferred_features': interest['preferred_features']
                    })
            
            return final_recommendations
            
        except Exception as e:
            logging.error(f"分析用戶興趣時發生錯誤: {str(e)}")
            return []

    def generate_recommendation(self, user_id, current_promotions):
        """生成個人化推薦"""
        try:
            interests = self.analyze_user_interests(user_id)
            if not interests:
                return None
            
            recommendations = []
            for interest in interests:
                # 只匹配相同類別的促銷活動
                matching_promos = [
                    promo for promo in current_promotions
                    if promo['category'].lower() == interest['category'].lower()
                ]
                
                if matching_promos:
                    # 使用 OpenAI 生成個性化推薦訊息
                    promo_context = {
                        'interest': interest,
                        'promotions': matching_promos
                    }
                    
                    # 添加日誌
                    logging.info(f"正在為用戶 {user_id} 生成 {interest['category']} 類別的推薦")
                    logging.info(f"找到 {len(matching_promos)} 個匹配的促銷活動")
                    
                    response = self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": """
                            根據用戶的興趣和情感分析生成個性化推薦。
                            注意：
                            1. 使用用戶偏好的特徵來突出商品優勢
                            2. 選擇最佳的推薦時機
                            3. 保持友善和個性化的語氣
                            4. 確保推薦的活動與用戶評論的產品類別相符
                            """},
                            {"role": "user", "content": str(promo_context)}
                        ]
                    )
                    
                    recommendations.append({
                        'message': response.choices[0].message.content,
                        'priority': interest['mentions'] * interest['avg_sentiment'],
                        'category': interest['category'],
                        'promotions': matching_promos
                    })
            
            # 根據優先級排序推薦
            recommendations.sort(key=lambda x: x['priority'], reverse=True)
            
            # 記錄推薦歷史
            if recommendations:
                self._log_recommendation(user_id, recommendations[0])
                logging.info(f"最終選擇推薦類別: {recommendations[0]['category']}")
            
            return recommendations[0] if recommendations else None
            
        except Exception as e:
            logging.error(f"生成推薦時發生錯誤: {str(e)}")
            return None

    def _log_recommendation(self, user_id, recommendation):
        """記錄推薦歷史"""
        try:
            sql = """
                INSERT INTO recommendation_history 
                (line_user_id, category, product_no, recommendation_content, is_clicked, is_purchased)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.db.execute_update(sql, (
                user_id,
                recommendation['category'],
                recommendation['product_no'],
                str(recommendation['recommendation']),
                False,  # 初始狀態設為未點擊
                False   # 初始狀態設為未購買
            ))
            
            # 獲取插入的記錄ID
            cursor = self.db.get_cursor()
            recommendation_id = cursor.lastrowid
            logging.info(f"已記錄推薦歷史，ID: {recommendation_id}, 用戶: {user_id}")
            
            return recommendation_id
            
        except Exception as e:
            logging.error(f"記錄推薦歷史時發生錯誤: {str(e)}")
            return None
        finally:
            self.db.close_cursor()

    def __del__(self):
        """確保資源正確釋放"""
        try:
            if hasattr(self, 'model'):
                self.model.cpu()
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if hasattr(self, 'db'):
                del self.db
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"清理資源時發生錯誤: {str(e)}")

    def get_active_promotions(self, category=None):
        """獲取當前有效的活動"""
        try:
            if category:
                sql = """
                    SELECT * FROM promotion_events 
                    WHERE is_active = TRUE 
                    AND start_date <= NOW() 
                    AND end_date >= NOW()
                    AND category = %s
                    ORDER BY created_at DESC
                """
                return self.db.execute_query(sql, (category,))
            else:
                sql = """
                    SELECT * FROM promotion_events 
                    WHERE is_active = TRUE 
                    AND start_date <= NOW() 
                    AND end_date >= NOW()
                    ORDER BY created_at DESC
                """
                return self.db.execute_query(sql)
        except Exception as e:
            logging.error(f"獲取活動資訊時發生錯誤: {str(e)}")
            return []

    def check_user_interests(self, user_id, category, threshold=0.7, min_mentions=3):
        """檢查用戶對特定類別的興趣程度"""
        try:
            # 使用简化的查询获取评论
            sql = """
                SELECT 
                    id,
                    message_text,
                    sentiment_score,
                    created_at
                FROM chat_history 
                WHERE line_user_id = %s 
                AND message_text != '指令'
                ORDER BY created_at DESC
            """
            
            logging.info(f"\n=== 執行資料庫查詢 ===")
            logging.info(f"SQL查詢: {sql}")
            logging.info(f"用戶ID: {user_id}")
            
            # 確保資料庫連接
            if not self.db.conn.is_connected():
                logging.info("重新連接資料庫...")
                self.db.reconnect()
                logging.info("資料庫重新連接成功")
            
            # 執行查詢
            self.db.cursor.execute(sql, (user_id,))
            comments = self.db.cursor.fetchall()
            
            logging.info(f"\n=== 查詢結果 ===")
            logging.info(f"總共找到 {len(comments)} 條評論")
            
            # 顯示所有評論
            for comment in comments:
                logging.info(f"\n評論詳情:")
                logging.info(f"ID: {comment['id']}")
                logging.info(f"內容: {comment['message_text']}")
                logging.info(f"情感分數: {comment['sentiment_score']}")
                logging.info(f"時間: {comment['created_at']}")
            
            matched_comments = []
            total_sentiment = 0
            
            for comment in comments:
                message = comment['message_text'].lower()
                
                # 检查是否包含产品关键字
                has_product = any(product in message for product in self.keyword_categories[category]["複合產品"] + self.keyword_categories[category]["單一產品"])
                
                # 检查是否包含评价词
                has_specific = any(word in message for word in self.category_specific_keywords.get(category, []))
                has_common = any(word in message for word in self.common_keywords)
                
                if has_product and (has_specific or has_common):
                    matched_comments.append(comment)
                    total_sentiment += float(comment['sentiment_score'])
                    logging.info(f"\n匹配到评论 - ID: {comment['id']}")
                    logging.info(f"内容: {comment['message_text']}")
                    logging.info(f"情感分数: {comment['sentiment_score']}")
            
            comment_count = len(matched_comments)
            avg_sentiment = total_sentiment / comment_count if comment_count > 0 else 0
            
            logging.info(f"\n=== 统计结果 ===")
            logging.info(f"匹配的评论数: {comment_count}")
            logging.info(f"平均情感分数: {avg_sentiment}")
            
            # 检查是否满足推荐条件
            should_recommend = comment_count >= min_mentions and avg_sentiment >= threshold
            
            if should_recommend:
                logging.info("✓ 满足推荐条件")
                trigger_reason = "近期多次正面評論"
            else:
                logging.info("✗ 不满足推荐条件")
                if comment_count < min_mentions:
                    logging.info(f"评论数不足: {comment_count} < {min_mentions}")
                if avg_sentiment < threshold:
                    logging.info(f"平均情感分数不足: {avg_sentiment} < {threshold}")
                trigger_reason = None
            
            return should_recommend, avg_sentiment, trigger_reason
            
        except Exception as e:
            logging.error(f"检查用户兴趣时发生错误: {str(e)}")
            return False, 0, None

    def check_recommendation_cooldown(self, user_id, category):
        """檢查推薦冷卻時間"""
        try:
            # 檢查是否在冷卻時間內
            sql = """
                SELECT rc.last_recommendation_time, rc.product_no,
                       COUNT(pd.id) as new_products
                FROM recommendation_cooldown rc
                LEFT JOIN product_details pd ON 
                    pd.category = rc.category AND 
                    pd.created_at > rc.last_recommendation_time AND
                    pd.is_active = TRUE
                WHERE rc.line_user_id = %s 
                AND rc.category = %s
                AND rc.last_recommendation_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY rc.last_recommendation_time, rc.product_no
            """
            
            self.db.cursor.execute(sql, (user_id, category))
            result = self.db.cursor.fetchone()
            
            if result:
                # 如果有新商品，允許推薦
                if result['new_products'] > 0:
                    logging.info(f"發現新商品，允許推薦 - 用戶: {user_id}, 類別: {category}")
                    return True
                
                # 如果在冷卻時間內且沒有新商品，不允許推薦
                logging.info(f"在冷卻時間內 - 用戶: {user_id}, 類別: {category}")
                return False
            
            # 如果沒有推薦記錄，允許推薦
            logging.info(f"無推薦記錄，允許推薦 - 用戶: {user_id}, 類別: {category}")
            return True
            
        except Exception as e:
            logging.error(f"檢查推薦冷卻時間時發生錯誤: {str(e)}")
            return True

    def update_recommendation_cooldown(self, user_id, category, product_no):
        """更新推薦冷卻時間"""
        try:
            sql = """
                INSERT INTO recommendation_cooldown 
                (line_user_id, category, last_recommendation_time, product_no)
                VALUES (%s, %s, NOW(), %s)
                ON DUPLICATE KEY UPDATE 
                    last_recommendation_time = NOW(),
                    product_no = %s
            """
            self.db.cursor.execute(sql, (user_id, category, product_no, product_no))
            self.db.conn.commit()
            logging.info(f"更新推薦冷卻時間 - 用戶: {user_id}, 類別: {category}, 商品編號: {product_no}")
            return True
        except Exception as e:
            logging.error(f"更新推薦冷卻時間時發生錯誤: {str(e)}")
            self.db.conn.rollback()
            return False

    def generate_promotion_recommendation(self, user_id, category):
        """生成商品推薦"""
        try:
            # 檢查冷卻時間
            if not self.check_recommendation_cooldown(user_id, category):
                logging.info(f"推薦處於冷卻時間 - 用戶: {user_id}, 類別: {category}")
                return None
            
            # 獲取相關商品
            sql = """
                SELECT *
                FROM product_details
                WHERE category = %s
                AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 5
            """
            
            self.db.cursor.execute(sql, (category,))
            active_products = self.db.cursor.fetchall()
            
            if not active_products:
                logging.info(f"沒有找到商品 - 類別: {category}")
                return None
            
            # 使用 OpenAI 生成推薦訊息
            prompt = {
                "role": "system",
                "content": """根據用戶的興趣和商品資訊生成個性化推薦。
                注意：
                1. 強調商品的特色和優惠價格
                2. 使用友善且個性化的語氣
                3. 清楚說明商品特點
                4. 提供具體的價格資訊
                請用繁體中文回覆。
                """
            }
            
            context = {
                "category": category,
                "products": active_products
            }
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    prompt,
                    {"role": "user", "content": str(context)}
                ]
            )
            
            recommendation = response.choices[0].message.content
            
            # 更新冷卻時間
            self.update_recommendation_cooldown(user_id, category, active_products[0]['product_no'])
            
            # 記錄推薦歷史
            self._log_recommendation(user_id, {
                'category': category,
                'product_no': active_products[0]['product_no'],
                'recommendation': recommendation
            })
            
            logging.info(f"成功生成推薦 - 用戶: {user_id}, 類別: {category}")
            return recommendation
            
        except Exception as e:
            logging.error(f"生成商品推薦時發生錯誤: {str(e)}")
            return None
