import sqlite3
import hashlib
import os
from datetime import datetime
import json
from typing import Optional, List, Dict, Any

class Database:
    def __init__(self, db_path: str = "ml_assistant.db"):
        """
        Initialize database connection and create tables if they don't exist.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database tables."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table for authentication
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Quiz results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quiz_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                quiz_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_questions INTEGER NOT NULL,
                correct_answers INTEGER NOT NULL,
                score_percentage REAL NOT NULL,
                time_taken_seconds INTEGER NOT NULL,
                difficulty_breakdown TEXT, -- JSON string with difficulty-wise scores
                keywords_attempted TEXT, -- JSON string with keywords from questions
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Chat history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                message_type TEXT NOT NULL, -- 'user' or 'assistant'
                message_content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # User preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                preference_key TEXT NOT NULL,
                preference_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, preference_key)
            )
        """)
        
        # Quiz performance data table for ML prediction
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quiz_performance_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                score_percentage REAL NOT NULL,
                time_taken_seconds INTEGER NOT NULL,
                skill_level TEXT NOT NULL, -- 'beginner', 'intermediate', 'advanced', 'master', 'professional'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, email: str, password: str) -> bool:
        """
        Create a new user account.
        
        Args:
            email (str): User email
            password (str): User password
            
        Returns:
            bool: True if user created successfully, False otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (email, password_hash)
            )
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.IntegrityError:
            # Email already exists
            return False
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user login.
        
        Args:
            email (str): User email
            password (str): User password
            
        Returns:
            Optional[Dict]: User data if authentication successful, None otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute(
                "SELECT id, email, created_at FROM users WHERE email = ? AND password_hash = ? AND is_active = 1",
                (email, password_hash)
            )
            
            user_data = cursor.fetchone()
            
            if user_data:
                # Update last login
                cursor.execute(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                    (user_data[0],)
                )
                conn.commit()
                
                return {
                    'id': user_data[0],
                    'email': user_data[1],
                    'created_at': user_data[2]
                }
            
            conn.close()
            return None
            
        except Exception as e:
            print(f"Error authenticating user: {e}")
            return None
    
    def save_quiz_result(self, user_id: int, total_questions: int, correct_answers: int, 
                        time_taken: int, difficulty_breakdown: Dict, keywords: List[str]) -> bool:
        """
        Save quiz result to database.
        
        Args:
            user_id (int): User ID
            total_questions (int): Total number of questions
            correct_answers (int): Number of correct answers
            time_taken (int): Time taken in seconds
            difficulty_breakdown (Dict): Breakdown by difficulty level
            keywords (List[str]): Keywords from attempted questions
            
        Returns:
            bool: True if saved successfully
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            score_percentage = (correct_answers / total_questions) * 100
            
            cursor.execute("""
                INSERT INTO quiz_results 
                (user_id, total_questions, correct_answers, score_percentage, 
                 time_taken_seconds, difficulty_breakdown, keywords_attempted)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, total_questions, correct_answers, score_percentage,
                time_taken, json.dumps(difficulty_breakdown), json.dumps(keywords)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving quiz result: {e}")
            return False
    
    def get_user_quiz_history(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get quiz history for a user.
        
        Args:
            user_id (int): User ID
            
        Returns:
            List[Dict]: List of quiz results
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT quiz_date, total_questions, correct_answers, score_percentage,
                       time_taken_seconds, difficulty_breakdown, keywords_attempted
                FROM quiz_results 
                WHERE user_id = ?
                ORDER BY quiz_date DESC
            """, (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            quiz_history = []
            for result in results:
                quiz_history.append({
                    'date': result[0],
                    'total_questions': result[1],
                    'correct_answers': result[2],
                    'score_percentage': result[3],
                    'time_taken': result[4],
                    'difficulty_breakdown': json.loads(result[5]) if result[5] else {},
                    'keywords': json.loads(result[6]) if result[6] else []
                })
            
            return quiz_history
            
        except Exception as e:
            print(f"Error getting quiz history: {e}")
            return []
    
    def save_chat_message(self, user_id: int, message_type: str, content: str, session_id: str = None) -> bool:
        """
        Save chat message to database.
        
        Args:
            user_id (int): User ID
            message_type (str): 'user' or 'assistant'
            content (str): Message content
            session_id (str): Optional session ID
            
        Returns:
            bool: True if saved successfully
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO chat_history (user_id, message_type, message_content, session_id)
                VALUES (?, ?, ?, ?)
            """, (user_id, message_type, content, session_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving chat message: {e}")
            return False
    
    def get_user_chat_history(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get chat history for a user.
        
        Args:
            user_id (int): User ID
            limit (int): Maximum number of messages to retrieve
            
        Returns:
            List[Dict]: List of chat messages
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT message_type, message_content, timestamp, session_id
                FROM chat_history 
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            chat_history = []
            for result in results:
                chat_history.append({
                    'type': result[0],
                    'content': result[1],
                    'timestamp': result[2],
                    'session_id': result[3]
                })
            
            return list(reversed(chat_history))  # Return in chronological order
            
        except Exception as e:
            print(f"Error getting chat history: {e}")
            return []
    
    def delete_user_chat_history(self, user_id: int, session_id: str = None) -> bool:
        """
        Delete chat history for a user.
        
        Args:
            user_id (int): User ID
            session_id (str): Optional session ID to delete specific session
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute(
                    "DELETE FROM chat_history WHERE user_id = ? AND session_id = ?",
                    (user_id, session_id)
                )
            else:
                cursor.execute(
                    "DELETE FROM chat_history WHERE user_id = ?",
                    (user_id,)
                )
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error deleting chat history: {e}")
            return False
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Get user statistics.
        
        Args:
            user_id (int): User ID
            
        Returns:
            Dict: User statistics
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Quiz stats
            cursor.execute("""
                SELECT COUNT(*) as total_quizzes,
                       AVG(score_percentage) as avg_score,
                       MAX(score_percentage) as best_score,
                       AVG(time_taken_seconds) as avg_time
                FROM quiz_results 
                WHERE user_id = ?
            """, (user_id,))
            
            quiz_stats = cursor.fetchone()
            
            # Chat stats
            cursor.execute("""
                SELECT COUNT(*) as total_messages
                FROM chat_history 
                WHERE user_id = ?
            """, (user_id,))
            
            chat_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_quizzes': quiz_stats[0] or 0,
                'average_score': round(quiz_stats[1] or 0, 2),
                'best_score': quiz_stats[2] or 0,
                'average_time': round(quiz_stats[3] or 0, 2),
                'total_chat_messages': chat_stats[0] or 0
            }
            
        except Exception as e:
            print(f"Error getting user stats: {e}")
            return {}
    
    def seed_performance_data(self) -> bool:
        """
        Seed the quiz_performance_data table with realistic training data.
        100 records each for beginner, intermediate, advanced, master, professional.
        
        Returns:
            bool: True if seeded successfully
        """
        import random
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check if data already exists
            cursor.execute("SELECT COUNT(*) FROM quiz_performance_data")
            if cursor.fetchone()[0] > 0:
                conn.close()
                return True  # Already seeded
            
            # Define realistic ranges for each skill level
            skill_data = {
                'beginner': {
                    'score_range': (20, 50),
                    'time_range': (300, 600)  # 5-10 minutes
                },
                'intermediate': {
                    'score_range': (45, 70),
                    'time_range': (240, 420)  # 4-7 minutes
                },
                'advanced': {
                    'score_range': (65, 85),
                    'time_range': (180, 360)  # 3-6 minutes
                },
                'master': {
                    'score_range': (80, 95),
                    'time_range': (120, 300)  # 2-5 minutes
                },
                'professional': {
                    'score_range': (90, 100),
                    'time_range': (90, 240)   # 1.5-4 minutes
                }
            }
            
            # Generate 100 records for each skill level
            for skill_level, ranges in skill_data.items():
                for _ in range(100):
                    score = round(random.uniform(*ranges['score_range']), 2)
                    time_taken = random.randint(*ranges['time_range'])
                    
                    cursor.execute("""
                        INSERT INTO quiz_performance_data (score_percentage, time_taken_seconds, skill_level)
                        VALUES (?, ?, ?)
                    """, (score, time_taken, skill_level))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error seeding performance data: {e}")
            return False
    
    def predict_skill_level(self, score_percentage: float, time_taken_seconds: int) -> str:
        """
        Predict skill level based on score and time using simple rule-based classification.
        
        Args:
            score_percentage (float): Quiz score percentage
            time_taken_seconds (int): Time taken in seconds
            
        Returns:
            str: Predicted skill level
        """
        try:
            # Simple rule-based classification
            # Higher scores and lower times indicate higher skill levels
            
            if score_percentage >= 90 and time_taken_seconds <= 240:
                return 'professional'
            elif score_percentage >= 80 and time_taken_seconds <= 300:
                return 'master'
            elif score_percentage >= 65 and time_taken_seconds <= 360:
                return 'advanced'
            elif score_percentage >= 45 and time_taken_seconds <= 420:
                return 'intermediate'
            else:
                return 'beginner'
                
        except Exception as e:
            print(f"Error predicting skill level: {e}")
            return 'beginner'
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the performance data.
        
        Returns:
            Dict: Performance statistics
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT skill_level, COUNT(*) as count,
                       AVG(score_percentage) as avg_score,
                       AVG(time_taken_seconds) as avg_time
                FROM quiz_performance_data
                GROUP BY skill_level
                ORDER BY 
                    CASE skill_level
                        WHEN 'beginner' THEN 1
                        WHEN 'intermediate' THEN 2
                        WHEN 'advanced' THEN 3
                        WHEN 'master' THEN 4
                        WHEN 'professional' THEN 5
                    END
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            stats = {}
            for result in results:
                stats[result[0]] = {
                    'count': result[1],
                    'avg_score': round(result[2], 2),
                    'avg_time': round(result[3], 2)
                }
            
            return stats
            
        except Exception as e:
            print(f"Error getting performance stats: {e}")
            return {}
    
    def save_user_skill_prediction(self, user_id: int, predicted_skill: str, 
                                 score_percentage: float, time_taken_seconds: int) -> bool:
        """
        Save user's predicted skill level to preferences.
        
        Args:
            user_id (int): User ID
            predicted_skill (str): Predicted skill level
            score_percentage (float): Quiz score
            time_taken_seconds (int): Time taken
            
        Returns:
            bool: True if saved successfully
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Save prediction data as JSON
            prediction_data = {
                'skill_level': predicted_skill,
                'last_score': score_percentage,
                'last_time': time_taken_seconds,
                'updated_at': datetime.now().isoformat()
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO user_preferences 
                (user_id, preference_key, preference_value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (user_id, 'skill_prediction', json.dumps(prediction_data)))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving skill prediction: {e}")
            return False
    
    def get_user_skill_level(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user's current skill level prediction.
        
        Args:
            user_id (int): User ID
            
        Returns:
            Optional[Dict]: Skill level data or None
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT preference_value FROM user_preferences
                WHERE user_id = ? AND preference_key = 'skill_prediction'
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return json.loads(result[0])
            
            return None
            
        except Exception as e:
            print(f"Error getting user skill level: {e}")
            return None
    
    def get_user_quiz_keywords(self, user_id: int, limit: int = 5) -> List[str]:
        """
        Get unique keywords from user's recent quiz attempts for content recommendations.
        
        Args:
            user_id (int): User ID
            limit (int): Number of recent quizzes to consider
            
        Returns:
            List[str]: List of unique keywords from recent quizzes
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT keywords_attempted
                FROM quiz_results 
                WHERE user_id = ?
                ORDER BY quiz_date DESC
                LIMIT ?
            """, (user_id, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            # Collect all keywords from recent quizzes
            all_keywords = []
            for result in results:
                if result[0]:
                    keywords = json.loads(result[0])
                    if isinstance(keywords, list):
                        all_keywords.extend(keywords)
            
            # Return unique keywords (most frequent first)
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            return [keyword for keyword, count in keyword_counts.most_common(20)]
            
        except Exception as e:
            print(f"Error getting user quiz keywords: {e}")
            return []
