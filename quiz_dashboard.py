import streamlit as st
import time
import random
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from database import Database
from auth import auth_manager
import json

class QuizManager:
    def __init__(self):
        """
        Initialize quiz manager with database connection.
        """
        self.db = Database()
        
        # Quiz questions organized by difficulty level
        self.questions = {
            'basic': [
                {
                    'question': 'What does AI stand for?',
                    'options': ['Artificial Intelligence', 'Automated Intelligence', 'Advanced Intelligence', 'Algorithmic Intelligence'],
                    'correct': 0,
                    'keywords': ['AI', 'artificial intelligence', 'definition']
                },
                {
                    'question': 'Which of the following is a supervised learning algorithm?',
                    'options': ['K-means clustering', 'Linear regression', 'DBSCAN', 'PCA'],
                    'correct': 1,
                    'keywords': ['supervised learning', 'linear regression', 'algorithm']
                },
                {
                    'question': 'What is the primary goal of machine learning?',
                    'options': ['To replace humans', 'To learn patterns from data', 'To create robots', 'To solve math problems'],
                    'correct': 1,
                    'keywords': ['machine learning', 'patterns', 'data', 'goal']
                },
                {
                    'question': 'Which programming language is most commonly used for AI?',
                    'options': ['Java', 'C++', 'Python', 'JavaScript'],
                    'correct': 2,
                    'keywords': ['programming', 'Python', 'AI development']
                }
            ],
            'intermediate': [
                {
                    'question': 'What is overfitting in machine learning?',
                    'options': ['Model performs well on training data but poorly on test data', 'Model performs poorly on all data', 'Model is too simple', 'Model trains too fast'],
                    'correct': 0,
                    'keywords': ['overfitting', 'training data', 'test data', 'generalization']
                },
                {
                    'question': 'Which activation function is commonly used in hidden layers of neural networks?',
                    'options': ['Sigmoid', 'ReLU', 'Tanh', 'Linear'],
                    'correct': 1,
                    'keywords': ['activation function', 'ReLU', 'neural networks', 'hidden layers']
                },
                {
                    'question': 'What is the purpose of cross-validation?',
                    'options': ['To increase training speed', 'To evaluate model performance', 'To reduce overfitting', 'To clean data'],
                    'correct': 1,
                    'keywords': ['cross-validation', 'model evaluation', 'performance']
                },
                {
                    'question': 'Which metric is best for evaluating classification models with imbalanced datasets?',
                    'options': ['Accuracy', 'F1-score', 'Mean Squared Error', 'R-squared'],
                    'correct': 1,
                    'keywords': ['F1-score', 'classification', 'imbalanced dataset', 'metrics']
                }
            ],
            'advanced': [
                {
                    'question': 'What is the vanishing gradient problem in deep neural networks?',
                    'options': ['Gradients become too large', 'Gradients become very small in early layers', 'Network stops learning', 'Weights become zero'],
                    'correct': 1,
                    'keywords': ['vanishing gradient', 'deep neural networks', 'backpropagation']
                },
                {
                    'question': 'Which technique is used to handle the exploding gradient problem?',
                    'options': ['Dropout', 'Batch normalization', 'Gradient clipping', 'Data augmentation'],
                    'correct': 2,
                    'keywords': ['exploding gradient', 'gradient clipping', 'optimization']
                },
                {
                    'question': 'What is the main advantage of using attention mechanisms in neural networks?',
                    'options': ['Faster training', 'Better handling of long sequences', 'Reduced memory usage', 'Simpler architecture'],
                    'correct': 1,
                    'keywords': ['attention mechanism', 'long sequences', 'neural networks']
                },
                {
                    'question': 'Which optimization algorithm adapts learning rates for each parameter?',
                    'options': ['SGD', 'Adam', 'Momentum', 'RMSprop'],
                    'correct': 1,
                    'keywords': ['Adam optimizer', 'adaptive learning rate', 'optimization']
                }
            ],
            'master': [
                {
                    'question': 'What is the key innovation in Transformer architecture?',
                    'options': ['Convolutional layers', 'Self-attention mechanism', 'Recurrent connections', 'Pooling layers'],
                    'correct': 1,
                    'keywords': ['Transformer', 'self-attention', 'architecture innovation']
                },
                {
                    'question': 'Which technique is used in GANs to improve training stability?',
                    'options': ['Spectral normalization', 'Dropout', 'Batch normalization', 'Weight decay'],
                    'correct': 0,
                    'keywords': ['GANs', 'spectral normalization', 'training stability']
                },
                {
                    'question': 'What is the purpose of the KL divergence in VAEs?',
                    'options': ['Reconstruction loss', 'Regularization term', 'Classification loss', 'Activation function'],
                    'correct': 1,
                    'keywords': ['KL divergence', 'VAE', 'regularization', 'latent space']
                },
                {
                    'question': 'Which technique is used to prevent mode collapse in GANs?',
                    'options': ['Feature matching', 'Dropout', 'Data augmentation', 'Early stopping'],
                    'correct': 0,
                    'keywords': ['mode collapse', 'GANs', 'feature matching']
                }
            ],
            'professional': [
                {
                    'question': 'What is the main challenge addressed by federated learning?',
                    'options': ['Model accuracy', 'Data privacy and distribution', 'Training speed', 'Memory usage'],
                    'correct': 1,
                    'keywords': ['federated learning', 'data privacy', 'distributed learning']
                },
                {
                    'question': 'Which technique is used in neural architecture search (NAS)?',
                    'options': ['Reinforcement learning', 'Supervised learning', 'Clustering', 'Regression'],
                    'correct': 0,
                    'keywords': ['neural architecture search', 'NAS', 'reinforcement learning']
                },
                {
                    'question': 'What is the key principle behind contrastive learning?',
                    'options': ['Minimizing reconstruction error', 'Learning similar representations for similar data', 'Maximizing likelihood', 'Reducing dimensionality'],
                    'correct': 1,
                    'keywords': ['contrastive learning', 'representation learning', 'similarity']
                },
                {
                    'question': 'Which technique is used to improve few-shot learning performance?',
                    'options': ['Data augmentation', 'Meta-learning', 'Transfer learning', 'Ensemble methods'],
                    'correct': 1,
                    'keywords': ['few-shot learning', 'meta-learning', 'learning to learn']
                }
            ]
        }
    
    def generate_quiz(self) -> List[Dict[str, Any]]:
        """
        Generate a quiz with 10 questions (2 from each difficulty level).
        
        Returns:
            List[Dict]: List of selected questions
        """
        quiz_questions = []
        
        for difficulty in ['basic', 'intermediate', 'advanced', 'master', 'professional']:
            # Select 2 random questions from each difficulty level
            selected = random.sample(self.questions[difficulty], 2)
            for question in selected:
                question['difficulty'] = difficulty
            quiz_questions.extend(selected)
        
        # Shuffle the questions
        random.shuffle(quiz_questions)
        return quiz_questions
    
    def render_quiz_page(self):
        """
        Render the quiz page with timer and questions.
        """
        user = auth_manager.get_current_user()
        if not user:
            st.error("Please log in to take the quiz")
            return
        
        # Custom CSS for quiz styling
        st.markdown("""
        <style>
        .quiz-container {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem 0;
        }
        
        .quiz-header {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 2rem;
        }
        
        .question-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .timer-display {
            background: linear-gradient(45deg, #ff6b6b, #ffa500);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        
        .difficulty-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        
        .basic { background: #2ecc71; color: white; }
        .intermediate { background: #f39c12; color: white; }
        .advanced { background: #e74c3c; color: white; }
        .master { background: #9b59b6; color: white; }
        .professional { background: #34495e; color: white; }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="quiz-container">', unsafe_allow_html=True)
        st.markdown('<div class="quiz-header"><h1>🧠 AI Knowledge Quiz</h1><p>Test your AI and Machine Learning knowledge across 5 difficulty levels</p></div>', unsafe_allow_html=True)
        
        # Initialize quiz session
        if 'quiz_questions' not in st.session_state:
            st.session_state.quiz_questions = self.generate_quiz()
            st.session_state.quiz_start_time = time.time()
            st.session_state.current_question = 0
            st.session_state.quiz_answers = {}
            st.session_state.quiz_completed = False
        
        # Timer display with auto-refresh
        if not st.session_state.quiz_completed:
            # Create timer container
            timer_container = st.empty()
            
            elapsed_time = time.time() - st.session_state.quiz_start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            
            # Display timer
            timer_container.markdown(f'<div class="timer-display">⏱️ Time Elapsed: {minutes:02d}:{seconds:02d}</div>', unsafe_allow_html=True)
            
            # Auto-refresh mechanism using session state
            if 'timer_refresh_count' not in st.session_state:
                st.session_state.timer_refresh_count = 0
            
            # Increment counter and refresh periodically
            st.session_state.timer_refresh_count += 1
            if st.session_state.timer_refresh_count % 10 == 0:  # Refresh every 10 renders
                time.sleep(0.1)
                st.rerun()
        
        # Progress bar
        progress = st.session_state.current_question / len(st.session_state.quiz_questions)
        st.progress(progress)
        st.write(f"Question {st.session_state.current_question + 1} of {len(st.session_state.quiz_questions)}")
        
        if not st.session_state.quiz_completed:
            self.render_current_question()
        else:
            self.render_quiz_results()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_current_question(self):
        """
        Render the current question.
        """
        if st.session_state.current_question >= len(st.session_state.quiz_questions):
            self.complete_quiz()
            return
        
        question_data = st.session_state.quiz_questions[st.session_state.current_question]
        
        st.markdown('<div class="question-card">', unsafe_allow_html=True)
        
        # Difficulty badge
        difficulty = question_data['difficulty']
        st.markdown(f'<span class="difficulty-badge {difficulty}">{difficulty.upper()}</span>', unsafe_allow_html=True)
        
        # Question
        st.markdown(f"### {question_data['question']}")
        
        # Options - use session state to track selection
        current_answer_key = f"question_{st.session_state.current_question}"
        
        # Initialize radio state if not exists
        if current_answer_key not in st.session_state:
            st.session_state[current_answer_key] = None
        
        # Check if user has already answered this question
        if st.session_state.current_question in st.session_state.quiz_answers:
            # Restore previous answer
            selected_index = st.session_state.quiz_answers[st.session_state.current_question]['selected']
            st.session_state[current_answer_key] = question_data['options'][selected_index]
        
        # Determine current selection
        if st.session_state[current_answer_key] is None:
            current_index = None  # No selection
        elif st.session_state[current_answer_key] in question_data['options']:
            current_index = question_data['options'].index(st.session_state[current_answer_key])
        else:
            current_index = None
        
        selected_option = st.radio(
            "Select your answer:",
            question_data['options'],
            index=current_index,
            key=f"{current_answer_key}_radio"
        )
        
        # Update session state and get actual answer
        answer = selected_option
        st.session_state[current_answer_key] = selected_option
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.current_question == 0):
                if st.session_state.current_question > 0:
                    st.session_state.current_question -= 1
                    st.rerun()
        
        with col2:
            button_text = "Next ➡️" if st.session_state.current_question < len(st.session_state.quiz_questions) - 1 else "Finish Quiz 🏁"
            
            if st.button(button_text):
                # Save answer
                st.session_state.quiz_answers[st.session_state.current_question] = {
                    'selected': question_data['options'].index(answer),
                    'correct': question_data['correct'],
                    'keywords': question_data['keywords'],
                    'difficulty': question_data['difficulty']
                }
                
                if st.session_state.current_question < len(st.session_state.quiz_questions) - 1:
                    st.session_state.current_question += 1
                else:
                    self.complete_quiz()
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def complete_quiz(self):
        """
        Complete the quiz and save results.
        """
        st.session_state.quiz_completed = True
        st.session_state.quiz_end_time = time.time()
        
        # Calculate results
        total_time = st.session_state.quiz_end_time - st.session_state.quiz_start_time
        correct_answers = sum(1 for answer in st.session_state.quiz_answers.values() if answer['selected'] == answer['correct'])
        total_questions = len(st.session_state.quiz_questions)
        score_percentage = (correct_answers / total_questions) * 100
        
        # Collect all keywords
        all_keywords = []
        for answer in st.session_state.quiz_answers.values():
            all_keywords.extend(answer['keywords'])
        
        # Save to database
        user = auth_manager.get_current_user()
        self.db.save_quiz_result(
            user_id=user['id'],
            total_questions=total_questions,
            correct_answers=correct_answers,
            time_taken=int(total_time),
            difficulty_breakdown=self.get_difficulty_breakdown(),
            keywords=all_keywords
        )
        
        # Predict and save skill level based on performance
        predicted_skill = self.db.predict_skill_level(score_percentage, int(total_time))
        self.db.save_user_skill_prediction(user['id'], predicted_skill, score_percentage, int(total_time))
        
        # Store predicted skill in session for immediate display
        st.session_state.latest_skill_prediction = predicted_skill
        
        # Mark first-time quiz as completed for new users
        st.session_state.first_time_quiz_completed = True
    
    def get_difficulty_breakdown(self) -> Dict[str, Dict[str, int]]:
        """
        Get breakdown of performance by difficulty level.
        
        Returns:
            Dict: Performance breakdown by difficulty
        """
        breakdown = {}
        
        for answer in st.session_state.quiz_answers.values():
            difficulty = answer['difficulty']
            if difficulty not in breakdown:
                breakdown[difficulty] = {'correct': 0, 'total': 0}
            
            breakdown[difficulty]['total'] += 1
            if answer['selected'] == answer['correct']:
                breakdown[difficulty]['correct'] += 1
        
        return breakdown
    
    def render_quiz_results(self):
        """
        Render quiz results.
        """
        total_time = st.session_state.quiz_end_time - st.session_state.quiz_start_time
        correct_answers = sum(1 for answer in st.session_state.quiz_answers.values() if answer['selected'] == answer['correct'])
        total_questions = len(st.session_state.quiz_questions)
        score_percentage = (correct_answers / total_questions) * 100
        
        # Results display
        st.markdown("### 🎉 Quiz Completed!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Score", f"{score_percentage:.1f}%", f"{correct_answers}/{total_questions}")
        
        with col2:
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            st.metric("Time Taken", f"{minutes:02d}:{seconds:02d}")
        
        with col3:
            if score_percentage >= 80:
                grade = "Excellent! 🌟"
            elif score_percentage >= 60:
                grade = "Good! 👍"
            elif score_percentage >= 40:
                grade = "Fair 📚"
            else:
                grade = "Keep Learning! 💪"
            st.metric("Grade", grade)
        
        # Display predicted skill level
        if 'latest_skill_prediction' in st.session_state:
            st.markdown("### 🏆 Your Skill Level")
            skill_level = st.session_state.latest_skill_prediction
            
            # Ensure skill_level is a string (safety check)
            if not isinstance(skill_level, str):
                skill_level = 'beginner'
            
            skill_badges = {
                'beginner': {'emoji': '🌱', 'color': '#4CAF50', 'title': 'Beginner'},
                'intermediate': {'emoji': '📚', 'color': '#2196F3', 'title': 'Intermediate'},
                'advanced': {'emoji': '🎯', 'color': '#FF9800', 'title': 'Advanced'},
                'master': {'emoji': '🏆', 'color': '#9C27B0', 'title': 'Master'},
                'professional': {'emoji': '👑', 'color': '#F44336', 'title': 'Professional'}
            }
            
            badge_info = skill_badges.get(skill_level, skill_badges['beginner'])
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {badge_info['color']}22 0%, {badge_info['color']}44 100%);
                border: 2px solid {badge_info['color']};
                border-radius: 10px;
                padding: 1rem;
                text-align: center;
                margin: 1rem 0;
            ">
                <h3 style="color: {badge_info['color']}; margin: 0;">
                    {badge_info['emoji']} {badge_info['title']}
                </h3>
                <p style="color: #666; margin: 0.3rem 0 0 0;">Based on your performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Difficulty breakdown
        st.markdown("### 📊 Performance by Difficulty")
        breakdown = self.get_difficulty_breakdown()
        
        for difficulty, stats in breakdown.items():
            percentage = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            st.write(f"**{difficulty.title()}**: {stats['correct']}/{stats['total']} ({percentage:.1f}%)")
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏠 Go to Dashboard"):
                # Clear quiz session and go to dashboard
                for key in ['quiz_questions', 'quiz_start_time', 'current_question', 'quiz_answers', 'quiz_completed', 'quiz_end_time']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.dashboard_page = 'overview'  # Redirect to overview page
                st.rerun()
        
        with col2:
            if st.button("🔄 Take Another Quiz"):
                # Clear quiz session
                for key in ['quiz_questions', 'quiz_start_time', 'current_question', 'quiz_answers', 'quiz_completed', 'quiz_end_time']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

class DashboardManager:
    def __init__(self):
        """
        Initialize dashboard manager.
        """
        self.db = Database()
        self.quiz_manager = QuizManager()
    
    def render_dashboard(self):
        """
        Render the main dashboard.
        """
        user = auth_manager.get_current_user()
        if not user:
            st.error("Please log in to access the dashboard")
            return
        
        # Custom CSS for dashboard
        st.markdown("""
        <style>
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin: 1rem 0;
            text-align: center;
        }
        
        .nav-button {
            margin: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown(f'<div class="dashboard-header"><h1>🎓 Welcome to Your Learning Dashboard</h1><p>Hello, {user["email"]}!</p></div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🧠 Take Quiz", key="nav_quiz"):
                st.session_state.dashboard_page = 'quiz'
        
        with col2:
            if st.button("📊 Quiz History", key="nav_history"):
                st.session_state.dashboard_page = 'history'
        
        with col3:
            if st.button("💬 AI Chat", key="nav_chat"):
                st.session_state.dashboard_page = 'chat'
        
        with col4:
            if st.button("👋 Logout", key="nav_logout"):
                auth_manager.logout()
        
        # Initialize dashboard page
        if 'dashboard_page' not in st.session_state:
            st.session_state.dashboard_page = 'overview'
        
        # Render selected page
        if st.session_state.dashboard_page == 'quiz':
            self.quiz_manager.render_quiz_page()
        elif st.session_state.dashboard_page == 'history':
            self.render_quiz_history()
        elif st.session_state.dashboard_page == 'chat':
            self.render_ai_chat()
        else:
            self.render_overview()
    
    def render_overview(self):
        """
        Render dashboard overview with statistics.
        """
        user = auth_manager.get_current_user()
        stats = self.db.get_user_stats(user['id'])
        
        st.markdown("### 📈 Your Learning Statistics")
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Quizzes", stats.get('total_quizzes', 0))
            
            with col2:
                avg_score = stats.get('average_score', 0)
                st.metric("Average Score", f"{avg_score:.1f}%")
            
            with col3:
                best_score = stats.get('best_score', 0)
                st.metric("Best Score", f"{best_score:.1f}%")
            
            with col4:
                total_time = stats.get('total_time', 0)
                hours = int(total_time // 3600)
                minutes = int((total_time % 3600) // 60)
                st.metric("Total Study Time", f"{hours}h {minutes}m")
            
            # Display skill level badge
            st.markdown("### 🏆 Your Current Skill Level")
            current_skill_data = self.db.get_user_skill_level(user['id'])
            
            if current_skill_data or 'latest_skill_prediction' in st.session_state:
                # Extract skill level string from the data dict
                if current_skill_data and isinstance(current_skill_data, dict):
                    skill_level = current_skill_data.get('skill_level', 'beginner')
                else:
                    skill_level = st.session_state.get('latest_skill_prediction', 'beginner')
                
                # Ensure skill_level is a string (safety check)
                if not isinstance(skill_level, str):
                    skill_level = 'beginner'
                
                # Additional safety: if it's still not a valid skill level, default to beginner
                valid_skills = ['beginner', 'intermediate', 'advanced', 'master', 'professional']
                if skill_level not in valid_skills:
                    skill_level = 'beginner'
                
                # Define skill level badges with colors and emojis
                skill_badges = {
                    'beginner': {'emoji': '🌱', 'color': '#4CAF50', 'title': 'Beginner', 'description': 'Just getting started!'},
                    'intermediate': {'emoji': '📚', 'color': '#2196F3', 'title': 'Intermediate', 'description': 'Making good progress!'},
                    'advanced': {'emoji': '🎯', 'color': '#FF9800', 'title': 'Advanced', 'description': 'Strong understanding!'},
                    'master': {'emoji': '🏆', 'color': '#9C27B0', 'title': 'Master', 'description': 'Excellent expertise!'},
                    'professional': {'emoji': '👑', 'color': '#F44336', 'title': 'Professional', 'description': 'Expert level mastery!'}
                }
                
                badge_info = skill_badges.get(skill_level, skill_badges['beginner'])
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {badge_info['color']}22 0%, {badge_info['color']}44 100%);
                    border: 2px solid {badge_info['color']};
                    border-radius: 15px;
                    padding: 1.5rem;
                    text-align: center;
                    margin: 1rem 0;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                ">
                    <h2 style="color: {badge_info['color']}; margin: 0;">
                        {badge_info['emoji']} {badge_info['title']}
                    </h2>
                    <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                        {badge_info['description']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("🎯 Complete a quiz to see your skill level prediction!")
        else:
            st.info("📚 Take your first quiz to see your statistics!")
            if st.button("🚀 Start Your First Quiz"):
                st.session_state.dashboard_page = 'quiz'
                st.rerun()
        
        # Add keyword-based recommendations section
        st.markdown("### 📚 Recommended Content")
        self.render_keyword_recommendations()
    
    def render_quiz_history(self):
        """
        Render quiz history and results.
        """
        user = auth_manager.get_current_user()
        quiz_history = self.db.get_user_quiz_history(user['id'])
        
        st.markdown("### 📚 Quiz History")
        
        if quiz_history:
            for i, quiz in enumerate(quiz_history):
                with st.expander(f"Quiz {i+1} - {quiz['score_percentage']:.1f}% - {quiz['date']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Score:** {quiz['score_percentage']:.1f}%")
                        st.write(f"**Correct:** {quiz['correct_answers']}/{quiz['total_questions']}")
                    
                    with col2:
                        minutes = int(quiz['time_taken'] // 60)
                        seconds = int(quiz['time_taken'] % 60)
                        st.write(f"**Time:** {minutes:02d}:{seconds:02d}")
                    
                    with col3:
                        if quiz['keywords']:
                            keywords = quiz['keywords'] if isinstance(quiz['keywords'], list) else quiz['keywords']
                            if isinstance(keywords, list):
                                st.write(f"**Topics:** {', '.join(keywords[:5])}")
                            else:
                                st.write(f"**Topics:** {str(keywords)[:50]}...")
        else:
            st.info("No quiz history found. Take your first quiz!")
    
    def render_ai_chat(self):
        """
        Render AI chat interface with history.
        """
        from streamlit_app import initialize_components, generate_response, process_uploaded_pdf, save_user_chat_history
        
        st.markdown("### 💬 AI Learning Assistant")
        
        # Initialize components
        initialize_components()
        
        # Chat interface styling
        st.markdown("""
        <style>
        .chat-container {
            background: #ffffff;
            border-radius: 15px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(255, 165, 0, 0.08);
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(255, 235, 59, 0.2);
        }
        
        .user-message {
            background: linear-gradient(135deg, #ff8c42 0%, #ffa726 100%);
            color: white;
            padding: 0.8rem 1.2rem;
            border-radius: 18px 18px 5px 18px;
            margin: 0.5rem 0;
            margin-left: 20%;
            font-weight: 500;
            box-shadow: 0 3px 10px rgba(255, 140, 66, 0.3);
        }
        
        .assistant-message {
            background: linear-gradient(135deg, #66bb6a 0%, #81c784 100%);
            color: white;
            padding: 0.8rem 1.2rem;
            border-radius: 18px 18px 18px 5px;
            margin: 0.5rem 0;
            margin-right: 20%;
            font-weight: 500;
            box-shadow: 0 3px 10px rgba(102, 187, 106, 0.3);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Chat display
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        user_query = st.text_input(
            "Ask me anything about ML books...",
            placeholder="e.g., What are the best algorithms for classification?",
            key="dashboard_chat_input"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Send Message", key="dashboard_send_message") and user_query:
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_query
                })
                
                # Search for similar documents
                if st.session_state.vector_embeddings:
                    with st.spinner("🔍 Searching knowledge base..."):
                        similar_docs = st.session_state.vector_embeddings.search_similar_documents(
                            user_query,
                            top_k=5
                        )
                        
                        # Generate response
                        response = generate_response(user_query, similar_docs)
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                        
                        # Save chat history to database
                        save_user_chat_history()
                        
                        # Rerun to update chat display
                        st.rerun()
                else:
                    st.error("❌ Vector embeddings not loaded. Please ensure the system is properly initialized.")
        
        with col2:
            if st.button("🗑️ Clear Chat", key="dashboard_clear_chat"):
                st.session_state.chat_history = []
                # Also clear from database
                user = auth_manager.get_current_user()
                if user:
                    self.db.delete_user_chat_history(user['id'])
                st.success("Chat cleared!")
                st.rerun()
        
        # File upload section
        st.markdown("---")
        st.markdown("#### 📤 Upload New PDF")
        
        uploaded_file = st.file_uploader(
            "Add a new document to the knowledge base",
            type="pdf",
            key="dashboard_upload"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process PDF", key="dashboard_process_pdf"):
                with st.spinner("Processing PDF..."):
                    processed_data, metadata = process_uploaded_pdf(uploaded_file)
                    
                    if processed_data:
                        st.success(f"✅ Successfully processed: {uploaded_file.name}")
                        
                        # Show processing results
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("Original Length", f"{len(processed_data['original_text'])} chars")
                        with col_b:
                            st.metric("Processed Length", f"{len(processed_data['processed_text_lemmatized'])} chars")
                    else:
                        st.error("❌ Failed to process PDF")
        
        # Show recent document sources if there's chat history
        if st.session_state.chat_history and st.session_state.vector_embeddings:
            st.markdown("---")
            st.markdown("#### 🔍 Recent Document Sources")
            
            # Get last user query
            last_query = None
            for message in reversed(st.session_state.chat_history):
                if message['role'] == 'user':
                    last_query = message['content']
                    break
            
            if last_query:
                similar_docs = st.session_state.vector_embeddings.search_similar_documents(
                    last_query,
                    top_k=3
                )
                
                if similar_docs:
                    for i, doc in enumerate(similar_docs):
                        score = doc['similarity_score']
                        filename = doc['metadata']['filename']
                        
                        # Color code based on similarity score
                        if score > 0.8:
                            color = "#4CAF50"  # Green
                        elif score > 0.6:
                            color = "#FF9800"  # Orange
                        else:
                            color = "#F44336"  # Red
                        
                        st.markdown(f"""
                        <div style="background: white; border-left: 4px solid {color}; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <strong>📄 {filename}</strong><br>
                            <small>Similarity: {score:.3f}</small>
                        </div>
                        """, unsafe_allow_html=True)
    
    def render_keyword_recommendations(self):
        """
        Render keyword-based document recommendations.
        """
        try:
            # Import here to avoid circular imports
            from streamlit_app import get_keyword_based_recommendations
            
            user = auth_manager.get_current_user()
            if not user:
                st.info("Please log in to see personalized recommendations.")
                return
            
            # Get recommendations based on quiz keywords
            recommendations = get_keyword_based_recommendations(
                user_id=user['id'],
                top_k=5  # Show top 5 recommendations
            )
            
            if recommendations:
                st.markdown("**Based on your quiz keywords, here are the top 5 most relevant documents from ML books:**")
                
                for i, doc in enumerate(recommendations, 1):
                    score = doc.get('similarity_score', 0)
                    filename = doc.get('metadata', {}).get('filename', f'Document {i}')
                    
                    # Color code based on similarity score like in AI chat
                    if score > 0.8:
                        color = "#4CAF50"  # Green
                        score_label = "Excellent Match"
                    elif score > 0.6:
                        color = "#FF9800"  # Orange
                        score_label = "Good Match"
                    elif score > 0.4:
                        color = "#2196F3"  # Blue
                        score_label = "Fair Match"
                    else:
                        color = "#F44336"  # Red
                        score_label = "Weak Match"
                    
                    # Display document card with similarity styling
                    st.markdown(f"""
                    <div style="background: white; border-left: 4px solid {color}; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <strong>📄 {filename}</strong><br>
                        <small style="color: {color};">Similarity: {score:.3f} ({score_label})</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add PDF preview button
                    pdf_path = doc.get('pdf_path', '')
                    if pdf_path and os.path.exists(pdf_path):
                        from streamlit_app import get_pdf_download_link
                        preview_link = get_pdf_download_link(pdf_path, filename)
                        st.markdown(preview_link, unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color: #666; font-size: 12px;'>📄 PDF preview not available</span>", unsafe_allow_html=True)
                    
                    with st.expander(f"View Content Preview - {filename}"):
                        # Show document content preview
                        content = doc.get('content', 'No content available')
                        if len(content) > 400:
                            content = content[:400] + "..."
                        st.write(content)
                        
                        # Show recommendation context
                        if 'recommendation_keywords' in doc:
                            st.markdown(f"**🔍 Matching Keywords:** {', '.join(doc['recommendation_keywords'])}")
                        
                        # Show metadata if available
                        metadata = doc.get('metadata', {})
                        if 'page_number' in metadata:
                            st.markdown(f"**📖 Page:** {metadata['page_number']}")
                        if 'chunk_index' in metadata:
                            st.markdown(f"**📝 Section:** {metadata['chunk_index']}")
            else:
                # Check if user has taken quizzes
                quiz_history = self.db.get_user_quiz_history(user['id'])
                if quiz_history:
                    st.info("No recommendations available. Try taking more quizzes to get personalized content suggestions!")
                else:
                    st.info("Take a quiz first to get personalized document recommendations based on your interests!")
                    
        except Exception as e:
            st.error(f"Error loading recommendations: {str(e)}")
            print(f"Recommendation error: {e}")

# Global instances
quiz_manager = QuizManager()
dashboard_manager = DashboardManager()