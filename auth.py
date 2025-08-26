import streamlit as st
import re
from database import Database
from typing import Optional, Dict, Any
import uuid

class AuthManager:
    def __init__(self):
        """
        Initialize authentication manager with database connection.
        """
        self.db = Database()
        
    def validate_email(self, email: str) -> bool:
        """
        Validate email format using regex.
        
        Args:
            email (str): Email address to validate
            
        Returns:
            bool: True if email is valid, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_password(self, password: str) -> Dict[str, bool]:
        """
        Validate password according to requirements:
        - At least 8 characters long
        - Contains at least one lowercase letter
        - Contains at least one uppercase letter
        - Contains at least one number
        - Contains at least one symbol
        
        Args:
            password (str): Password to validate
            
        Returns:
            Dict[str, bool]: Validation results for each requirement
        """
        validation = {
            'length': len(password) >= 8,
            'lowercase': bool(re.search(r'[a-z]', password)),
            'uppercase': bool(re.search(r'[A-Z]', password)),
            'number': bool(re.search(r'\d', password)),
            'symbol': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
        }
        
        validation['valid'] = all(validation.values())
        return validation
    
    def display_password_requirements(self, password: str = ""):
        """
        Display password requirements with real-time validation.
        
        Args:
            password (str): Current password input
        """
        validation = self.validate_password(password)
        
        st.write("**Password Requirements:**")
        
        requirements = [
            ("At least 8 characters", validation['length']),
            ("One lowercase letter (a-z)", validation['lowercase']),
            ("One uppercase letter (A-Z)", validation['uppercase']),
            ("One number (0-9)", validation['number']),
            ("One symbol (!@#$%^&*)", validation['symbol'])
        ]
        
        for req, met in requirements:
            if met:
                st.success(f"✅ {req}")
            else:
                st.error(f"❌ {req}")
    
    def render_auth_page(self):
        """
        Render the authentication page with sign up and sign in forms.
        """
        # Custom CSS for gradient background and styling
        st.markdown("""
        <style>
        .auth-container {
            background: linear-gradient(135deg, #fff 0%, #ffe4b5 25%, #f0fff0 50%, #fffacd 75%, #fff 100%);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .auth-title {
            text-align: center;
            color: #2c3e50;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .auth-subtitle {
            text-align: center;
            color: #7f8c8d;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        .auth-form {
            background: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .stButton > button {
            background: linear-gradient(45deg, #ff6b6b, #ffa500);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
        }
        
        .auth-switch {
            text-align: center;
            margin-top: 1rem;
            color: #7f8c8d;
        }
        
        .auth-switch a {
            color: #ff6b6b;
            text-decoration: none;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main container
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        
        # Title
        st.markdown('<h1 class="auth-title">🎓 Welcome to Personalized Learning using AI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="auth-subtitle">Enhance your learning journey with AI-powered insights</p>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'auth_mode' not in st.session_state:
            st.session_state.auth_mode = None
        
        # Auth mode toggle with active state styling
        col1, col2 = st.columns(2)
        
        with col1:
            signin_type = "primary" if st.session_state.auth_mode == 'signin' else "secondary"
            if st.button("🔑 Sign In", key="signin_btn", type=signin_type):
                st.session_state.auth_mode = 'signin'
                st.rerun()
        
        with col2:
            signup_type = "primary" if st.session_state.auth_mode == 'signup' else "secondary"
            if st.button("📝 Sign Up", key="signup_btn", type=signup_type):
                st.session_state.auth_mode = 'signup'
                st.rerun()
        
        # Only show form if a mode is selected
        if st.session_state.auth_mode is not None:
            st.markdown('<div class="auth-form">', unsafe_allow_html=True)
            
            if st.session_state.auth_mode == 'signup':
                self.render_signup_form()
            else:
                self.render_signin_form()
            
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_signup_form(self):
        """
        Render the sign up form.
        """
        st.markdown("### 📝 Create Your Account")
        
        with st.form("signup_form"):
            email = st.text_input("📧 Email Address", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Create a strong password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
            
            # Real-time email validation
            if email:
                if self.validate_email(email):
                    st.success("✅ Valid email format")
                else:
                    st.error("❌ Invalid email format")
            
            # Real-time password validation
            if password:
                self.display_password_requirements(password)
            
            # Password confirmation check
            if password and confirm_password:
                if password == confirm_password:
                    st.success("✅ Passwords match")
                else:
                    st.error("❌ Passwords do not match")
            
            submit_button = st.form_submit_button("🚀 Create Account")
            
            if submit_button:
                self.handle_signup(email, password, confirm_password)
    
    def render_signin_form(self):
        """
        Render the sign in form.
        """
        st.markdown("### 🔑 Sign In to Your Account")
        
        with st.form("signin_form"):
            email = st.text_input("📧 Email Address", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            submit_button = st.form_submit_button("🔓 Sign In")
            
            if submit_button:
                self.handle_signin(email, password)
    
    def handle_signup(self, email: str, password: str, confirm_password: str):
        """
        Handle user signup process.
        
        Args:
            email (str): User email
            password (str): User password
            confirm_password (str): Password confirmation
        """
        # Validate inputs
        if not email or not password or not confirm_password:
            st.error("❌ Please fill in all fields")
            return
        
        if not self.validate_email(email):
            st.error("❌ Please enter a valid email address")
            return
        
        password_validation = self.validate_password(password)
        if not password_validation['valid']:
            st.error("❌ Password does not meet requirements")
            return
        
        if password != confirm_password:
            st.error("❌ Passwords do not match")
            return
        
        # Attempt to create user
        if self.db.create_user(email, password):
            st.success("🎉 Account created successfully! Please sign in.")
            st.session_state.auth_mode = 'signin'
            st.rerun()
        else:
            st.error("❌ Email already exists or registration failed")
    
    def handle_signin(self, email: str, password: str):
        """
        Handle user signin process.
        
        Args:
            email (str): User email
            password (str): User password
        """
        # Validate inputs
        if not email or not password:
            st.error("❌ Please fill in all fields")
            return
        
        # Attempt authentication
        user_data = self.db.authenticate_user(email, password)
        
        if user_data:
            # Set session state
            st.session_state.authenticated = True
            st.session_state.user_id = user_data['id']
            st.session_state.user_email = user_data['email']
            st.session_state.session_id = str(uuid.uuid4())
            
            st.success(f"🎉 Welcome back, {email}!")
            st.rerun()
        else:
            st.error("❌ Invalid email or password")
    
    def logout(self):
        """
        Handle user logout.
        """
        # Clear session state
        for key in ['authenticated', 'user_id', 'user_email', 'session_id']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("👋 Logged out successfully!")
        st.rerun()
    
    def is_authenticated(self) -> bool:
        """
        Check if user is authenticated.
        
        Returns:
            bool: True if user is authenticated
        """
        return st.session_state.get('authenticated', False)
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """
        Get current authenticated user data.
        
        Returns:
            Optional[Dict]: User data if authenticated, None otherwise
        """
        if self.is_authenticated():
            return {
                'id': st.session_state.get('user_id'),
                'email': st.session_state.get('user_email'),
                'session_id': st.session_state.get('session_id')
            }
        return None
    
    def require_auth(self):
        """
        Decorator-like function to require authentication.
        Redirects to auth page if not authenticated.
        
        Returns:
            bool: True if authenticated, False otherwise
        """
        if not self.is_authenticated():
            self.render_auth_page()
            return False
        return True

# Global auth manager instance
auth_manager = AuthManager()