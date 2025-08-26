# 📚 ML AI Assistant Setup Guide

> A comprehensive guide to set up and run the Machine Learning AI Assistant with RAG capabilities

---

## 📋 Prerequisites

- Python 3.8 or higher
- Git installed on your system
- Internet connection for downloading dependencies
- API keys from OpenAI and/or Google Gemini

---

## 🚀 Step-by-Step Setup Instructions

### **Step 0: Download ML Books** 📖

1. **Access the Google Drive folder:**
   - Open: [ML Books Collection](https://drive.google.com/drive/folders/1jIJMyBOeWiVxLCUUtLvEFEFCnWxbh6cs?usp=drive_link)
   
2. **Download all contents:**
   - Select all files in the folder
   - Download as ZIP or individual files
   
3. **Organize the files:**
   - Create a folder named `ML Books` in your project directory
   - Extract all downloaded PDFs into this folder
   - **Important:** If there are subfolders, move all PDF contents to the main `ML Books` folder

---

### **Step 1: Create Local Directory** 📁

```bash
# Create your project directory
mkdir ml-ai-assistant
cd ml-ai-assistant
```

---

### **Step 2: Clone the Repository** 🔄

```bash
# Clone the repository
git clone https://github.com/CodeAKD/Machine-Learning-AI-Assistant.git
cd Machine-Learning-AI-Assistant

# Alternative: Download as ZIP from GitHub
# Extract the ZIP file to your desired location
```

---

### **Step 3: Create API Keys** 🔑

#### **OpenAI API Key:**
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to **API Keys** section
4. Click **"Create new secret key"**
5. Copy and save your API key securely

#### **Google Gemini API Key (Optional):**
1. Visit [Google AI Studio](https://makersuite.google.com/)
2. Sign in with your Google account
3. Create a new API key
4. Copy and save your API key securely

---

### **Step 4: Create Environment File** 🔧

Create a `.env` file in the project root directory:

```bash
# Create .env file
touch .env
```

Add your API keys to the `.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Model Configuration
OPENAI_MODEL=gpt-3.5-turbo
```

> ⚠️ **Security Note:** Never commit your `.env` file to version control!

---

### **Step 5: Create Virtual Environment** 🐍

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

---

### **Step 6: Install Dependencies** 📦

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install python-dotenv (if not in requirements.txt)
pip install python-dotenv

# Optional: Install additional packages if needed
pip install streamlit nltk sentence-transformers faiss-cpu openai
```

---

### **Step 7: Read Technical Documentation** 📚

**Before proceeding, please read:**
- [`ML_AI_Assistant_Technical_Documentation.md`](./ML_AI_Assistant_Technical_Documentation.md)

This document provides:
- 🏗️ **System Architecture Overview**
- 🔧 **Module-level Analysis**
- 🔄 **Data Flow Diagrams**
- 🧠 **Technical Concepts Explained**
- ⚡ **Performance Considerations**

---

### **Step 8: Verify ML Books Setup** ✅

**Ensure Step 0 is completed:**

```bash
# Check if ML Books directory exists and contains PDFs
ls "ML Books/"

# You should see PDF files like:
# - Deep Learning by Ian Goodfellow.pdf
# - Hands-On Large Language Models.pdf
# - NLP with Transformer models.pdf
# - etc.
```

---

### **Step 9: Run NLP Pipeline** 🔄

```bash
# Process the PDF documents
python nlp_pipeline.py
```

**Expected Output:**
- ✅ PDF text extraction
- ✅ Text cleaning and preprocessing
- ✅ Tokenization and NLP processing
- ✅ Cleaned data saved to `cleaned ML Books/`

---

### **Step 10: Generate Vector Embeddings** 🎯

```bash
# Create vector embeddings for similarity search
python vector_embeddings.py
```

**Expected Output:**
- ✅ Document loading from cleaned data
- ✅ Sentence transformer model initialization
- ✅ Vector embeddings generation
- ✅ FAISS index creation and saving
- ✅ Document statistics display

---

### **Step 11: Launch Streamlit Application** 🚀

```bash
# Start the web application
streamlit run streamlit_app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

---

### **Step 12: Responsible Usage Guidelines** ⚠️

**Please use the application responsibly:**

- 🔒 **API Usage:** Monitor your API usage to avoid unexpected charges
- 📊 **Rate Limits:** Respect OpenAI/Gemini API rate limits
- 💾 **Data Privacy:** Don't upload sensitive or confidential documents
- 🔄 **Resource Management:** Close the application when not in use
- 📝 **Content Guidelines:** Use for educational and research purposes

---

## 🎯 Application Features

### **Core Functionality:**
- 📄 **PDF Document Processing**
- 🔍 **Semantic Search & Similarity**
- 🤖 **AI-Powered Q&A with RAG**
- 📊 **Document Statistics & Analytics**
- 💬 **Interactive Chat Interface**

### **Technical Stack:**
- **Frontend:** Streamlit
- **NLP:** NLTK, Sentence Transformers
- **Vector Search:** FAISS
- **AI Models:** OpenAI GPT, Google Gemini
- **Document Processing:** PyPDF2, pandas

---

## 🛠️ Troubleshooting

### **Common Issues:**

#### **1. Module Not Found Error:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### **2. API Key Issues:**
```bash
# Check .env file exists and contains valid keys
cat .env

# Verify environment variables are loaded
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

#### **3. NLTK Data Missing:**
```python
# Run in Python to download NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

#### **4. Large File Issues:**
- Ensure PDF files are under 100MB each
- Check available disk space
- Verify `ML Books/` directory permissions

---

## 📞 Support

If you encounter issues:
1. Check the [Technical Documentation](./ML_AI_Assistant_Technical_Documentation.md)
2. Review error messages carefully
3. Ensure all prerequisites are met
4. Verify API keys are correctly configured

---

## 🎉 Success!

Once setup is complete, you'll have a fully functional ML AI Assistant that can:
- Process and analyze ML textbooks
- Answer questions using RAG technology
- Provide semantic search capabilities
- Offer interactive learning experiences

**Happy Learning! 🚀📚**
