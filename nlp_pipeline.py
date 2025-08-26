import os
import re
import string
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
import pandas as pd
import json
from pathlib import Path

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('omw-1.4', quiet=True)

class NLPPipeline:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    def remove_punctuation(self, text):
        """Remove punctuation from text"""
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def tokenize_text(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """Apply lemmatization to tokens"""
        # Get POS tags for better lemmatization
        pos_tags = pos_tag(tokens)
        lemmatized = []
        
        for word, pos in pos_tags:
            # Convert POS tag to WordNet format
            if pos.startswith('V'):
                pos_tag_wn = 'v'  # Verb
            elif pos.startswith('N'):
                pos_tag_wn = 'n'  # Noun
            elif pos.startswith('R'):
                pos_tag_wn = 'r'  # Adverb
            elif pos.startswith('J'):
                pos_tag_wn = 'a'  # Adjective
            else:
                pos_tag_wn = 'n'  # Default to noun
            
            lemmatized.append(self.lemmatizer.lemmatize(word, pos_tag_wn))
        
        return lemmatized
    
    def process_text(self, text, include_stemming=True, include_lemmatization=True):
        """Complete NLP pipeline for text processing"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Remove punctuation
        cleaned_text = self.remove_punctuation(cleaned_text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Apply stemming if requested
        if include_stemming:
            stemmed_tokens = self.stem_tokens(tokens)
        else:
            stemmed_tokens = tokens
        
        # Apply lemmatization if requested
        if include_lemmatization:
            lemmatized_tokens = self.lemmatize_tokens(tokens)
        else:
            lemmatized_tokens = tokens
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'stemmed_tokens': stemmed_tokens,
            'lemmatized_tokens': lemmatized_tokens,
            'processed_text_stemmed': ' '.join(stemmed_tokens),
            'processed_text_lemmatized': ' '.join(lemmatized_tokens)
        }
    
    def process_pdf_file(self, pdf_path, output_dir):
        """Process a single PDF file and save cleaned data"""
        print(f"Processing: {pdf_path}")
        
        # Extract text from PDF
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        if not raw_text.strip():
            print(f"No text extracted from {pdf_path}")
            return None
        
        # Process the text
        processed_data = self.process_text(raw_text)
        
        # Create output filename
        pdf_name = Path(pdf_path).stem
        output_file = os.path.join(output_dir, f"{pdf_name}_cleaned.json")
        
        # Save processed data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # Also save just the cleaned text for easy access
        text_output_file = os.path.join(output_dir, f"{pdf_name}_cleaned.txt")
        with open(text_output_file, 'w', encoding='utf-8') as f:
            f.write(processed_data['processed_text_lemmatized'])
        
        print(f"Saved cleaned data to: {output_file}")
        return processed_data
    
    def process_pdf_directory(self, input_dir, output_dir):
        """Process all PDF files in a directory"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all PDF files
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return []
        
        processed_files = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            try:
                result = self.process_pdf_file(pdf_path, output_dir)
                if result:
                    processed_files.append({
                        'filename': pdf_file,
                        'processed_data': result
                    })
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                continue
        
        print(f"\nProcessed {len(processed_files)} PDF files successfully.")
        return processed_files

def main():
    """Main function to process ML Books directory"""
    # Initialize NLP pipeline
    nlp = NLPPipeline()
    
    # Define paths
    input_dir = "ML Books"
    output_dir = "cleaned ML Books"
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' not found!")
        return
    
    # Process all PDF files
    print("Starting NLP Pipeline for ML Books...")
    processed_files = nlp.process_pdf_directory(input_dir, output_dir)
    
    # Create a summary file
    summary_file = os.path.join(output_dir, "processing_summary.json")
    summary_data = {
        'total_files_processed': len(processed_files),
        'processed_files': [f['filename'] for f in processed_files],
        'output_directory': output_dir
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete! Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()