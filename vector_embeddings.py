import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
import faiss
from typing import List, Dict, Tuple

class VectorEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the vector embeddings system
        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = []
        self.document_metadata = []
        self.index = None
        
    def load_cleaned_documents(self, cleaned_dir):
        """
        Load all cleaned documents from the directory
        Args:
            cleaned_dir: Directory containing cleaned text files
        """
        print(f"Loading documents from: {cleaned_dir}")
        
        if not os.path.exists(cleaned_dir):
            print(f"Directory {cleaned_dir} does not exist!")
            return
        
        # Load text files
        text_files = [f for f in os.listdir(cleaned_dir) if f.endswith('_cleaned.txt')]
        
        for text_file in text_files:
            file_path = os.path.join(cleaned_dir, text_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if content:
                    self.documents.append(content)
                    # Extract original filename
                    original_name = text_file.replace('_cleaned.txt', '.pdf')
                    self.document_metadata.append({
                        'filename': original_name,
                        'cleaned_filename': text_file,
                        'file_path': file_path,
                        'content_length': len(content)
                    })
                    
            except Exception as e:
                print(f"Error loading {text_file}: {str(e)}")
                continue
        
        print(f"Loaded {len(self.documents)} documents")
    
    def create_embeddings(self):
        """
        Create embeddings for all loaded documents
        """
        if not self.documents:
            print("No documents loaded. Please load documents first.")
            return
        
        print("Creating embeddings for documents...")
        
        # Create embeddings
        self.embeddings = self.model.encode(self.documents, show_progress_bar=True)
        
        # Create FAISS index for efficient similarity search
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype('float32'))
        
        print(f"Created embeddings with dimension: {dimension}")
    
    def save_embeddings(self, save_dir='embeddings'):
        """
        Save embeddings and metadata to disk
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save embeddings
        embeddings_file = os.path.join(save_dir, 'embeddings.npy')
        np.save(embeddings_file, self.embeddings)
        
        # Save metadata
        metadata_file = os.path.join(save_dir, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.document_metadata, f, indent=2, ensure_ascii=False)
        
        # Save FAISS index
        index_file = os.path.join(save_dir, 'faiss_index.bin')
        faiss.write_index(self.index, index_file)
        
        # Save model info
        model_info = {
            'model_name': self.model._modules['0'].auto_model.name_or_path,
            'num_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1]
        }
        
        model_info_file = os.path.join(save_dir, 'model_info.json')
        with open(model_info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Embeddings saved to: {save_dir}")
    
    def load_embeddings(self, save_dir='embeddings'):
        """
        Load embeddings and metadata from disk
        """
        try:
            # Load embeddings
            embeddings_file = os.path.join(save_dir, 'embeddings.npy')
            self.embeddings = np.load(embeddings_file)
            
            # Load metadata
            metadata_file = os.path.join(save_dir, 'metadata.json')
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.document_metadata = json.load(f)
            
            # Load FAISS index
            index_file = os.path.join(save_dir, 'faiss_index.bin')
            self.index = faiss.read_index(index_file)
            
            # Reconstruct documents list (for compatibility)
            self.documents = [''] * len(self.document_metadata)  # Placeholder
            
            print(f"Loaded embeddings for {len(self.document_metadata)} documents")
            return True
            
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return False
    
    def search_similar_documents(self, query_text, top_k=5):
        """
        Search for similar documents using the query text
        Args:
            query_text: Text to search for
            top_k: Number of top similar documents to return
        Returns:
            List of tuples (similarity_score, document_metadata)
        """
        if self.embeddings is None or self.index is None:
            print("Embeddings not loaded. Please create or load embeddings first.")
            return []
        
        # Create embedding for query
        query_embedding = self.model.encode([query_text])
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search using FAISS
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.document_metadata):
                results.append({
                    'rank': i + 1,
                    'similarity_score': float(similarity),
                    'metadata': self.document_metadata[idx]
                })
        
        return results
    
    def add_new_document(self, text_content, metadata, save_dir='embeddings'):
        """
        Add a new document to the existing embeddings
        Args:
            text_content: Cleaned text content of the new document
            metadata: Metadata dictionary for the new document
            save_dir: Directory to save updated embeddings
        """
        # Create embedding for new document
        new_embedding = self.model.encode([text_content])
        
        # Add to existing embeddings
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, new_embedding])
        else:
            self.embeddings = new_embedding
        
        # Add metadata
        self.document_metadata.append(metadata)
        
        # Update FAISS index
        if self.index is not None:
            # Normalize new embedding
            normalized_embedding = new_embedding / np.linalg.norm(new_embedding, axis=1, keepdims=True)
            self.index.add(normalized_embedding.astype('float32'))
        else:
            # Create new index
            dimension = new_embedding.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            normalized_embedding = new_embedding / np.linalg.norm(new_embedding, axis=1, keepdims=True)
            self.index.add(normalized_embedding.astype('float32'))
        
        # Save updated embeddings
        self.save_embeddings(save_dir)
        
        print(f"Added new document: {metadata.get('filename', 'Unknown')}")
    
    def get_document_stats(self):
        """
        Get statistics about the document collection
        """
        if not self.document_metadata:
            return {}
        
        total_docs = len(self.document_metadata)
        avg_content_length = np.mean([doc['content_length'] for doc in self.document_metadata])
        
        return {
            'total_documents': total_docs,
            'average_content_length': avg_content_length,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'documents': [doc['filename'] for doc in self.document_metadata]
        }

def main():
    """
    Main function to create embeddings for cleaned ML Books
    """
    # Initialize vector embeddings system
    ve = VectorEmbeddings()
    
    # Load cleaned documents
    cleaned_dir = "cleaned ML Books"
    ve.load_cleaned_documents(cleaned_dir)
    
    if not ve.documents:
        print("No documents found. Please run nlp_pipeline.py first.")
        return
    
    # Create embeddings
    ve.create_embeddings()
    
    # Save embeddings
    ve.save_embeddings()
    
    # Print statistics
    stats = ve.get_document_stats()
    print("\nDocument Collection Statistics:")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Average content length: {stats['average_content_length']:.0f} characters")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    
    # Test search functionality
    print("\nTesting search functionality...")
    test_query = "machine learning algorithms"
    results = ve.search_similar_documents(test_query, top_k=3)
    
    print(f"\nTop 3 results for query: '{test_query}'")
    for result in results:
        print(f"Rank {result['rank']}: {result['metadata']['filename']} (Score: {result['similarity_score']:.4f})")

if __name__ == "__main__":
    main()