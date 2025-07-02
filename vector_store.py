import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from openai import OpenAI

class VectorStore:
    """Handles document embeddings and similarity search using OpenAI embeddings."""
    
    def __init__(self, api_key: str, storage_file: str = "vector_store.pkl"):
        self.client = OpenAI(api_key=api_key)
        self.storage_file = storage_file
        self.documents = []
        self.embeddings = []
        self.load_from_disk()
    
    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Add documents and generate their embeddings."""
        try:
            # Clear existing data for new upload
            self.documents = []
            self.embeddings = []
            
            # Generate embeddings for all documents
            texts = [doc["content"] for doc in documents]
            embeddings = self._generate_embeddings(texts)
            
            # Store documents and embeddings
            self.documents = documents
            self.embeddings = embeddings
            
            # Save to disk
            self.save_to_disk()
            
        except Exception as e:
            raise Exception(f"Failed to add documents: {str(e)}")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using OpenAI API."""
        try:
            embeddings = []
            # Process in batches to avoid API limits
            batch_size = 100
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, str], float]]:
        """Find the most similar documents to the query."""
        if not self.documents or not self.embeddings:
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = self._generate_embeddings([query])[0]
            
            # Calculate cosine similarities
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((self.documents[i], similarity))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
            
        except Exception as e:
            raise Exception(f"Failed to perform similarity search: {str(e)}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def save_to_disk(self) -> None:
        """Save the vector store to disk."""
        try:
            data = {
                'documents': self.documents,
                'embeddings': self.embeddings
            }
            with open(self.storage_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            # Don't raise error for save failures, just continue
            print(f"Warning: Failed to save vector store: {str(e)}")
    
    def load_from_disk(self) -> None:
        """Load the vector store from disk."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.embeddings = data.get('embeddings', [])
        except Exception as e:
            # Don't raise error for load failures, just start fresh
            print(f"Warning: Failed to load vector store: {str(e)}")
            self.documents = []
            self.embeddings = []
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        return len(self.documents)
