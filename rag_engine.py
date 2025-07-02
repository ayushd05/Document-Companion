from typing import List, Dict, Tuple
from openai import OpenAI
from vector_store import VectorStore

class RAGEngine:
    """Retrieval-Augmented Generation engine using OpenAI GPT models."""
    
    def __init__(self, api_key: str, vector_store: VectorStore):
        self.client = OpenAI(api_key=api_key)
        self.vector_store = vector_store
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # Do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
    
    def query(self, question: str, max_chunks: int = 5) -> str:
        """Generate an answer based on retrieved relevant documents."""
        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=max_chunks)
            
            if not relevant_docs:
                return "I don't have any relevant information to answer your question. Please make sure you have uploaded and processed documents first."
            
            # Prepare context from retrieved documents
            context = self._prepare_context(relevant_docs)
            
            # Generate response using GPT
            response = self._generate_response(question, context)
            
            return response
            
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def _prepare_context(self, relevant_docs: List[Tuple[Dict[str, str], float]]) -> str:
        """Prepare context string from retrieved documents."""
        context_parts = []
        
        for i, (doc, similarity) in enumerate(relevant_docs):
            source = doc.get('source', 'Unknown')
            content = doc.get('content', '')
            
            # Add document info
            context_parts.append(f"Document {i+1} (Source: {source}):")
            context_parts.append(content)
            context_parts.append("")  # Add empty line for separation
        
        return "\n".join(context_parts)
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate response using OpenAI GPT model."""
        try:
            system_prompt = """You are a helpful assistant that answers questions based on the provided document context. 

Instructions:
1. Use only the information provided in the context to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be specific and cite which document(s) you're referencing when possible
4. Provide clear, concise, and helpful answers
5. If asked about something not in the documents, politely explain that the information is not available in the uploaded documents

The context below contains excerpts from documents that may be relevant to the user's question."""

            user_prompt = f"""Context from documents:
{context}

User question: {question}

Please answer the question based on the provided context."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def get_relevant_sources(self, question: str, max_chunks: int = 3) -> List[str]:
        """Get the sources of documents most relevant to the question."""
        try:
            relevant_docs = self.vector_store.similarity_search(question, k=max_chunks)
            sources = []
            
            for doc, similarity in relevant_docs:
                source = doc.get('source', 'Unknown')
                if source not in sources:
                    sources.append(source)
            
            return sources
            
        except Exception as e:
            return []
