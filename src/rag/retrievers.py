import os
import json
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
import asyncio
from .vector_db import VectorDatabase
from ..data.question import Question

class RAGRetriever:
    """
    Handles RAG retrieval operations for neurology questions.
    """
    def __init__(self, vector_db: VectorDatabase):
        """
        Initialize the RAG retriever.
        
        Args:
            vector_db: Vector database instance for retrievals
        """
        self.vector_db = vector_db
    
    def retrieve_for_question(self, question: Question, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant content for a specific question.
        
        Args:
            question: Question object
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieval result dictionaries
        """
        # Format the question
        query = question.format_text()
        
        # Perform retrieval
        results = self.vector_db.similarity_search(query, k=top_k)
        
        return results
    
    def save_retrievals(self, question: Question, retrievals: List[Dict[str, Any]], 
                       output_dir: str, top_k: int):
        """
        Save retrieval results to a file.
        
        Args:
            question: Question object
            retrievals: List of retrieval results
            output_dir: Directory to save the results
            top_k: The top_k value used for retrieval
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a subdirectory for the specific top_k value
        top_k_dir = os.path.join(output_dir, f'top_{top_k}')
        os.makedirs(top_k_dir, exist_ok=True)
        
        # Create filename based on exam and question number
        filename = f"{question.exam}_q{question.number}.json"
        
        # Extract content from retrievals
        retrieval_contents = [r["content"] for r in retrievals]
        
        # Create output data
        output_data = {
            'exam': question.exam,
            'question_number': question.number,
            'retrievals': retrieval_contents
        }
        
        # Save retrievals
        with open(os.path.join(top_k_dir, filename), 'w') as f:
            json.dump(output_data, f, indent=2)

class RAGManager:
    """
    Manages RAG operations for a dataset of questions.
    """
    def __init__(self, vector_db: VectorDatabase, output_base_dir: str):
        """
        Initialize the RAG manager.
        
        Args:
            vector_db: Vector database instance
            output_base_dir: Base directory for saving retrieval results
        """
        self.vector_db = vector_db
        self.retriever = RAGRetriever(vector_db)
        self.output_base_dir = output_base_dir
    
    def process_questions(self, questions: List[Question], top_k_values: List[int]):
        """
        Process a list of questions and save retrievals for different top_k values.
        
        Args:
            questions: List of Question objects
            top_k_values: List of top_k values to use for retrieval
        """
        for top_k in top_k_values:
            output_dir = os.path.join(self.output_base_dir, 'full_question')
            
            for question in questions:
                # Skip questions with images if necessary
                if question.contains_image:
                    continue
                
                # Get retrievals
                retrievals = self.retriever.retrieve_for_question(question, top_k=top_k)
                
                # Save retrievals
                self.retriever.save_retrievals(question, retrievals, output_dir, top_k)
    
    @staticmethod
    def load_retrievals(exam: str, question_number: int, top_k: int, 
                      base_dir: str, retrieval_type: str = 'full_question') -> List[str]:
        """
        Load saved retrievals for a specific question.
        
        Args:
            exam: Exam identifier
            question_number: Question number
            top_k: Top-k value of the retrievals to load
            base_dir: Base directory where retrievals are stored
            retrieval_type: Type of retrieval ('full_question' or 'question')
            
        Returns:
            List of retrieved context strings
        """
        file_path = os.path.join(base_dir, retrieval_type, f'top_{top_k}', f"{exam}_q{question_number}.json")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data['retrievals']
        except FileNotFoundError:
            print(f"No retrievals found for exam {exam}, question {question_number}, top_k {top_k}")
            return []