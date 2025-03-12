import pickle
from typing import List, Optional, Dict
from pathlib import Path
import pandas as pd
from .question import Question, Choice

class NeurologyDataset:
    """
    Handles loading and managing neurology exam questions.
    """
    def __init__(self, data_path: str):
        """
        Initialize the dataset from the given path.
        
        Args:
            data_path: Path to the pickle file containing the questions
        """
        self.data_path = data_path
        self.questions = self._load_questions(data_path)
    
    def _load_questions(self, filepath: str) -> List[Question]:
        """
        Load questions from a pickle file.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_non_image_questions(self) -> List[Question]:
        """
        Filter questions to return only those without images.
        """
        return [q for q in self.questions if not q.contains_image]
    
    def get_image_questions(self) -> List[Question]:
        """
        Filter questions to return only those with images.
        """
        return [q for q in self.questions if q.contains_image]
    
    def get_questions_by_category(self, category: str) -> List[Question]:
        """
        Filter questions by their category.
        """
        return [q for q in self.questions if q.category == category]
    
    def get_question(self, exam: str, number: int) -> Optional[Question]:
        """
        Get a specific question by exam and number.
        """
        for question in self.questions:
            if question.exam == exam and question.number == number:
                return question
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert questions to a pandas DataFrame.
        """
        rows = []
        for q in self.questions:
            # Convert choices to a formatted string to store in DataFrame
            choices_str = ' | '.join([f"{c.label}: {c.text}" for c in q.choices])
            
            # Create a dictionary for this question
            row = {
                'exam': q.exam,
                'number': q.number,
                'text': q.text,
                'choices': choices_str,
                'category': q.category,
                'contains_image': q.contains_image,
                'question_image_paths': str(q.question_image_paths) if q.question_image_paths else '',
                'correct_answers': ','.join(sorted(q.correct_answers)) if q.correct_answers else ''
            }
            rows.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Set column order
        column_order = [
            'exam', 
            'number', 
            'text', 
            'choices',
            'category',
            'contains_image',
            'question_image_paths',
            'correct_answers'
        ]
        
        return df[column_order]
    
    def save_questions(self, filepath: str) -> None:
        """Save questions to a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.questions, f)