from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

@dataclass
class Choice:
    """
    Represents a choice in a multiple-choice question.
    """
    label: str
    text: str
    image_path: Optional[Path] = None

@dataclass
class Question:
    """
    Represents a multiple-choice question in a neurology exam.
    """
    exam: str
    number: int
    text: str
    choices: List[Choice]
    category: str
    contains_image: bool
    question_image_paths: Optional[List[Path]] = None
    correct_answers: Set[str] = field(default_factory=set)
    
    def format_text(self) -> str:
        """
        Formats the question text and choices as a single string.
        """
        formatted_text = [self.text.strip()]
        for choice in self.choices:
            formatted_text.append(f"{choice.label}. {choice.text}")
        return "\n".join(formatted_text)