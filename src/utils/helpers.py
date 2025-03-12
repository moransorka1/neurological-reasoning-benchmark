import re
from typing import Dict, Any, Optional

def extract_model_answer(response_text: str) -> Optional[str]:
    """
    Extract model answer from response text.
    
    Args:
        response_text: Text response from the model
        
    Returns:
        Extracted answer letter (a, b, c, d) or None if not found
    """
    patterns = [
        r'Answer:\s*([A-Da-d])',         # Answer: X
        r'\*Answer\*:\s*([A-Da-d])',     # *Answer*: X
        r'\*\*Answer:\*\*\s*([A-Da-d])', # **Answer:** X
        r'\*\*Answer\*\*:\s*([A-Da-d])', # **Answer**: X
        r'\*\*Selected answer:\*\*\s*([A-Da-d])', # **Selected answer:** X
        r'Selected answer:\s\*\*([A-Da-d])\*\*', # Selected answer: **X**
        r'Selected answer:\s([A-Da-d])', # Selected answer: X
        r'Selected answer:\s\[([A-Da-d])\]', # Selected answer: [X]
        r'The correct answer is\s*([A-Da-d])',  # The correct answer is X
        r'Therefore.*answer.*?([A-Da-d])[^A-Da-d]*$'  # Therefore...answer...X at the end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    return None

def create_prompt(question_text: str, choices_text: str, system_instruction: str = None) -> str:
    """
    Create a well-formatted prompt for the model.
    
    Args:
        question_text: The question text
        choices_text: The choices text
        system_instruction: Optional system instructions
        
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    if system_instruction:
        prompt_parts.append(system_instruction)
    
    prompt_parts.extend([
        "Please analyze each option carefully and choose the most appropriate answer.",
        "Provide your answer in the following format:",
        "1. Selected answer: [letter only - a, b, c, or d]",
        "2. Explanation: [your reasoning]",
        "\nQuestion:",
        question_text,
        "\nChoices:",
        choices_text
    ])
    
    return "\n".join(prompt_parts)

def create_rag_prompt(question_text: str, choices_text: str, context: str, system_instruction: str = None) -> str:
    """
    Create a well-formatted RAG prompt that includes retrieved context.
    
    Args:
        question_text: The question text
        choices_text: The choices text
        context: Retrieved context from RAG
        system_instruction: Optional system instructions
        
    Returns:
        Formatted prompt string with context
    """
    prompt_parts = []
    
    if system_instruction:
        prompt_parts.append(system_instruction)
    
    prompt_parts.extend([
        f"## Context: {context}",
        "## Question: " + question_text,
        "## Choices: " + choices_text,
        "Provide your answer in the following format:",
        "1. Selected answer: [letter only - a, b, c, or d]",
        "2. Explanation: [your reasoning]"
    ])
    
    return "\n".join(prompt_parts)