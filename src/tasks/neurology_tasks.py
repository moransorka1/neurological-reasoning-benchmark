from typing import Dict, Any, List, Optional, Tuple
from crewai import Task
from ..agents.neurology_agents import (
    ClassifyQuestionOutput, 
    OptimizeQuestionOutput,
    RetrievalOutput,
    SynthesizeAnswerOutput,
    ValidatorOutput
)
import json
import yaml

class NeurologyTaskManager:
    """
    Manages the creation and configuration of neurology-related tasks.
    """
    def __init__(self, tasks_config_file: str):
        """
        Initialize the task manager from a configuration file.
        
        Args:
            tasks_config_file: Path to the YAML file containing task configurations
        """
        self.tasks_config = self._load_yaml_config(tasks_config_file)
    
    def _load_yaml_config(self, file_path: str) -> dict:
        """
        Load configuration from a YAML file.
        """
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found: {file_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return {}
    
    def create_classify_question_task(self, question_complexity_classifier_agent: Any) -> Task:
        """
        Create a task for classifying the complexity of a question.
        
        Args:
            question_complexity_classifier_agent: Agent for question classification
            
        Returns:
            CrewAI Task object
        """
        config = self.tasks_config.get('classify_question_task', {})
        
        return Task(
            description=config.get('description', ''),
            agent=question_complexity_classifier_agent,
            expected_output=config.get('expected_output', ''),
            output_pydantic=ClassifyQuestionOutput
        )
    
    def create_optimize_question_task(self, question_interpreter_agent: Any, context: List[Task] = None) -> Task:
        """
        Create a task for optimizing questions for retrieval.
        
        Args:
            question_interpreter_agent: Agent for question interpretation
            context: Context tasks
            
        Returns:
            CrewAI Task object
        """
        config = self.tasks_config.get('optimize_question_task', {})
        
        return Task(
            description=config.get('description', ''),
            agent=question_interpreter_agent,
            expected_output=config.get('expected_output', ''),
            output_pydantic=OptimizeQuestionOutput,
            context=context
        )
    
    def create_retrieve_store_knowledge_task(self, research_retrieval_agent: Any, context: List[Task] = None) -> Task:
        """
        Create a task for retrieving and storing knowledge.
        
        Args:
            research_retrieval_agent: Agent for retrieval
            context: Context tasks
            
        Returns:
            CrewAI Task object
        """
        config = self.tasks_config.get('retrieve_store_knowledge_task', {})
        
        return Task(
            description=config.get('description', ''),
            agent=research_retrieval_agent,
            expected_output=config.get('expected_output', ''),
            output_pydantic=RetrievalOutput,
            context=context
        )
    
    def create_synthesize_answer_task(self, answer_synthesis_agent: Any, context: List[Task] = None) -> Task:
        """
        Create a task for synthesizing answers.
        
        Args:
            answer_synthesis_agent: Agent for answer synthesis
            context: Context tasks
            
        Returns:
            CrewAI Task object
        """
        config = self.tasks_config.get('synthesize_answer_task', {})
        
        return Task(
            description=config.get('description', ''),
            agent=answer_synthesis_agent,
            expected_output=config.get('expected_output', ''),
            output_pydantic=SynthesizeAnswerOutput,
            context=context
        )
        
    def create_validate_answer_task(self, validator_agent: Any, context: List[Task] = None) -> Task:
        """
        Create a task for validating answers.
        
        Args:
            validator_agent: Agent for answer validation
            context: Context tasks
            
        Returns:
            CrewAI Task object
        """
        config = self.tasks_config.get('validate_answer_task', {})
        
        return Task(
            description=config.get('description', ''),
            agent=validator_agent,
            expected_output=config.get('expected_output', ''),
            output_pydantic=ValidatorOutput,
            context=context
        )

def validate_retrieval_output(result: Any) -> Tuple[bool, Any]:
    """
    Validates the retrieval output by attempting structured parsing.
    
    Args:
        result: Result from retrieval task
        
    Returns:
        Tuple of (valid, result)
    """
    try:
        # First attempt: Direct Pydantic validation
        output = RetrievalOutput.model_validate(result)
        return True, output

    except Exception:
        try:
            # Second attempt: Parse as JSON and then validate
            if hasattr(result, 'raw'):
                result_str = result.raw
            else:
                result_str = str(result)
                
            response_json = json.loads(result_str)
            output = RetrievalOutput(**response_json)
            return True, output

        except Exception:
            # Final fallback: Return the raw response
            return False, result

def extract_final_answer(task_output: Any) -> str:
    """
    Extract the final answer from a task output.
    
    Args:
        task_output: Output from a CrewAI task
        
    Returns:
        The extracted answer
    """
    # If it's already a string with raw JSON
    if hasattr(task_output, 'raw'):
        try:
            data = json.loads(task_output.raw)
            # First check for final_answer (from validator)
            if 'final_answer' in data:
                return data.get('final_answer', '')
            # Otherwise use selected_answer (from synthesis)
            return data.get('selected_answer', '')
        except json.JSONDecodeError:
            pass

    # If it's a TaskOutput object with direct attribute access
    elif hasattr(task_output, 'final_answer'):
        return task_output.final_answer
    elif hasattr(task_output, 'selected_answer'):
        return task_output.selected_answer

    # If it's a dictionary
    elif isinstance(task_output, dict):
        if 'final_answer' in task_output:
            return task_output.get('final_answer', '')
        return task_output.get('selected_answer', '')

    # Fallback
    return "Unable to determine"