from typing import Dict, Any, List, Optional
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, validator
from .base_agent import BaseNeurologyAgent
import json
import os
from pathlib import Path
import time

# Common Pydantic models for agent inputs/outputs
class ClassifyQuestionOutput(BaseModel):
    original_question: str
    classification: str  # simple | complex
    classification_reasoning: str  # Detailed step-by-step analysis

    class ConfigDict:
        extra = "allow"

class OptimizeQuestionOutput(BaseModel):
    original_question: str
    extracted_concepts: List[str]
    optimized_queries: List[str]

    class ConfigDict:
        extra = "allow"

class MultiQueryRAGInput(BaseModel):
    original_question: str
    retrieval_queries: List[str] = Field(..., description="List of queries to search")
    top_k: int = Field(default=20, description="Results per query")

class Retrieval(BaseModel):
    query: str
    content: str
    relevance_score: float

class RetrievalOutput(BaseModel):
    original_question: str
    retrieval_queries: List[str]

    class ConfigDict:
        extra = "allow"

class SynthesizeAnswerOutput(BaseModel):
    original_question: str
    selected_answer: str  # a | b | c | d
    reasoning: str  # Explanation for answer choice

    class ConfigDict:
        extra = "allow"


class QuestionComplexityClassifier(BaseNeurologyAgent):
    """
    Agent that classifies the complexity of neurology questions.
    """
    def __init__(self, config: Dict[str, Any], llm: Any):
        super().__init__(config, llm)
    
    def _get_tools(self) -> List[Any]:
        # This agent doesn't use specialized tools
        return []


class QuestionInterpreter(BaseNeurologyAgent):
    """
    Agent that interprets and optimizes questions for retrieval.
    """
    def __init__(self, config: Dict[str, Any], llm: Any):
        super().__init__(config, llm)
    
    def _get_tools(self) -> List[Any]:
        # This agent doesn't use specialized tools
        return []


class RAGSearchTool(BaseTool):
    """
    Tool for retrieving relevant information from knowledge base.
    """
    name: str = "RAG Search Tool"
    description: str = "A specialized tool for retrieving relevant neurology information from the knowledge base."
    args_schema: type = MultiQueryRAGInput
    vectordb: Any = None

    def __init__(self, vectordb: Any):
        super().__init__()
        self.vectordb = vectordb

    def _calculate_relevance_score(self, search_results) -> float:
        """Calculate a relevance score for retrieved content."""
        try:
            import numpy as np
            similarities = [np.exp(-score) for _, score in search_results]
            return sum(similarities) / len(similarities) if similarities else 0.5
        except:
            return 0.5  # Default score if calculation fails

    def _process_single_query(self, query: str, top_k: int) -> Retrieval:
        """Retrieve relevant content for a single query."""
        try:
            search_results = self.vectordb.similarity_search_with_score(query, k=top_k)
            content = "\n".join(doc[0].page_content for doc in search_results)
            relevance_score = self._calculate_relevance_score(search_results)

            return Retrieval(
                query=query,
                content=content,
                relevance_score=relevance_score
            )
        except Exception as e:
            raise Exception(f"Error processing query '{query}': {str(e)}")

    def _run(self, original_question: str, retrieval_queries: List[str], top_k: int = 20) -> RetrievalOutput:
        """Execute the RAG retrieval process and store results in a file."""
        try:
            model_name = getattr(self, 'model_name', 'default-model')
            storage_path = f"retrieved_data_{model_name.replace(':','-')}.json"
            retrievals = []
            
            # Extract question text without answer choices if present
            question_only = original_question.split('Choices:')[0] if 'Choices:' in original_question else original_question

            # Process original question
            original_results = self._process_single_query(question_only, top_k)
            retrievals.append(original_results)

            # Process optimized sub-queries
            for query in retrieval_queries:
                retrievals.append(self._process_single_query(query, top_k))

            # Compute confidence metrics
            avg_relevance = sum(r.relevance_score for r in retrievals) / len(retrievals)
            coverage_score = min(len(retrievals) * 20, 100)

            # Save retrieved data to a file
            retrieved_data = {
                "original_question": original_question,
                "retrieval_queries": retrieval_queries,
                "retrieved_passages": [
                    {"query": r.query, "content": r.content, "relevance_score": r.relevance_score}
                    for r in retrievals
                ]
            }
            
            with open(storage_path, "w") as f:
                json.dump(retrieved_data, f, indent=4)
            
            # Also save a timestamped version for logging
            timestr = time.strftime("%Y%m%d-%H%M%S")
            log_dir = Path(f"rag_logs/{model_name.replace(':','-')}")
            log_dir.mkdir(parents=True, exist_ok=True)
            rag_logfile = log_dir / f"retrieved_data_{timestr}.json"
            
            with open(rag_logfile, "w") as f:
                json.dump(retrieved_data, f, indent=4)

            return RetrievalOutput(
                original_question=original_question,
                retrieval_queries=retrieval_queries
            )
        except Exception as e:
            raise Exception(f"Error in neurology RAG search: {str(e)}")


class ResearchRetrievalAgent(BaseNeurologyAgent):
    """
    Agent that retrieves information for answering neurology questions.
    """
    def __init__(self, config: Dict[str, Any], llm: Any, vectordb: Any, model_name: str = None):
        self.vectordb = vectordb
        self.model_name = model_name
        super().__init__(config, llm)
    
    def _get_tools(self) -> List[Any]:
        rag_tool = RAGSearchTool(vectordb=self.vectordb)
        if self.model_name:
            rag_tool.model_name = self.model_name
        return [rag_tool]


class AnswerSynthesisInput(BaseModel):
    original_question: str

class AnswerSynthesisTool(BaseTool):
    """
    Tool that synthesizes answers based on retrieved information.
    """
    name: str = "Answer Synthesis Tool"
    description: str = "Reads retrieved knowledge and answers the given neurology question."
    args_schema: type = AnswerSynthesisInput
    llm: Any = None
    model_name: str = None

    def __init__(self, llm, model_name=None):
        super().__init__()
        self.llm = llm
        self.model_name = model_name
    
    def _run(self, original_question: str) -> SynthesizeAnswerOutput:
        """Reads retrieved knowledge and synthesizes an answer."""
        model_name_safe = self.model_name.replace(':', '-') if self.model_name else 'default-model'
        retrieved_data_file_path = f"retrieved_data_{model_name_safe}.json"
        
        try:
            # Read the retrieved knowledge from file
            with open(retrieved_data_file_path, 'r') as f:
                retrieved_data = f.read()

            system_prompt = """
You are a medical professional specializing in neurology. 
Based on the following context, answer the following exam multiple-choice question.
Please analyze each option carefully and choose the most appropriate answer.
Provide your answer in the following format:
1. Selected answer: [letter only - a, b, c, or d]
2. Explanation: [your reasoning]"""
            
            # Construct prompt for LLM
            prompt = f"""
            ## CONTEXT:
            {retrieved_data}

            ## QUESTION:
            {original_question}
            
            Based on the context, select the most accurate answer choice (a, b, c, or d).
            If there is more than one correct answer, you have to choose ONLY ONE CHOICE that is the most correct one.
            Explain why this choice is correct and why the other options are incorrect.
            Response format (JSON):
            {{
              "selected_answer": "a / b / c / d",
              "reasoning": "Detailed explanation of the answer choice."
            }}
            """

            # Get the answer from the LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            
            response = self.llm.chat(messages=messages, temperature=0.7)
            model_response = response.get("text", "")

            try:
                # Try parsing the response as JSON
                response_json = json.loads(model_response)
                output = SynthesizeAnswerOutput(
                    original_question=original_question,
                    selected_answer=response_json.get('selected_answer', ''),
                    reasoning=response_json.get('reasoning', '')
                )
                return output
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response
                return SynthesizeAnswerOutput(
                    original_question=original_question,
                    selected_answer="Unable to determine",
                    reasoning=model_response
                )
                
        except Exception as e:
            print(f"Error synthesizing answer: {e}")
            return SynthesizeAnswerOutput(
                original_question=original_question,
                selected_answer="Error",
                reasoning=f"An error occurred: {str(e)}"
            )


class AnswerSynthesisAgent(BaseNeurologyAgent):
    """
    Agent that synthesizes answers based on retrieved information.
    """
    def __init__(self, config: Dict[str, Any], llm: Any, model_name: str = None):
        self.model_name = model_name
        super().__init__(config, llm)
    
    def _get_tools(self) -> List[Any]:
        return [AnswerSynthesisTool(llm=self.llm, model_name=self.model_name)]