#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import time
import json
import yaml
from crewai import Crew, LLM
from dataclasses import asdict

# Add the src directory to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import NeurologyDataset
from src.models.openai_model import AzureOpenAIModel
from src.models.ollama_model import OllamaModel
from src.rag.vector_db import VectorDatabase
from src.agents.neurology_agents import (
    QuestionComplexityClassifier,
    QuestionInterpreter,
    ResearchRetrievalAgent,
    AnswerSynthesisAgent
)
from src.tasks.neurology_tasks import (
    NeurologyTaskManager,
    extract_final_answer,
    validate_retrieval_output
)
from src.evaluation.metrics import EvaluationMetrics

def parse_crew_output(crew_output):
    """
    Parse the CrewAI output to extract task results.
    
    Args:
        crew_output: Output from CrewAI kickoff
        
    Returns:
        Dictionary containing parsed results
    """
    tasks_results = []
    
    for task_output in crew_output.tasks_output:
        task_result = {
            "task_name": task_output.name if hasattr(task_output, "name") else "Unknown",
            "agent": task_output.agent if hasattr(task_output, "agent") else "Unknown",
            "raw_output": task_output.raw if hasattr(task_output, "raw") else "",
            "output_format": task_output.output_format if hasattr(task_output, "output_format") else "Unknown"
        }
        tasks_results.append(task_result)
    
    return {
        "tasks_results": tasks_results,
        "raw_output": crew_output.raw if hasattr(crew_output, "raw") else "",
        "token_usage": crew_output.token_usage if hasattr(crew_output, "token_usage") else {}
    }

def save_crew_results(output, file_id, output_dir):
    """
    Save crew results to a file.
    
    Args:
        output: Parsed CrewAI output
        file_id: Identifier for the file
        output_dir: Directory to save results
        
    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"crew_output_{file_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    return filepath

def run_crewai_experiment(
    dataset,
    agents_config_file,
    tasks_config_file,
    vector_db,
    output_dir,
    llm,
    model_name=None,
    max_questions=None
):
    """Run the CrewAI experiment on a dataset."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "crew_logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Get non-image questions from the dataset
    questions = dataset.get_non_image_questions()
    
    # Limit the number of questions if specified
    if max_questions is not None and max_questions > 0:
        questions = questions[:max_questions]
    
    # Load agent configurations
    with open(agents_config_file, 'r') as f:
        agents_config = yaml.safe_load(f)
    
    # Create task manager
    task_manager = NeurologyTaskManager(tasks_config_file)
    
    results = []
    
    # Process each question
    for question in tqdm(questions, desc="Processing questions"):
        # Format the question for processing
        formatted_question = f"{question.text} \nChoices: {chr(10).join([f'{choice.label}. {choice.text}' for choice in question.choices])}"
        
        try:
            # Create agents
            question_complexity_classifier = QuestionComplexityClassifier(
                config=agents_config['question_complexity_classifier'],
                llm=llm
            )
            
            question_interpreter = QuestionInterpreter(
                config=agents_config['question_interpreter'],
                llm=llm
            )
            
            research_retrieval_agent = ResearchRetrievalAgent(
                config=agents_config['research_retrieval_agent'],
                llm=llm,
                vectordb=vector_db,
                model_name=model_name
            )
            
            answer_synthesis_agent = AnswerSynthesisAgent(
                config=agents_config['answer_synthesis_agent'],
                llm=llm,
                model_name=model_name
            )
            
            # Create tasks
            classify_question_task = task_manager.create_classify_question_task(
                question_complexity_classifier_agent=question_complexity_classifier.agent
            )
            
            optimize_question_task = task_manager.create_optimize_question_task(
                question_interpreter_agent=question_interpreter.agent,
                context=[classify_question_task]
            )
            
            retrieve_store_knowledge_task = task_manager.create_retrieve_store_knowledge_task(
                research_retrieval_agent=research_retrieval_agent.agent,
                context=[optimize_question_task]
            )
            
            synthesize_answer_task = task_manager.create_synthesize_answer_task(
                answer_synthesis_agent=answer_synthesis_agent.agent,
                context=[retrieve_store_knowledge_task]
            )
            
            # Create crew
            crew = Crew(
                agents=[
                    question_complexity_classifier.agent,
                    question_interpreter.agent,
                    research_retrieval_agent.agent,
                    answer_synthesis_agent.agent
                ],
                tasks=[
                    classify_question_task,
                    optimize_question_task,
                    retrieve_store_knowledge_task,
                    synthesize_answer_task
                ]
            )
            
            # Run the crew
            crew_output = crew.kickoff(
                inputs={
                    "original_question": formatted_question,
                    "top_k": 20
                }
            )
            
            # Parse crew output
            parsed_output = parse_crew_output(crew_output)
            
            # Save crew results
            file_id = f"{question.exam}_{question.number}"
            output_path = save_crew_results(parsed_output, file_id, logs_dir)
            
            # Extract final answer
            model_answer = extract_final_answer(crew_output.tasks_output[-1])
            
            # Determine if answer is correct
            is_correct = model_answer in question.correct_answers if model_answer else False
            
            # Create result entry
            token_usage = crew_output.token_usage
            total_usage = {
                "total_tokens": token_usage.total_tokens,
                "prompt_tokens": token_usage.prompt_tokens,
                "completion_tokens": token_usage.completion_tokens
            }
            
            question_answer = {
                'exam': question.exam,
                'number': question.number,
                'selected_answer': model_answer,
                'is_correct': is_correct,
                'model': model_name,
                'model_usage': total_usage,
                'response_content': logs_dir,
                'text': question.text,
                'choices': [{'label': c.label, 'text': c.text} for c in question.choices],
                'category': question.category,
                'contains_image': question.contains_image,
                'question_image_paths': str(question.question_image_paths) if question.question_image_paths else None,
                'correct_answers': ','.join(sorted(question.correct_answers)),       
            }
            
            results.append(question_answer)
            
        except Exception as e:
            print(f'Error in question number {question.number} exam {question.exam}')
            print(e)
            
            # Create error entry
            question_answer = {
                'exam': question.exam,
                'number': question.number,
                'selected_answer': None,
                'is_correct': None,
                'model': model_name,
                'model_usage': None,
                'response_content': f"Error: {str(e)}",
                'text': question.text,
                'choices': [{'label': c.label, 'text': c.text} for c in question.choices],
                'category': question.category,
                'contains_image': question.contains_image,
                'question_image_paths': str(question.question_image_paths) if question.question_image_paths else None,
                'correct_answers': ','.join(sorted(question.correct_answers)),       
            }
            
            results.append(question_answer)
        
        # Save results after each question in case of interruption
        results_df = pd.DataFrame(results)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(output_dir, f"agents_{model_name}_results_{timestamp}.csv")
        results_df.to_csv(results_file)
        
        # Allow some time between questions
        time.sleep(2)
    
    # Save final results
    final_results_df = pd.DataFrame(results)
    final_results_file = os.path.join(output_dir, f"agents_{model_name}_final_results.csv")
    final_results_df.to_csv(final_results_file)
    
    # Generate and save metrics
    metrics = EvaluationMetrics(final_results_df)
    basic_metrics = metrics.calculate_basic_metrics()
    category_metrics = metrics.calculate_category_metrics()
    
    # Combine metrics
    combined_metrics = {
        "basic_metrics": basic_metrics,
        "category_metrics": category_metrics
    }
    
    # Save metrics to JSON
    metrics_file = os.path.join(output_dir, f"agents_{model_name}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(combined_metrics, f, indent=2)
    
    # Generate plots
    plot_file = os.path.join(output_dir, f"agents_{model_name}_category_accuracy.png")
    metrics.plot_accuracy_by_category(output_file=plot_file)
    
    print(f"Experiment completed. Results saved to {final_results_file}")
    
    return final_results_df, combined_metrics

def main():
    parser = argparse.ArgumentParser(description='Run CrewAI experiments on neurology questions')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset pickle file')
    parser.add_argument('--model', type=str, required=True, choices=['azure', 'ollama'], help='Model type to use')
    parser.add_argument('--agents_config', type=str, default='config/agents.yaml', help='Path to agents configuration file')
    parser.add_argument('--tasks_config', type=str, default='config/tasks.yaml', help='Path to tasks configuration file')
    parser.add_argument('--output_dir', type=str, default='results/crewai', help='Directory to save results')
    parser.add_argument('--vector_db_dir', type=str, required=True, help='Directory for the vector database')
    parser.add_argument('--embeddings_model', type=str, help='Path to embeddings model')
    parser.add_argument('--max_questions', type=int, default=None, help='Maximum number of questions to process')
    
    # Azure OpenAI specific arguments
    parser.add_argument('--azure_endpoint', type=str, help='Azure OpenAI endpoint')
    parser.add_argument('--azure_key', type=str, help='Azure OpenAI API key')
    parser.add_argument('--azure_version', type=str, default='2024-08-01-preview', help='Azure API version')
    parser.add_argument('--deployment', type=str, help='Azure deployment name')
    
    # Ollama specific arguments
    parser.add_argument('--ollama_model', type=str, help='Ollama model name')
    parser.add_argument('--ollama_host', type=str, default='http://localhost:11434', help='Ollama host URL')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = NeurologyDataset(args.data)
    print(f"Loaded dataset with {len(dataset.questions)} questions")
    
    # Create CrewAI LLM
    if args.model == 'azure':
        if not all([args.azure_endpoint, args.azure_key, args.deployment]):
            raise ValueError("Azure OpenAI requires endpoint, key, and deployment arguments")
        
        llm = LLM(
            api_key=args.azure_key,
            model=f'azure/{args.deployment}',
            base_url=args.azure_endpoint,
            api_version=args.azure_version
        )
        model_name = args.deployment
        
    elif args.model == 'ollama':
        if not args.ollama_model:
            raise ValueError("Ollama requires model name argument")
        
        llm = LLM(
            model=f"ollama/{args.ollama_model}",
            base_url=args.ollama_host,
            temperature=0.7
        )
        model_name = args.ollama_model
    
    # Create vector database
    vector_db = VectorDatabase(
        persist_directory=args.vector_db_dir,
        embedding_model_path=args.embeddings_model
    )
    print(f"Vector database loaded from {args.vector_db_dir}")
    
    # Run experiment
    results_df, metrics = run_crewai_experiment(
        dataset=dataset,
        agents_config_file=args.agents_config,
        tasks_config_file=args.tasks_config,
        vector_db=vector_db.vectorstore,
        output_dir=args.output_dir,
        llm=llm,
        model_name=model_name,
        max_questions=args.max_questions
    )
    
    # Print summary
    print("\nExperiment Summary:")
    print(f"Total questions: {metrics['basic_metrics']['total_questions']}")
    print(f"Correct answers: {metrics['basic_metrics']['correct_answers']}")
    print(f"Accuracy: {metrics['basic_metrics']['accuracy']:.2%}")

if __name__ == "__main__":
    main()