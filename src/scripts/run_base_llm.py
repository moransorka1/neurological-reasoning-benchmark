#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import time
import json

# Add the src directory to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import NeurologyDataset
from src.models.openai_model import AzureOpenAIModel
from src.models.ollama_model import OllamaModel
from src.utils.helpers import extract_model_answer
from src.evaluation.metrics import EvaluationMetrics

def create_prompt(question_text, choices_text):
    """Create a well-formatted prompt for the LLM."""
    prompt = [
        "You are a medical professional specializing in medicine. Answer the following multiple-choice question.",
        "Answer the following multiple-choice question from Neurology exam.",
        "Please analyze each option carefully and choose the most appropriate answer.",
        "Provide your answer in the following format:",
        "1. Selected answer: [letter only - a, b, c, or d]",
        "2. Confidence: [0-1]",
        "3. Explanation: [your reasoning]",
        "\nQuestion:",
        question_text,
        "\nChoices:"
    ]
    
    prompt.append(choices_text)
    
    return "\n".join(prompt)

def run_experiment(model, dataset, output_dir, max_questions=None):
    """Run the base LLM experiment on a dataset."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get non-image questions from the dataset
    questions = dataset.get_non_image_questions()
    
    # Limit the number of questions if specified
    if max_questions is not None and max_questions > 0:
        questions = questions[:max_questions]
    
    answers = []
    
    for question in tqdm(questions, desc="Processing questions"):
        # Format question for the model
        choices_text = "\n".join([f"{choice.label}. {choice.text}" for choice in question.choices])
        prompt = create_prompt(question.text, choices_text)
        
        try:
            # Query the model
            response = model.query(prompt)
            
            # Extract model answer
            model_response = response.get("text", "")
            model_answer = extract_model_answer(model_response)
            
            # Determine if answer is correct
            is_correct = model_answer in question.correct_answers if model_answer else False
            
            # Create result entry
            question_answer = {
                'exam': question.exam,
                'number': question.number,
                'selected_answer': model_answer,
                'is_correct': is_correct,
                'model': response.get("model", "unknown"),
                'model_usage': response.get("usage", {}),
                'response_content': model_response,
                'text': question.text,
                'choices': [{'label': c.label, 'text': c.text} for c in question.choices],
                'category': question.category,
                'contains_image': question.contains_image,
                'question_image_paths': str(question.question_image_paths) if question.question_image_paths else None,
                'correct_answers': ','.join(sorted(question.correct_answers)),       
            }
            
            answers.append(question_answer)
            
        except Exception as e:
            print(f'Error in question number {question.number} exam {question.exam}')
            print(e)
            
            # Create error entry
            question_answer = {
                'exam': question.exam,
                'number': question.number,
                'selected_answer': None,
                'is_correct': None,
                'model': None,
                'model_usage': None,
                'response_content': f"Error: {str(e)}",
                'text': question.text,
                'choices': [{'label': c.label, 'text': c.text} for c in question.choices],
                'category': question.category,
                'contains_image': question.contains_image,
                'question_image_paths': str(question.question_image_paths) if question.question_image_paths else None,
                'correct_answers': ','.join(sorted(question.correct_answers)),       
            }
            
            answers.append(question_answer)
        
        # Save results after each question in case of interruption
        results_df = pd.DataFrame(answers)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(output_dir, f"base_llm_results_{timestamp}.csv")
        results_df.to_csv(results_file)
    
    # Generate and save metrics
    metrics = EvaluationMetrics(results_df)
    basic_metrics = metrics.calculate_basic_metrics()
    category_metrics = metrics.calculate_category_metrics()
    
    # Combine metrics
    combined_metrics = {
        "basic_metrics": basic_metrics,
        "category_metrics": category_metrics
    }
    
    # Save metrics to JSON
    metrics_file = os.path.join(output_dir, f"base_llm_metrics_{timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump(combined_metrics, f, indent=2)
    
    # Generate plots
    plot_file = os.path.join(output_dir, f"category_accuracy_{timestamp}.png")
    metrics.plot_accuracy_by_category(output_file=plot_file)
    
    print(f"Experiment completed. Results saved to {results_file}")
    
    return results_df, combined_metrics

def main():
    parser = argparse.ArgumentParser(description='Run base LLM experiments on neurology questions')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset pickle file')
    parser.add_argument('--model', type=str, required=True, choices=['azure', 'ollama'], help='Model type to use')
    parser.add_argument('--output_dir', type=str, default='results/base_llm', help='Directory to save results')
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
    
    # Create model
    if args.model == 'azure':
        if not all([args.azure_endpoint, args.azure_key, args.deployment]):
            raise ValueError("Azure OpenAI requires endpoint, key, and deployment arguments")
        
        model = AzureOpenAIModel(
            endpoint=args.azure_endpoint,
            api_key=args.azure_key,
            api_version=args.azure_version,
            deployment=args.deployment
        )
        print(f"Using Azure OpenAI model {args.deployment}")
        
    elif args.model == 'ollama':
        if not args.ollama_model:
            raise ValueError("Ollama requires model name argument")
        
        model = OllamaModel(
            model_name=args.ollama_model,
            host=args.ollama_host
        )
        print(f"Using Ollama model {args.ollama_model}")
    
    # Run experiment
    results_df, metrics = run_experiment(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        max_questions=args.max_questions
    )
    
    # Print summary
    print("\nExperiment Summary:")
    print(f"Total questions: {metrics['basic_metrics']['total_questions']}")
    print(f"Correct answers: {metrics['basic_metrics']['correct_answers']}")
    print(f"Accuracy: {metrics['basic_metrics']['accuracy']:.2%}")

if __name__ == "__main__":
    main()