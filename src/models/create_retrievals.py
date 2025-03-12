#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from tqdm import tqdm

# Add the src directory to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import NeurologyDataset
from src.rag.vector_db import VectorDatabase
from src.rag.retrievers import RAGManager

def main():
    parser = argparse.ArgumentParser(description='Create RAG retrievals for neurology questions')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset pickle file')
    parser.add_argument('--output_dir', type=str, default='retrievals', help='Directory to save retrievals')
    parser.add_argument('--vector_db_dir', type=str, required=True, help='Directory for the vector database')
    parser.add_argument('--embeddings_model', type=str, help='Path to embeddings model')
    parser.add_argument('--top_k_values', type=str, default='5,10,20,30,40', help='Comma-separated list of top_k values')
    parser.add_argument('--include_images', action='store_true', help='Include image questions')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = NeurologyDataset(args.data)
    print(f"Loaded dataset with {len(dataset.questions)} questions")
    
    # Create vector database
    vector_db = VectorDatabase(
        persist_directory=args.vector_db_dir,
        embedding_model_path=args.embeddings_model
    )
    print(f"Vector database loaded from {args.vector_db_dir}")
    
    # Create RAG manager
    rag_manager = RAGManager(
        vector_db=vector_db.vectorstore,
        output_base_dir=args.output_dir
    )
    print(f"RAG manager created with output directory {args.output_dir}")
    
    # Get questions to process
    if args.include_images:
        questions = dataset.questions
    else:
        questions = dataset.get_non_image_questions()
    
    print(f"Processing {len(questions)} questions")
    
    # Parse top_k values
    top_k_values = [int(k) for k in args.top_k_values.split(',')]
    print(f"Using top_k values: {top_k_values}")
    
    # Process questions
    rag_manager.process_questions(questions, top_k_values)
    
    print("Retrievals created and saved successfully")

if __name__ == "__main__":
    main()