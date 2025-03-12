# Neurological Reasoning Benchmark

A comprehensive framework for evaluating the reasoning capabilities of Large Language Models (LLMs) on complex neurological multiple-choice questions.

## Overview

This repository contains the implementation of a multi-agent framework for neurological reasoning as described in our paper. It enables benchmarking LLMs' clinical reasoning abilities using various approaches:

1. **Base LLM evaluation**: Test performance of LLMs on neurological questions without additional context
2. **RAG-enhanced evaluation**: Test LLMs with retrieval-augmented generation from neurological knowledge sources
3. **Multi-agent system**: Test specialized agent-based approaches with decomposed reasoning steps

## Features

- Multiple-choice question dataset from neurological board exams
- Retrieval-augmented generation (RAG) using specialized neurological knowledge
- Multi-agent system with CrewAI for decomposed reasoning
- Support for different model backends (OpenAI/Azure, Ollama)
- Detailed evaluation metrics and analysis
- Configurable agent and task definitions

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/neurological-reasoning-benchmark.git
cd neurological-reasoning-benchmark

# Install dependencies
pip install -r requirements.txt

## Usage

### 1. Base LLM Evaluation

```bash
python scripts/run_base_llm.py \
  --data path/to/questions.pkl \
  --model azure \
  --azure_endpoint "https://your-endpoint.openai.azure.com/" \
  --azure_key "your-api-key" \
  --deployment "gpt-4o" \
  --output_dir results/base_llm
```

For Ollama models:

```bash
python scripts/run_base_llm.py \
  --data path/to/questions.pkl \
  --model ollama \
  --ollama_model "llama3:70b-instruct" \
  --output_dir results/base_llm
```

### 2. Create RAG Retrievals

```bash
python scripts/create_retrievals.py \
  --data path/to/questions.pkl \
  --vector_db_dir path/to/vectordb \
  --embeddings_model "embedding-model-path" \
  --output_dir retrievals \
  --top_k_values 5,10,20,40
```

### 3. Run RAG Experiments

```bash
python scripts/run_rag.py \
  --data path/to/questions.pkl \
  --model azure \
  --azure_endpoint "https://your-endpoint.openai.azure.com/" \
  --azure_key "your-api-key" \
  --deployment "gpt-4o" \
  --vector_db_dir path/to/vectordb \
  --retrieval_dir retrievals \
  --top_k 20 \
  --output_dir results/rag
```

### 4. Run Multi-Agent Experiments

```bash
python scripts/run_crewai.py \
  --data path/to/questions.pkl \
  --model azure \
  --azure_endpoint "https://your-endpoint.openai.azure.com/" \
  --azure_key "your-api-key" \
  --deployment "gpt-4o" \
  --vector_db_dir path/to/vectordb \
  --agents_config config/agents.yaml \
  --tasks_config config/tasks.yaml \
  --output_dir results/crewai
```

## Vector Database Setup

This benchmark uses a ChromaDB vector database with embeddings. To set up with a different knowledge source:

1. Extract text from your knowledge source
2. Split text into chunks with appropriate overlap
3. Create embeddings using a suitable model 
4. Store in ChromaDB format

## Configuration

The system uses YAML configuration files for agent and task definitions:

- `config/agents.yaml`: Defines role, goal, and backstory for each agent
- `config/tasks.yaml`: Defines task descriptions and expected outputs

## Citation

If you use this code in your research, please cite our paper:

```

```
