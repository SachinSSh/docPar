from collections import defaultdict
from datasets import Dataset  
import os
import re
import ast
import json
import traceback
import torch
import numpy as np
import pandas as pd
import transformers
import PyPDF2
#import tensorflow as tf 
import spacy
import nltk
import networkx as nx
import multiprocessing
from typing import List, Dict, Any, Optional, Tuple, Union
from torch.nn import functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM,
    EncoderDecoderModel,
    T5ForConditionalGeneration,
    GPT2LMHeadModel,
    Trainer, 
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class CodeLanguageModel:
    """
    language model for code generation and understanding
    """
    def __init__(self, 
                 embedding_model='t5-small', 
                 code_model='codeparrot/codeparrot-small'):
        
        self.code_tokenizer = AutoTokenizer.from_pretrained(code_model)
        self.code_model = AutoModelForCausalLM.from_pretrained(code_model,pad_token_id=self.code_tokenizer.pad_token_id)

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)

        if self.code_tokenizer.pad_token is None:
            self.code_tokenizer.pad_token = self.code_tokenizer.eos_token
            #self.code_model.config.pad_token_id = self.code_tokenizer.pad_token_id

        self.embedding_model = AutoModelForSeq2SeqLM.from_pretrained(embedding_model)
        
        self.code_model.config.pad_token_id = self.code_tokenizer.pad_token_id
        self.code_model.resize_token_embeddings(len(self.code_tokenizer))
        

        self.generation_config = {
            'max_length': 200,
            'do_sample': True,  
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.95,
            'repetition_penalty': 1.2,
            'pad_token_id': self.code_tokenizer.pad_token_id,
        }
    
    
    def generate_code_variants(self, prompt: str, num_variants: int = 3) -> List[str]:
        prompt = (
            f"Write Python code that implements the following functionality:\n"
            f"{prompt}\n"
            f"Provide only the code without explanations."
        )
        
        inputs = self.code_tokenizer(
            prompt, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True,
            max_length= 512
        )

        # Model-specific generation parameters
        gen_kwargs = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'max_new_tokens': 400,
            'num_return_sequences': num_variants,
            'num_beams': num_variants, 
            'early_stopping': True,
            'pad_token_id': self.code_tokenizer.pad_token_id,
            'eos_token_id': self.code_tokenizer.eos_token_id,
            'do_sample': False 
        }

        if 'typical_p' in gen_kwargs: 
            del gen_kwargs['typical_p']
        if 'bad_words_ids' in gen_kwargs:
            del gen_kwargs['bad_words_ids']

        if not isinstance(self.code_model, transformers.T5ForConditionalGeneration):
            gen_kwargs['attention_mask'] = inputs.attention_mask

        outputs = self.code_model.generate(**gen_kwargs)
        
        return [self.code_tokenizer.decode(out, skip_special_tokens=True) 
                for out in outputs]

class AdvancedSemanticEncoder(nn.Module):
    """
    Sophisticated semantic encoding neural network
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class CodeKnowledgeGraph:
    """
    Intelligent knowledge graph for code relationships
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concept_embeddings = {}
        
    def add_code_relationship(self, 
                               source_concept: str, 
                               target_concept: str, 
                               relationship_type: str):
        """
        Add semantic relationship between code concepts
        """
        self.graph.add_edge(source_concept, target_concept, type=relationship_type)
    
    def find_related_concepts(self, 
                               initial_concept: str, 
                               max_depth: int = 2) -> List[str]:
        """
        Find related code concepts through graph traversal
        """
        related_concepts = []
        try:
            neighbors = list(nx.single_source_shortest_path_length(
                self.graph, 
                initial_concept, 
                cutoff=max_depth
            ).keys())
            
            related_concepts = [
                concept for concept in neighbors 
                if concept != initial_concept
            ]
        except Exception as e:
            print(f"Graph traversal error: {e}")
        
        return related_concepts

class CodeSimilarityAnalyzer:
    """
    Advanced code similarity and analysis system
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), 
            stop_words='english',
            max_features=2048,
            token_pattern=r'(?u)\b\w\w+\b|[\{\}\(\)=<>:;,]'  
        )
    
    def calculate_semantic_similarity(self, 
                                      code_snippets: List[str]) -> np.ndarray:
        """
        Calculate comprehensive semantic similarity matrix
        """
        try:
            vectorized_codes = self.vectorizer.fit_transform(code_snippets)
            similarity_matrix = cosine_similarity(vectorized_codes)
            return similarity_matrix
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return np.zeros((len(code_snippets), len(code_snippets)))
    
    def detect_code_patterns(self, code_snippets: List[str]) -> Dict[str, Any]:
        """
        Detect recurring code patterns and structures
        """
        patterns = {
            'common_imports': defaultdict(int),
            'function_patterns': defaultdict(int),
            'loop_structures': defaultdict(int),
            'class_hierarchies': []
        }
        
        for snippet in code_snippets:
            if not self._is_valid_python(snippet):
                continue
                
            try:
                tree = ast.parse(snippet)
                self._analyze_ast(tree, patterns)
            except SyntaxError:
                continue
        
        return patterns
    
    def _is_valid_python(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _analyze_ast(self, tree, patterns):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    patterns['common_imports'][alias.name] += 1
            elif isinstance(node, ast.FunctionDef):
                patterns['function_patterns'][len(node.args.args)] += 1
            elif isinstance(node, (ast.For, ast.While)):
                patterns['loop_structures'][type(node).__name__] += 1
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'methods': [n.name for n in node.body 
                              if isinstance(n, ast.FunctionDef)]
                }
                patterns['class_definitions'].append(class_info)


class AdvancedCodeGenerator:
    """
    Comprehensive code generation system
    """
    def __init__(self):
        self.language_model = CodeLanguageModel()
        self.knowledge_graph = CodeKnowledgeGraph()
        self.similarity_analyzer = CodeSimilarityAnalyzer()
    
    def generate_context_aware_code(self, 
                                    documentation: str, 
                                    context_depth: int = 3) -> List[str]:
        """
        Generate contextually rich and diverse code variants
        """
        code_variants = self.language_model.generate_code_variants(
            documentation, 
            num_variants=context_depth
        )
        
        similarity_matrix = self.similarity_analyzer.calculate_semantic_similarity(
            code_variants
        )
        code_patterns = self.similarity_analyzer.detect_code_patterns(code_variants)
        
        return {
            'variants': code_variants,
            'similarity_matrix': similarity_matrix,
            'detected_patterns': code_patterns
        }

class DocumentationCodeLearner:
    """
    Advanced machine learning-powered documentation code learning system
    """
    def __init__(self):
        self.code_generator = AdvancedCodeGenerator()
        self.multiprocessing_enabled = multiprocessing.cpu_count() > 1
    
    def learn_from_documentation(self, documentation_paths: List[str]) -> Dict[str, Any]:
        learning_results = {}
    
        for path in documentation_paths:
            try:
                doc_content = self._read_documentation(path)
                if not doc_content.strip():
                    raise ValueError("Empty documentation content")
                    
                result = self.code_generator.generate_context_aware_code(doc_content)
                learning_results[path] = result
                
            except Exception as e:
                print(f"Critical error processing {path}: {str(e)}")
                learning_results[path] = {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        return learning_results
    
    def _read_documentation(self, path: str) -> str:
        """
        Read documentation from various sources
        """
        code_spec_pattern = re.compile(
            r'Code Examples.*?(?:```python\n(.*?)```)',
            re.DOTALL | re.IGNORECASE
        )
        encodings = ['utf-8', 'latin-1', 'cp1252'] 

        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read(10000)
                    matches = code_spec_pattern.search(content)
                    if matches:
                        return f"Generate Python code that implements: {matches.group(1)}"
                    return "Create a Python function that processes data efficiently"
            except UnicodeDecodeError:
                continue

        return "Create a Python function to process data efficiently."
         
    
class CodeFineTuner:
    def __init__(self, model:nn.Module):
        self.model = model.code_model  
        self.tokenizer = model.code_tokenizer  

    def fine_tune(self, dataset_path: str):
        """
        Fine-tune on Python code corpus
        """
        code_dataset = self._load_dataset(dataset_path)

        training_args = TrainingArguments(
            output_dir='./results',
            per_device_train_batch_size=4,
            num_train_epochs=3,
            logging_dir='./logs',
            save_strategy='epoch',
            evaluation_strategy='epoch'
        )
        
        trainer = Trainer(
            model=self.code_model,
            args=training_args,
            train_dataset=code_dataset,
            data_collator=self._data_collator
        )
        trainer.train()

    def _load_dataset(self, dataset_path: str) -> Dataset:
        """Load and tokenize code dataset"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            code_samples = [line.strip() for line in f if line.strip()]

        tokenized_samples = []
        for code in code_samples:
            tokens = self.tokenizer(
                code,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            tokenized_samples.append({
                'input_ids': tokens.input_ids.squeeze(),
                'attention_mask': tokens.attention_mask.squeeze()
            })

        return Dataset.from_list(tokenized_samples)

    def _data_collator(self, batch):
        """Custom data collator for code generation"""
        return {
            'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in batch]),
            'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in batch]),
            'labels': torch.stack([torch.tensor(x['input_ids']) for x in batch])
        }
    
def main():
    code_lm = CodeLanguageModel()
    fine_tuner = CodeFineTuner(code_lm) 

    fine_tuner.fine_tune('python_code_corpus.txt')

    if os.path.exists('python_code_corpus.txt'):
        print("Starting fine-tuning...")
        fine_tuner = CodeFineTuner(code_lm)
        fine_tuner.fine_tune('python_code_corpus.txt')
        print("Fine-tuning completed!")
    else:
        print("No fine-tuning corpus found, using base model")

    ### usage
    learner = DocumentationCodeLearner()
    
    documentation_paths = [
        'data_processing.txt',
        'machine_learning.txt'
    ]
    
    results = learner.learn_from_documentation(documentation_paths)
    
    for path, result in results.items():
        print(f"\nResults for {path}:")
        if 'error' in result:
            print(f"  Error: {result['error']}")
            if 'traceback' in result:
                print("  Traceback:", result['traceback'])
        else:
            print("  Code Variants:")
            for i, variant in enumerate(result['variants'], 1):
                print(f"  Variant {i}:\n{variant[:500]}...")
            
            print("\n  Similarity Matrix:")
            print(result['similarity_matrix'])
            
            print("\n  Detected Patterns:")
            for pattern_type, patterns in result['detected_patterns'].items():
                print(f"  {pattern_type}:")
                if isinstance(patterns, list):
                    for item in patterns[:3]:
                        print(f"    - {str(item)[:80]}")
                else:
                    for k, v in list(patterns.items())[:5]:
                        print(f"    {k}: {v}")
    

if __name__ == '__main__':
    main()
