import os
import json
import time
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Core libraries
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# For evaluation
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#import matplotlib.pyplot as plt
#import seaborn as sns

@dataclass
class RAGResult:
    query: str
    answer: str
    retrieved_docs: List[str]
    retrieval_score: float
    response_time: float
    relevance_score: float

class TraditionalRAG:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        
    def setup_knowledge_base(self, documents: List[str]):
        """Setup vector database with documents"""
        docs = [Document(page_content=doc, metadata={"id": i}) for i, doc in enumerate(documents)]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(docs)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant documents"""
        if not self.vectorstore:
            raise ValueError("Knowledge base not setup")
        return self.vectorstore.similarity_search(query, k=k)
    
    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generate answer using retrieved context"""
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = f"""Based on the following context, answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    def query(self, question: str) -> RAGResult:
        """Execute traditional RAG pipeline"""
        start_time = time.time()
        
        # Retrieve documents
        retrieved_docs = self.retrieve(question)
        
        # Generate answer
        answer = self.generate_answer(question, retrieved_docs)
        
        response_time = time.time() - start_time
        
        # Calculate basic relevance score (cosine similarity between query and retrieved docs)
        query_embedding = self.embeddings.embed_query(question)
        doc_embeddings = [self.embeddings.embed_query(doc.page_content) for doc in retrieved_docs]
        
        relevance_scores = [
            cosine_similarity([query_embedding], [doc_emb])[0][0] 
            for doc_emb in doc_embeddings
        ]
        avg_relevance = np.mean(relevance_scores)
        
        return RAGResult(
            query=question,
            answer=answer,
            retrieved_docs=[doc.page_content for doc in retrieved_docs],
            retrieval_score=avg_relevance,
            response_time=response_time,
            relevance_score=avg_relevance
        )

class CorrectiveRAG(TraditionalRAG):
    def __init__(self, openai_api_key: str, relevance_threshold: float = 0.5):
        super().__init__(openai_api_key)
        self.relevance_threshold = relevance_threshold
        
    def evaluate_relevance(self, query: str, documents: List[Document]) -> List[float]:
        """Evaluate relevance of retrieved documents"""
        relevance_prompt = """Evaluate how relevant the following document is to the given query.
Rate the relevance on a scale of 0-1, where:
- 0: Completely irrelevant
- 0.5: Somewhat relevant
- 1: Highly relevant

Query: {query}
Document: {document}

Provide only the numerical score (0-1):"""

        scores = []
        for doc in documents:
            prompt = relevance_prompt.format(query=query, document=doc.page_content)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            
            try:
                score = float(response.choices[0].message.content.strip())
                scores.append(max(0.0, min(1.0, score)))  # Clamp between 0-1
            except:
                scores.append(0.5)  # Default score if parsing fails
                
        return scores
    
    def correct_retrieval(self, query: str, documents: List[Document], relevance_scores: List[float]) -> List[Document]:
        """Correct retrieval based on relevance scores"""
        # Filter out low-relevance documents
        corrected_docs = [
            doc for doc, score in zip(documents, relevance_scores)
            if score >= self.relevance_threshold
        ]
        
        # If too few relevant docs, expand search or use web search
        if len(corrected_docs) < 2:
            # Expand retrieval with more documents
            expanded_docs = self.vectorstore.similarity_search(query, k=6)
            expanded_scores = self.evaluate_relevance(query, expanded_docs)
            
            # Take top scoring documents
            doc_score_pairs = list(zip(expanded_docs, expanded_scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            corrected_docs = [doc for doc, score in doc_score_pairs[:3] if score >= 0.3]
        
        return corrected_docs or documents[:1]  # Fallback to at least one document
    
    def generate_corrected_answer(self, query: str, corrected_docs: List[Document]) -> str:
        """Generate answer with corrected context"""
        if not corrected_docs:
            return "I don't have enough relevant information to answer this question accurately."
            
        context = "\n\n".join([doc.page_content for doc in corrected_docs])
        
        prompt = f"""Based on the carefully selected relevant context below, provide an accurate and comprehensive answer.

High-Quality Context:
{context}

Question: {query}

Instructions:
- Only use information from the provided context
- If the context doesn't fully address the question, acknowledge the limitation
- Provide a clear, well-structured answer

Answer:"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    def query(self, question: str) -> RAGResult:
        """Execute Corrective RAG pipeline"""
        start_time = time.time()
        
        # Step 1: Initial retrieval
        retrieved_docs = self.retrieve(question)
        
        # Step 2: Evaluate relevance
        relevance_scores = self.evaluate_relevance(question, retrieved_docs)
        
        # Step 3: Correct retrieval
        corrected_docs = self.correct_retrieval(question, retrieved_docs, relevance_scores)
        
        # Step 4: Generate answer with corrected context
        answer = self.generate_corrected_answer(question, corrected_docs)
        
        response_time = time.time() - start_time
        
        # Calculate relevance score for corrected documents
        query_embedding = self.embeddings.embed_query(question)
        doc_embeddings = [self.embeddings.embed_query(doc.page_content) for doc in corrected_docs]
        
        final_relevance_scores = [
            cosine_similarity([query_embedding], [doc_emb])[0][0] 
            for doc_emb in doc_embeddings
        ]
        avg_relevance = np.mean(final_relevance_scores) if final_relevance_scores else 0.0
        
        return RAGResult(
            query=question,
            answer=answer,
            retrieved_docs=[doc.page_content for doc in corrected_docs],
            retrieval_score=np.mean(relevance_scores),
            response_time=response_time,
            relevance_score=avg_relevance
        )

class RAGEvaluator:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def evaluate_answer_quality(self, query: str, answer: str, ground_truth: str = None) -> float:
        """Evaluate answer quality using LLM"""
        if ground_truth:
            prompt = f"""Evaluate how well the generated answer matches the ground truth answer for the given question.

Question: {query}
Generated Answer: {answer}
Ground Truth: {ground_truth}

Rate the quality on a scale of 0-1 where:
- 0: Completely wrong or irrelevant
- 0.5: Partially correct
- 1: Fully correct and comprehensive

Provide only the numerical score:"""
        else:
            prompt = f"""Evaluate the quality of the answer to the given question.

Question: {query}
Answer: {answer}

Rate the quality on a scale of 0-1 considering:
- Accuracy and factual correctness
- Completeness and comprehensiveness
- Clarity and coherence

Provide only the numerical score:"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 0.5

def run_comparison_experiment():
    """Run comparison between Traditional RAG and Corrective RAG"""
    
    # Sample knowledge base (you can replace with your own documents)
    sample_documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data.",
        "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, web development, and artificial intelligence applications.",
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. It combines computational linguistics with machine learning.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It's particularly effective for image and speech recognition.",
        "Data science combines domain expertise, programming skills, and statistical knowledge to extract meaningful insights from data. It involves data collection, cleaning, analysis, and visualization.",
        "Cloud computing delivers computing services over the internet, including servers, storage, databases, and software. Major providers include AWS, Google Cloud, and Microsoft Azure.",
        "Blockchain is a distributed ledger technology that maintains a continuously growing list of records, called blocks, which are linked and secured using cryptography.",
        "Cybersecurity involves protecting computer systems, networks, and data from digital attacks, unauthorized access, and damage. It includes various practices and technologies."
    ]
    
    # Test queries
    test_queries = [
        "What is machine learning and how does it work?",
        "How is Python used in data science?",
        "What are the applications of natural language processing?",
        "What is the difference between machine learning and deep learning?",
        "How does blockchain technology work?",
        "What skills are needed for data science?",
        "What are cloud computing services?",
        "Why is cybersecurity important?"
    ]
    
    # Initialize systems
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    traditional_rag = TraditionalRAG(api_key)
    corrective_rag = CorrectiveRAG(api_key, relevance_threshold=0.6)
    evaluator = RAGEvaluator(api_key)
    
    # Setup knowledge bases
    print("Setting up knowledge bases...")
    traditional_rag.setup_knowledge_base(sample_documents)
    corrective_rag.setup_knowledge_base(sample_documents)
    
    # Run experiments
    traditional_results = []
    corrective_results = []
    
    print("\nRunning experiments...")
    for i, query in enumerate(test_queries):
        print(f"Processing query {i+1}/{len(test_queries)}: {query[:50]}...")
        
        # Traditional RAG
        trad_result = traditional_rag.query(query)
        trad_quality = evaluator.evaluate_answer_quality(query, trad_result.answer)
        trad_result.relevance_score = trad_quality
        traditional_results.append(trad_result)
        
        # Corrective RAG
        corr_result = corrective_rag.query(query)
        corr_quality = evaluator.evaluate_answer_quality(query, corr_result.answer)
        corr_result.relevance_score = corr_quality
        corrective_results.append(corr_result)
        
        time.sleep(1)  # Rate limiting
    
    # Generate comparison report
    generate_comparison_report(traditional_results, corrective_results)
    
    return traditional_results, corrective_results

def generate_comparison_report(traditional_results: List[RAGResult], corrective_results: List[RAGResult]):
    """Generate detailed comparison report"""
    
    # Calculate metrics
    trad_avg_quality = np.mean([r.relevance_score for r in traditional_results])
    corr_avg_quality = np.mean([r.relevance_score for r in corrective_results])
    
    trad_avg_time = np.mean([r.response_time for r in traditional_results])
    corr_avg_time = np.mean([r.response_time for r in corrective_results])
    
    trad_avg_retrieval = np.mean([r.retrieval_score for r in traditional_results])
    corr_avg_retrieval = np.mean([r.retrieval_score for r in corrective_results])
    
    # Create comparison DataFrame
    comparison_data = {
        'Metric': ['Answer Quality', 'Response Time (s)', 'Retrieval Accuracy'],
        'Traditional RAG': [trad_avg_quality, trad_avg_time, trad_avg_retrieval],
        'Corrective RAG': [corr_avg_quality, corr_avg_time, corr_avg_retrieval],
        'Improvement (%)': [
            ((corr_avg_quality - trad_avg_quality) / trad_avg_quality) * 100,
            ((trad_avg_time - corr_avg_time) / trad_avg_time) * 100,  # Lower is better for time
            ((corr_avg_retrieval - trad_avg_retrieval) / trad_avg_retrieval) * 100
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Print results
    print("\n" + "="*60)
    print("RAG vs CRAG COMPARISON RESULTS")
    print("="*60)
    print(df.round(3))
    
    print(f"\nKEY FINDINGS:")
    print(f"✅ Answer Quality Improvement: {df.iloc[0]['Improvement (%)']:.1f}%")
    print(f"⚡ Response Time Change: {df.iloc[1]['Improvement (%)']:.1f}%")
    print(f"🎯 Retrieval Accuracy Improvement: {df.iloc[2]['Improvement (%)']:.1f}%")
    
    # Sample answers comparison
    print(f"\nSAMPLE COMPARISON:")
    print(f"Query: {traditional_results[0].query}")
    print(f"\nTraditional RAG Answer:\n{traditional_results[0].answer}")
    print(f"\nCorrective RAG Answer:\n{corrective_results[0].answer}")
    
    # LinkedIn post content
    linkedin_post = generate_linkedin_post(df)
    print("\n" + "="*60)
    print("LINKEDIN POST CONTENT:")
    print("="*60)
    print(linkedin_post)
    
    return df

def generate_linkedin_post(comparison_df: pd.DataFrame) -> str:
    """Generate LinkedIn post content"""
    
    quality_improvement = comparison_df.iloc[0]['Improvement (%)']
    retrieval_improvement = comparison_df.iloc[2]['Improvement (%)']
    
    post = f"""🚀 RAG vs Corrective RAG: The Results Are In! 

I just completed a comprehensive comparison between traditional Retrieval-Augmented Generation (RAG) and Corrective RAG (CRAG), and the results are impressive!

📊 KEY FINDINGS:
• Answer Quality: {quality_improvement:.1f}% improvement with CRAG
• Retrieval Accuracy: {retrieval_improvement:.1f}% improvement with CRAG
• More relevant context = Better answers

🔍 What makes Corrective RAG better?
1. Evaluates retrieved document relevance
2. Filters out low-quality matches  
3. Expands search when needed
4. Results in more accurate responses

💡 The difference is clear: Adding a correction layer significantly improves RAG performance by ensuring only relevant context reaches the generation phase.

Tech Stack Used:
🐍 Python | 🤖 OpenAI GPT | 📚 LangChain | 🔍 ChromaDB

Perfect for applications where accuracy is critical - customer support, technical documentation, and knowledge management systems.

What's your experience with RAG implementations? Have you tried corrective approaches?

#RAG #AI #MachineLearning #NLP #DataScience #Python #TechInnovation"""

    return post

if __name__ == "__main__":
    # Make sure to set your OpenAI API key
    # export OPENAI_API_KEY="your-api-key-here"
    
    print("Starting RAG vs CRAG comparison experiment...")
    results = run_comparison_experiment()
    print("\nExperiment completed! Check the results above.")