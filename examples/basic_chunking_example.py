#!/usr/bin/env python3
"""
Basic example demonstrating Elysia's chunking functionality.
This example shows how to use different chunking strategies.
"""

from elysia.tools.retrieval.chunk import Chunker

def main():
    # Sample document text
    sample_text = """
    Artificial intelligence is a broad field of computer science. 
    It focuses on creating systems that can perform tasks typically requiring human intelligence. 
    These tasks include learning, reasoning, problem-solving, perception, and language understanding.
    Machine learning is a subset of artificial intelligence. 
    It enables computers to learn and improve from experience without being explicitly programmed.
    Deep learning, in turn, is a subset of machine learning.
    It uses neural networks with multiple layers to model and understand complex patterns in data.
    Natural language processing is another important area of AI.
    It focuses on the interaction between computers and human language.
    Computer vision enables machines to interpret and understand visual information from the world.
    """
    
    print("=== Elysia Chunking Example ===\n")
    
    # Example 1: Sentence-based chunking
    print("1. Sentence-based chunking (2 sentences per chunk):")
    sentence_chunker = Chunker(chunking_strategy="sentences", num_sentences=2)
    chunks, spans = sentence_chunker.chunk(sample_text.strip())
    
    for i, (chunk, span) in enumerate(zip(chunks, spans)):
        print(f"Chunk {i+1} (chars {span[0]}-{span[1]}):")
        print(f"  {chunk.strip()}")
        print()
    
    print(f"Total chunks created: {len(chunks)}\n")
    print("-" * 60 + "\n")
    
    # Example 2: Token-based chunking
    print("2. Token-based chunking (20 tokens per chunk with 5 token overlap):")
    token_chunker = Chunker(chunking_strategy="fixed", num_tokens=20)
    chunks, spans = token_chunker.chunk_by_tokens(sample_text.strip(), num_tokens=20, overlap_tokens=5)
    
    for i, (chunk, span) in enumerate(zip(chunks, spans)):
        token_count = len(chunk.split())
        print(f"Chunk {i+1} ({token_count} tokens, chars {span[0]}-{span[1]}):")
        print(f"  {chunk.strip()}")
        print()
    
    print(f"Total chunks created: {len(chunks)}\n")
    print("-" * 60 + "\n")
    
    # Example 3: Getting chunk statistics
    print("3. Chunk statistics:")
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Original token count: {sentence_chunker.count_tokens(sample_text)} tokens")
    print(f"Sentence chunks: {len(sentence_chunker.chunk(sample_text)[0])}")
    print(f"Token chunks (20 tokens): {len(token_chunker.chunk_by_tokens(sample_text, num_tokens=20)[0])}")

if __name__ == "__main__":
    main()