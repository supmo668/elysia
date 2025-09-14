#!/usr/bin/env python3
"""
Basic example demonstrating Elysia's memory system functionality.
This example shows how to use the Environment class for memory management.
"""

from elysia.tree.objects import Environment

def main():
    print("=== Elysia Memory System Example ===\n")
    
    # Create a new environment
    env = Environment()
    
    # Example 1: Adding objects to memory
    print("1. Adding objects to memory:")
    
    # Add some query results
    query_results = [
        {"id": "doc1", "title": "AI Introduction", "content": "Artificial intelligence basics..."},
        {"id": "doc2", "title": "Machine Learning", "content": "ML fundamentals..."}
    ]
    query_metadata = {
        "query_text": "introduction to AI",
        "search_type": "semantic", 
        "timestamp": "2024-01-01T10:00:00",
        "collection": "ai_docs"
    }
    
    env.add_objects("query", "ai_collection", query_results, query_metadata)
    print(f"Added {len(query_results)} query results to memory")
    
    # Add some aggregation results  
    agg_results = [{"total_docs": 150, "categories": ["AI", "ML", "NLP"]}]
    agg_metadata = {
        "operation": "count", 
        "timestamp": "2024-01-01T10:05:00",
        "collection": "ai_docs"
    }
    
    env.add_objects("aggregate", "ai_collection", agg_results, agg_metadata)
    print(f"Added {len(agg_results)} aggregation results to memory")
    print()
    
    # Example 2: Retrieving objects from memory
    print("2. Retrieving objects from memory:")
    
    # Get query results
    stored_queries = env.find("query", "ai_collection")
    print(f"Retrieved {len(stored_queries)} query result sets")
    
    if stored_queries:
        print(f"Query metadata: {stored_queries[0]['metadata']}")
        print(f"Number of objects: {len(stored_queries[0]['objects'])}")
        print(f"First object: {stored_queries[0]['objects'][0]}")
    print()
    
    # Get aggregation results
    stored_aggs = env.find("aggregate", "ai_collection")
    print(f"Retrieved {len(stored_aggs)} aggregation result sets")
    
    if stored_aggs:
        print(f"Aggregation result: {stored_aggs[0]['objects'][0]}")
    print()
    
    # Example 3: Memory statistics
    print("3. Memory statistics:")
    print(f"Is environment empty? {env.is_empty()}")
    print(f"Available tools in memory: {list(env.environment.keys())}")
    
    # Count total objects
    total_objects = 0
    for tool_name, tool_results in env.environment.items():
        if tool_name != "SelfInfo":  # Exclude self-info
            for result_name, result_list in tool_results.items():
                for result in result_list:
                    total_objects += len(result.get('objects', []))
    
    print(f"Total objects in memory: {total_objects}")
    print()
    
    # Example 4: Working with hidden environment
    print("4. Hidden environment (not visible to LLM):")
    
    # Add some internal state
    env.hidden_environment['user_preferences'] = {
        'preferred_language': 'python',
        'experience_level': 'intermediate'
    }
    env.hidden_environment['session_stats'] = {
        'queries_made': 3,
        'start_time': '2024-01-01T09:00:00'
    }
    
    print(f"Hidden environment keys: {list(env.hidden_environment.keys())}")
    print(f"User preferences: {env.hidden_environment['user_preferences']}")
    print()
    
    # Example 5: JSON serialization
    print("5. JSON serialization:")
    
    env_json = env.to_json()
    print(f"Serialized environment keys: {list(env_json.keys())}")
    print(f"Environment contains {len(env_json['environment'])} tool types")
    print(f"Hidden environment has {len(env_json['hidden_environment'])} entries")
    
    # Reconstruct environment from JSON
    env_reconstructed = Environment.from_json(env_json)
    print(f"Reconstructed environment is empty: {env_reconstructed.is_empty()}")

if __name__ == "__main__":
    main()