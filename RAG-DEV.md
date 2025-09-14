# RAG Development Guide for Elysia

This document provides comprehensive guidance for developers looking to contribute to and customize Elysia's RAG (Retrieval-Augmented Generation) system. Elysia is built with modularity in mind, allowing for extensive customization of its core RAG components.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Chunking and Indexing Mechanisms](#chunking-and-indexing-mechanisms)
3. [Retrieval System Customization](#retrieval-system-customization)
4. [Continuous Chat Memory](#continuous-chat-memory)
5. [Development Setup](#development-setup)
6. [Testing Guidelines](#testing-guidelines)
7. [Contributing Guidelines](#contributing-guidelines)

## Architecture Overview

Elysia's RAG system consists of several key components:

- **Chunking & Indexing**: Document preprocessing and chunk creation (`elysia/tools/retrieval/chunk.py`)
- **Retrieval System**: Vector search and hybrid retrieval (`elysia/tools/retrieval/query.py`)
- **Memory Management**: Conversation context and environment state (`elysia/tree/objects.py`)
- **Preprocessing**: Collection analysis and metadata generation (`elysia/preprocessing/collection.py`)
- **Vector Database Integration**: Primarily Weaviate with extensible architecture

The system follows a tree-based decision making approach where agents communicate through structured environments and can utilize various tools for retrieval, aggregation, and generation tasks.

## Chunking and Indexing Mechanisms

### Overview

The chunking system is responsible for breaking down large documents into smaller, manageable pieces that can be effectively vectorized and retrieved. Elysia supports multiple chunking strategies and creates parallel collections in Weaviate for optimized retrieval.

### Key Components

#### 1. Chunker Class (`elysia/tools/retrieval/chunk.py`)

The `Chunker` class provides the core chunking functionality:

```python
from elysia.tools.retrieval.chunk import Chunker

# Initialize chunker with different strategies
chunker = Chunker(
    chunking_strategy="sentences",  # or "fixed" for token-based
    num_tokens=256,                 # for token-based chunking
    num_sentences=5                 # for sentence-based chunking
)

# Chunk a document
document = "Your long document text here..."
chunks, spans = chunker.chunk(document)
```

#### 2. Chunking Strategies

**Sentence-based Chunking:**
```python
def chunk_by_sentences(
    self,
    document: str,
    num_sentences: int | None = None,
    overlap_sentences: int = 1,
) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Chunks document by sentences using spaCy for sentence detection.
    Returns chunks and their character span annotations.
    """
```

**Token-based Chunking:**
```python
def chunk_by_tokens(
    self, 
    document: str, 
    num_tokens: int | None = None, 
    overlap_tokens: int = 32
) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Chunks document by tokens with configurable overlap.
    Uses spaCy for tokenization.
    """
```

### Customizing Chunking

#### Adding New Chunking Strategies

To add a new chunking strategy:

1. **Extend the Chunker class:**

```python
class CustomChunker(Chunker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom initialization
    
    def chunk_by_paragraphs(
        self, 
        document: str, 
        max_paragraph_size: int = 1000
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """Custom paragraph-based chunking strategy."""
        paragraphs = document.split('\n\n')
        chunks = []
        spans = []
        current_pos = 0
        
        for paragraph in paragraphs:
            if len(paragraph) <= max_paragraph_size:
                chunks.append(paragraph)
                spans.append((current_pos, current_pos + len(paragraph)))
            else:
                # Split large paragraphs using existing token method
                sub_chunks, sub_spans = self.chunk_by_tokens(
                    paragraph, num_tokens=max_paragraph_size//4
                )
                for chunk, (start, end) in zip(sub_chunks, sub_spans):
                    chunks.append(chunk)
                    spans.append((current_pos + start, current_pos + end))
            
            current_pos += len(paragraph) + 2  # Account for \n\n
        
        return chunks, spans
    
    def chunk(self, document: str) -> tuple[list[str], list[tuple[int, int]]]:
        if self.chunking_strategy == "paragraphs":
            return self.chunk_by_paragraphs(document)
        else:
            return super().chunk(document)
```

2. **Update the AsyncCollectionChunker:**

```python
class CustomAsyncCollectionChunker(AsyncCollectionChunker):
    def __init__(self, collection_name: str, chunking_strategy: str = "paragraphs"):
        super().__init__(collection_name)
        self.chunker = CustomChunker(chunking_strategy, num_sentences=5)
```

#### Customizing Vectorization

The chunking system automatically inherits vectorization settings from the parent collection. To customize vectorization for chunks:

```python
async def get_custom_vectoriser(
    self, content_field: str, client: WeaviateAsyncClient
) -> _VectorConfigCreate:
    """Custom vectorizer configuration for chunks."""
    return Configure.Vectors.text2vec_openai(
        model="text-embedding-3-large",
        dimensions=3072,
        source_properties=[content_field],
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric="cosine",
            ef_construction=256,
            max_connections=32,
            quantizer=Configure.VectorIndex.Quantizer.pq(
                training_limit=50000,
                segments=256
            )
        )
    )
```

#### Chunk Size Optimization

The system automatically determines when chunking is needed based on document size:

```python
def _evaluate_needs_chunking(
    self,
    display_type: str,
    query_type: str,
    schema: dict,
    threshold: int = 400,  # Customize this threshold
) -> bool:
    content_field, content_len = self._evaluate_content_field(schema["fields"])
    
    return (
        content_field is not None
        and content_len > threshold
        and query_type != "filter_only"
        and display_type == "document"
    )
```

To customize the threshold or add conditions:

```python
class CustomQuery(Query):
    def _evaluate_needs_chunking(self, display_type, query_type, schema, threshold=600):
        # Custom logic for determining chunking needs
        content_field, content_len = self._evaluate_content_field(schema["fields"])
        
        # Custom conditions
        if display_type == "conversation":
            return content_len > 200  # Lower threshold for conversations
        elif schema.get("collection_type") == "technical_docs":
            return content_len > 800  # Higher threshold for technical docs
        
        return super()._evaluate_needs_chunking(display_type, query_type, schema, threshold)
```

### Best Practices for Chunking

1. **Choose appropriate chunk sizes**: 
   - For semantic search: 200-500 tokens
   - For question answering: 100-300 tokens
   - For summarization: 500-1000 tokens

2. **Use overlap**: Include 10-20% overlap between chunks to maintain context

3. **Preserve structure**: Respect document boundaries (paragraphs, sections) when possible

4. **Test performance**: Monitor retrieval quality with different chunking strategies

## Retrieval System Customization

### Overview

Elysia's retrieval system supports hybrid search combining semantic, keyword, and filtered search capabilities. The system is designed to work primarily with Weaviate but can be extended to support other vector databases.

### Core Retrieval Components

#### 1. Query Tool (`elysia/tools/retrieval/query.py`)

The main retrieval interface that handles:
- Hybrid search (semantic + keyword)
- Multiple collection querying
- Dynamic query generation via LLMs
- Result formatting and display

#### 2. Query Generation

The system uses DSPy for dynamic query generation:

```python
from elysia.tools.retrieval.prompt_templates import QueryCreatorPrompt
from elysia.util.elysia_chain_of_thought import ElysiaChainOfThought

query_generator = ElysiaChainOfThought(
    QueryCreatorPrompt,
    tree_data=tree_data,
    environment=True,
    collection_schemas=True,
    tasks_completed=True,
    message_update=True,
    collection_names=collection_names,
)
```

### Supporting Different Vector Databases

While Elysia is primarily built for Weaviate, you can extend it to support other vector databases:

#### 1. Create a Database Adapter

```python
# elysia/util/adapters/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class VectorDatabaseAdapter(ABC):
    """Base adapter for vector database integration."""
    
    @abstractmethod
    async def connect(self, **kwargs) -> Any:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    async def create_collection(self, name: str, schema: Dict) -> Any:
        """Create a new collection/index."""
        pass
    
    @abstractmethod
    async def insert_documents(self, collection: str, documents: List[Dict]) -> None:
        """Insert documents into collection."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        collection: str, 
        query_vector: List[float] = None,
        query_text: str = None,
        filters: Dict = None,
        limit: int = 10
    ) -> List[Dict]:
        """Perform search operation."""
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        collection: str,
        query_text: str,
        alpha: float = 0.5,
        limit: int = 10
    ) -> List[Dict]:
        """Perform hybrid search (semantic + keyword)."""
        pass
```

#### 2. Implement Specific Adapters

**Pinecone Adapter Example:**

```python
# elysia/util/adapters/pinecone_adapter.py
import pinecone
from pinecone import Pinecone
from .base import VectorDatabaseAdapter

class PineconeAdapter(VectorDatabaseAdapter):
    def __init__(self, api_key: str, environment: str):
        self.api_key = api_key
        self.environment = environment
        self.client = None
    
    async def connect(self, **kwargs):
        self.client = Pinecone(api_key=self.api_key)
        return self.client
    
    async def create_collection(self, name: str, schema: Dict):
        return self.client.create_index(
            name=name,
            dimension=schema.get('dimension', 1536),
            metric=schema.get('metric', 'cosine'),
            spec=pinecone.ServerlessSpec(
                cloud='aws',
                region=self.environment
            )
        )
    
    async def insert_documents(self, collection: str, documents: List[Dict]):
        index = self.client.Index(collection)
        vectors = [
            (doc['id'], doc['vector'], doc.get('metadata', {}))
            for doc in documents
        ]
        index.upsert(vectors=vectors)
    
    async def search(self, collection: str, query_vector: List[float] = None, 
                    query_text: str = None, filters: Dict = None, limit: int = 10):
        index = self.client.Index(collection)
        
        if query_vector:
            results = index.query(
                vector=query_vector,
                filter=filters,
                top_k=limit,
                include_metadata=True
            )
            return [
                {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                }
                for match in results.matches
            ]
        else:
            raise NotImplementedError("Text-only search requires embedding generation")
    
    async def hybrid_search(self, collection: str, query_text: str, 
                           alpha: float = 0.5, limit: int = 10):
        # Implement hybrid search logic
        # This would require combining dense and sparse vectors
        raise NotImplementedError("Hybrid search not yet implemented for Pinecone")
```

**Qdrant Adapter Example:**

```python
# elysia/util/adapters/qdrant_adapter.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from .base import VectorDatabaseAdapter

class QdrantAdapter(VectorDatabaseAdapter):
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self.client = None
    
    async def connect(self, **kwargs):
        self.client = QdrantClient(host=self.host, port=self.port)
        return self.client
    
    async def create_collection(self, name: str, schema: Dict):
        return self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=schema.get('dimension', 1536),
                distance=Distance.COSINE
            )
        )
    
    async def insert_documents(self, collection: str, documents: List[Dict]):
        points = [
            PointStruct(
                id=doc['id'],
                vector=doc['vector'],
                payload=doc.get('metadata', {})
            )
            for doc in documents
        ]
        self.client.upsert(collection_name=collection, points=points)
    
    async def search(self, collection: str, query_vector: List[float] = None,
                    query_text: str = None, filters: Dict = None, limit: int = 10):
        results = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=filters,
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                'id': result.id,
                'score': result.score,
                'metadata': result.payload
            }
            for result in results
        ]
    
    async def hybrid_search(self, collection: str, query_text: str,
                           alpha: float = 0.5, limit: int = 10):
        # Implement using Qdrant's hybrid search capabilities
        # This would use both dense and sparse vectors
        raise NotImplementedError("Hybrid search implementation needed")
```

#### 3. Integrate Adapters with Elysia

Create a database manager that handles different adapters:

```python
# elysia/util/database_manager.py
from typing import Optional, Union
from .adapters.base import VectorDatabaseAdapter
from .adapters.pinecone_adapter import PineconeAdapter
from .adapters.qdrant_adapter import QdrantAdapter
from .client import ClientManager  # Existing Weaviate client

class DatabaseManager:
    def __init__(self, db_type: str = "weaviate", **kwargs):
        self.db_type = db_type
        self.adapter: Optional[VectorDatabaseAdapter] = None
        self.weaviate_client: Optional[ClientManager] = None
        
        if db_type == "weaviate":
            self.weaviate_client = ClientManager(**kwargs)
        elif db_type == "pinecone":
            self.adapter = PineconeAdapter(**kwargs)
        elif db_type == "qdrant":
            self.adapter = QdrantAdapter(**kwargs)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    async def connect(self):
        if self.adapter:
            return await self.adapter.connect()
        elif self.weaviate_client:
            return self.weaviate_client.connect_to_async_client()
    
    async def search(self, **kwargs):
        if self.adapter:
            return await self.adapter.search(**kwargs)
        elif self.weaviate_client:
            # Use existing Weaviate search logic
            return await self._weaviate_search(**kwargs)
    
    async def _weaviate_search(self, **kwargs):
        # Implement Weaviate search using existing logic
        pass
```

#### 4. Custom Query Execution

Extend the query execution to support different databases:

```python
# elysia/tools/retrieval/multi_db_query.py
from elysia.tools.retrieval.query import Query
from elysia.util.database_manager import DatabaseManager

class MultiDBQuery(Query):
    def __init__(self, db_type: str = "weaviate", **kwargs):
        super().__init__(**kwargs)
        self.db_manager = DatabaseManager(db_type=db_type, **kwargs)
    
    async def execute_search(self, query_output, **kwargs):
        """Execute search across different database types."""
        if self.db_manager.db_type == "weaviate":
            # Use existing Weaviate logic
            return await super().execute_weaviate_query(query_output, **kwargs)
        else:
            # Use adapter for other databases
            return await self._execute_adapter_search(query_output, **kwargs)
    
    async def _execute_adapter_search(self, query_output, **kwargs):
        await self.db_manager.connect()
        
        results = []
        for collection in query_output.target_collections:
            collection_results = await self.db_manager.search(
                collection=collection,
                query_text=query_output.search_query,
                limit=query_output.limit,
                # Convert filters to database-specific format
                filters=self._convert_filters(query_output.filters)
            )
            results.extend(collection_results)
        
        return results
    
    def _convert_filters(self, weaviate_filters):
        """Convert Weaviate filters to database-specific format."""
        # Implement conversion logic based on self.db_manager.db_type
        pass
```

### Advanced Retrieval Techniques

#### 1. Custom Similarity Functions

```python
def custom_similarity_search(
    query_embedding: List[float],
    document_embeddings: List[List[float]],
    metadata: List[Dict],
    top_k: int = 10
) -> List[Tuple[float, Dict]]:
    """Custom similarity computation with business logic."""
    
    similarities = []
    for i, doc_embedding in enumerate(document_embeddings):
        # Custom similarity computation
        base_similarity = cosine_similarity(query_embedding, doc_embedding)
        
        # Apply business logic modifications
        doc_meta = metadata[i]
        
        # Boost recent documents
        recency_boost = calculate_recency_boost(doc_meta.get('timestamp'))
        
        # Boost by document type
        type_boost = get_type_boost(doc_meta.get('document_type'))
        
        # User preference boost
        user_boost = get_user_preference_boost(doc_meta, user_preferences)
        
        final_score = base_similarity * recency_boost * type_boost * user_boost
        similarities.append((final_score, doc_meta))
    
    return sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]
```

#### 2. Multi-Vector Search

```python
class MultiVectorRetriever:
    """Retrieve using multiple vector representations."""
    
    def __init__(self, vector_configs: List[Dict]):
        self.vector_configs = vector_configs
    
    async def multi_vector_search(
        self,
        query: str,
        collection: str,
        weights: List[float] = None
    ) -> List[Dict]:
        """Search using multiple vector representations and combine results."""
        
        if weights is None:
            weights = [1.0 / len(self.vector_configs)] * len(self.vector_configs)
        
        all_results = []
        
        for i, config in enumerate(self.vector_configs):
            # Generate query vector for this configuration
            query_vector = await self._generate_vector(query, config)
            
            # Search with this vector
            results = await self._vector_search(
                collection=collection,
                query_vector=query_vector,
                vector_name=config['name']
            )
            
            # Weight the results
            for result in results:
                result['weighted_score'] = result['score'] * weights[i]
                result['vector_type'] = config['name']
            
            all_results.extend(results)
        
        # Combine and re-rank results
        return self._combine_results(all_results)
    
    def _combine_results(self, results: List[Dict]) -> List[Dict]:
        """Combine results from multiple vectors using RRF or similar."""
        # Group by document ID
        doc_groups = {}
        for result in results:
            doc_id = result['id']
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(result)
        
        # Apply Reciprocal Rank Fusion (RRF)
        combined_results = []
        for doc_id, doc_results in doc_groups.items():
            rrf_score = sum(1 / (60 + rank) for rank, _ in enumerate(doc_results))
            combined_results.append({
                'id': doc_id,
                'combined_score': rrf_score,
                'individual_scores': doc_results,
                'metadata': doc_results[0]['metadata']  # Assume same metadata
            })
        
        return sorted(combined_results, key=lambda x: x['combined_score'], reverse=True)
```

#### 3. Contextual Retrieval

```python
class ContextualRetriever:
    """Retriever that considers conversation context."""
    
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
    
    async def contextual_search(
        self,
        query: str,
        conversation_history: List[Dict],
        collection: str,
        context_weight: float = 0.3
    ) -> List[Dict]:
        """Search considering conversation context."""
        
        # Extract context from conversation history
        context = self._extract_context(conversation_history)
        
        # Generate context-aware query
        enhanced_query = await self._enhance_query_with_context(query, context)
        
        # Perform standard search
        results = await self.base_retriever.search(
            query=enhanced_query,
            collection=collection
        )
        
        # Re-rank results based on context relevance
        return self._rerank_with_context(results, context, context_weight)
    
    def _extract_context(self, conversation_history: List[Dict]) -> Dict:
        """Extract relevant context from conversation history."""
        context = {
            'entities': set(),
            'topics': set(),
            'previous_queries': [],
            'user_preferences': {}
        }
        
        for turn in conversation_history[-5:]:  # Last 5 turns
            # Extract entities
            entities = extract_named_entities(turn.get('content', ''))
            context['entities'].update(entities)
            
            # Extract topics
            topics = extract_topics(turn.get('content', ''))
            context['topics'].update(topics)
            
            # Store previous queries
            if turn.get('type') == 'query':
                context['previous_queries'].append(turn['content'])
        
        return context
    
    async def _enhance_query_with_context(self, query: str, context: Dict) -> str:
        """Enhance query with context information."""
        # Use LLM to enhance query with context
        enhancement_prompt = f"""
        Original query: {query}
        
        Conversation context:
        - Recent entities mentioned: {', '.join(list(context['entities'])[:10])}
        - Recent topics: {', '.join(list(context['topics'])[:5])}
        - Previous queries: {', '.join(context['previous_queries'][-3:])}
        
        Enhanced query (maintain original intent but add relevant context):
        """
        
        # Use your LLM to enhance the query
        # enhanced_query = await self.llm.generate(enhancement_prompt)
        # For now, simple concatenation
        enhanced_query = query
        if context['entities']:
            enhanced_query += f" (related to: {', '.join(list(context['entities'])[:3])})"
        
        return enhanced_query
```

## Continuous Chat Memory

### Overview

Elysia's memory system manages conversation context and maintains state across interactions through the `Environment` class. This system enables continuity in conversations and allows agents to build upon previous interactions.

### Core Memory Components

#### 1. Environment Class (`elysia/tree/objects.py`)

The Environment class provides structured storage for all interaction data:

```python
from elysia.tree.objects import Environment

# Initialize environment
env = Environment()

# Add retrieval results  
env.add_objects("query", "collection_name", objects=[...], metadata={...})

# Find previous results
previous_results = env.find("query", "collection_name")
```

#### 2. Memory Structure

The environment organizes data hierarchically:

```python
{
    "tool_name": {           # e.g., "query", "aggregate", "summarize"
        "result_name": [     # e.g., collection name, task identifier
            {
                "metadata": {...},    # Query parameters, timestamps, etc.
                "objects": [...]      # Retrieved/processed objects
            }
        ]
    },
    "hidden_environment": {...}  # Internal state not shown to LLM
}
```

### Customizing Memory Behavior

#### 1. Persistent Memory Storage

Extend the Environment class to support persistent storage:

```python
# elysia/tree/persistent_environment.py
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from .objects import Environment

class PersistentEnvironment(Environment):
    """Environment with persistent storage capabilities."""
    
    def __init__(self, user_id: str, conversation_id: str, storage_backend: str = "file", **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.storage_backend = storage_backend
        self._storage = self._init_storage()
    
    def _init_storage(self):
        if self.storage_backend == "file":
            return FileStorage(self.user_id, self.conversation_id)
        elif self.storage_backend == "redis":
            return RedisStorage(self.user_id, self.conversation_id)
        elif self.storage_backend == "weaviate":
            return WeaviateStorage(self.user_id, self.conversation_id)
        else:
            raise ValueError(f"Unsupported storage backend: {self.storage_backend}")
    
    async def save_state(self):
        """Save current environment state to persistent storage."""
        state = {
            'environment': self.environment,
            'hidden_environment': self.hidden_environment,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        }
        await self._storage.save(state)
    
    async def load_state(self):
        """Load environment state from persistent storage."""
        state = await self._storage.load()
        if state:
            self.environment = state.get('environment', {})
            self.hidden_environment = state.get('hidden_environment', {})
            return True
        return False
    
    async def add_objects(self, tool_name: str, result_name: str, objects: list, metadata: dict):
        """Override to auto-save after adding objects."""
        super().add_objects(tool_name, result_name, objects, metadata)
        await self.save_state()

class FileStorage:
    """File-based storage for conversation state."""
    
    def __init__(self, user_id: str, conversation_id: str, base_path: str = "./conversations"):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.file_path = f"{base_path}/{user_id}/{conversation_id}.json"
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
    
    async def save(self, state: Dict[str, Any]):
        with open(self.file_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    async def load(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None

class WeaviateStorage:
    """Weaviate-based storage for conversation state."""
    
    def __init__(self, user_id: str, conversation_id: str):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.collection_name = "ELYSIA_CONVERSATIONS"
    
    async def save(self, state: Dict[str, Any]):
        # Implement Weaviate storage
        from elysia.util.client import ClientManager
        
        client_manager = ClientManager()
        async with client_manager.connect_to_async_client() as client:
            if not await client.collections.exists(self.collection_name):
                await self._create_conversation_collection(client)
            
            collection = client.collections.get(self.collection_name)
            
            # Check if conversation exists
            existing = await collection.query.fetch_objects(
                filters=Filter.by_property("conversation_id").equal(self.conversation_id),
                limit=1
            )
            
            if existing.objects:
                # Update existing conversation
                await collection.data.update(
                    uuid=existing.objects[0].uuid,
                    properties={"state": json.dumps(state)}
                )
            else:
                # Create new conversation
                await collection.data.insert({
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id,
                    "state": json.dumps(state),
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                })
    
    async def load(self) -> Optional[Dict[str, Any]]:
        from elysia.util.client import ClientManager
        
        client_manager = ClientManager()
        async with client_manager.connect_to_async_client() as client:
            if not await client.collections.exists(self.collection_name):
                return None
            
            collection = client.collections.get(self.collection_name)
            results = await collection.query.fetch_objects(
                filters=Filter.by_property("conversation_id").equal(self.conversation_id),
                limit=1
            )
            
            if results.objects:
                state_str = results.objects[0].properties.get("state")
                if state_str:
                    return json.loads(state_str)
            
            return None
    
    async def _create_conversation_collection(self, client):
        """Create the conversation storage collection."""
        await client.collections.create(
            self.collection_name,
            properties=[
                Property(name="user_id", data_type=DataType.TEXT),
                Property(name="conversation_id", data_type=DataType.TEXT),
                Property(name="state", data_type=DataType.TEXT),
                Property(name="created_at", data_type=DataType.DATE),
                Property(name="updated_at", data_type=DataType.DATE),
            ],
            vectorizer_config=Configure.Vectorizer.none()
        )
```

#### 2. Smart Memory Management

Implement intelligent memory management with relevance-based retention:

```python
class SmartMemoryEnvironment(PersistentEnvironment):
    """Environment with intelligent memory management."""
    
    def __init__(self, max_memory_items: int = 100, relevance_threshold: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.max_memory_items = max_memory_items
        self.relevance_threshold = relevance_threshold
        self.memory_scorer = MemoryScorer()
    
    async def add_objects(self, tool_name: str, result_name: str, objects: list, metadata: dict):
        """Add results with automatic memory management."""
        # Add new objects  
        super().add_objects(tool_name, result_name, objects, metadata)
        
        # Check if memory cleanup is needed
        total_items = self._count_total_items()
        if total_items > self.max_memory_items:
            await self._cleanup_memory()
        
        await self.save_state()
    
    def _count_total_items(self) -> int:
        """Count total items in environment."""
        count = 0
        for tool_results in self.environment.values():
            for result_list in tool_results.values():
                count += len(result_list)
        return count
    
    async def _cleanup_memory(self):
        """Remove least relevant memories to stay under limit."""
        # Score all memory items
        scored_items = []
        
        for tool_name, tool_results in self.environment.items():
            for result_name, result_list in tool_results.items():
                for i, result in enumerate(result_list):
                    score = await self.memory_scorer.score_memory_item(
                        result, 
                        tool_name, 
                        result_name,
                        self.environment
                    )
                    scored_items.append({
                        'score': score,
                        'tool_name': tool_name,
                        'result_name': result_name,
                        'index': i,
                        'result': result
                    })
        
        # Sort by score (keep highest scoring items)
        scored_items.sort(key=lambda x: x['score'], reverse=True)
        
        # Keep only top items
        items_to_keep = scored_items[:self.max_memory_items]
        
        # Rebuild environment with only kept items
        new_environment = {}
        for item in items_to_keep:
            tool_name = item['tool_name']
            result_name = item['result_name']
            
            if tool_name not in new_environment:
                new_environment[tool_name] = {}
            if result_name not in new_environment[tool_name]:
                new_environment[tool_name][result_name] = []
            
            new_environment[tool_name][result_name].append(item['result'])
        
        self.environment = new_environment

class MemoryScorer:
    """Scores memory items for relevance and importance."""
    
    async def score_memory_item(
        self, 
        memory_item: Dict, 
        tool_name: str, 
        result_name: str,
        full_environment: Dict
    ) -> float:
        """Score a memory item for its importance/relevance."""
        
        score = 0.0
        metadata = memory_item.get('metadata', {})
        
        # Recency score (more recent = higher score)
        recency_score = self._calculate_recency_score(metadata.get('timestamp'))
        score += recency_score * 0.3
        
        # Usage frequency score
        usage_score = self._calculate_usage_score(memory_item, full_environment)
        score += usage_score * 0.2
        
        # Content relevance score
        relevance_score = await self._calculate_relevance_score(memory_item, full_environment)
        score += relevance_score * 0.3
        
        # Tool importance score
        tool_importance = self._get_tool_importance(tool_name)
        score += tool_importance * 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_recency_score(self, timestamp: str) -> float:
        """Calculate score based on how recent the memory is."""
        if not timestamp:
            return 0.0
        
        try:
            memory_time = datetime.fromisoformat(timestamp)
            now = datetime.utcnow()
            time_diff = (now - memory_time).total_seconds()
            
            # Exponential decay over 7 days
            decay_factor = 7 * 24 * 3600  # 7 days in seconds
            return math.exp(-time_diff / decay_factor)
        except:
            return 0.0
    
    def _calculate_usage_score(self, memory_item: Dict, full_environment: Dict) -> float:
        """Calculate score based on how often this memory is referenced."""
        # Simple implementation: count how many times similar objects appear
        objects = memory_item.get('objects', [])
        if not objects:
            return 0.0
        
        # Count similar objects in environment
        similar_count = 0
        total_count = 0
        
        for tool_results in full_environment.values():
            for result_list in tool_results.values():
                for result in result_list:
                    total_count += 1
                    if self._objects_similar(objects, result.get('objects', [])):
                        similar_count += 1
        
        return similar_count / max(total_count, 1) if total_count > 0 else 0.0
    
    async def _calculate_relevance_score(self, memory_item: Dict, full_environment: Dict) -> float:
        """Calculate relevance score based on semantic similarity to recent queries."""
        # Extract recent query context
        recent_queries = self._extract_recent_queries(full_environment)
        if not recent_queries:
            return 0.5  # Neutral score if no recent queries
        
        # Calculate semantic similarity (simplified)
        memory_text = self._extract_text_from_memory(memory_item)
        query_text = " ".join(recent_queries[-3:])  # Last 3 queries
        
        # Use embedding similarity (placeholder implementation)
        similarity = await self._calculate_semantic_similarity(memory_text, query_text)
        return similarity
    
    def _get_tool_importance(self, tool_name: str) -> float:
        """Get importance score for different tools."""
        importance_map = {
            'query': 0.9,      # Query results are very important
            'aggregate': 0.7,   # Aggregation results are important
            'summarize': 0.8,   # Summaries are important
            'text': 0.6,       # Text responses are moderately important
            'visualize': 0.5    # Visualizations are less critical
        }
        return importance_map.get(tool_name, 0.5)
```

#### 3. Context-Aware Memory Retrieval

```python
class ContextAwareEnvironment(SmartMemoryEnvironment):
    """Environment that provides context-aware memory retrieval."""
    
    async def get_relevant_context(
        self, 
        current_query: str, 
        max_items: int = 10,
        context_types: List[str] = None
    ) -> Dict[str, Any]:
        """Retrieve relevant context for the current query."""
        
        if context_types is None:
            context_types = ['query', 'aggregate', 'summarize']
        
        relevant_items = []
        
        # Score all memory items for relevance to current query
        for tool_name in context_types:
            if tool_name in self.environment:
                for result_name, result_list in self.environment[tool_name].items():
                    for result in result_list:
                        relevance_score = await self._calculate_query_relevance(
                            current_query, result
                        )
                        if relevance_score > 0.3:  # Relevance threshold
                            relevant_items.append({
                                'score': relevance_score,
                                'tool_name': tool_name,
                                'result_name': result_name,
                                'result': result
                            })
        
        # Sort by relevance and return top items
        relevant_items.sort(key=lambda x: x['score'], reverse=True)
        top_items = relevant_items[:max_items]
        
        # Format context for consumption
        context = {
            'relevant_retrievals': [],
            'relevant_summaries': [],
            'related_queries': [],
            'entities_mentioned': set(),
            'topics_covered': set()
        }
        
        for item in top_items:
            tool_name = item['tool_name']
            result = item['result']
            
            if tool_name == 'query':
                context['relevant_retrievals'].append({
                    'collection': item['result_name'],
                    'objects': result.get('objects', []),
                    'metadata': result.get('metadata', {}),
                    'relevance_score': item['score']
                })
            elif tool_name == 'summarize':
                context['relevant_summaries'].append({
                    'summary': result.get('objects', [{}])[0].get('summary', ''),
                    'source': item['result_name'],
                    'relevance_score': item['score']
                })
            
            # Extract entities and topics
            text_content = self._extract_text_from_result(result)
            entities = self._extract_entities(text_content)
            topics = self._extract_topics(text_content)
            
            context['entities_mentioned'].update(entities)
            context['topics_covered'].update(topics)
        
        # Convert sets to lists for JSON serialization
        context['entities_mentioned'] = list(context['entities_mentioned'])
        context['topics_covered'] = list(context['topics_covered'])
        
        return context
    
    async def _calculate_query_relevance(self, query: str, memory_result: Dict) -> float:
        """Calculate how relevant a memory result is to the current query."""
        
        # Extract text from memory result
        memory_text = self._extract_text_from_result(memory_result)
        
        # Calculate semantic similarity
        semantic_score = await self._calculate_semantic_similarity(query, memory_text)
        
        # Calculate entity overlap
        query_entities = self._extract_entities(query)
        memory_entities = self._extract_entities(memory_text)
        entity_overlap = len(query_entities & memory_entities) / max(len(query_entities), 1)
        
        # Calculate keyword overlap
        query_keywords = set(query.lower().split())
        memory_keywords = set(memory_text.lower().split())
        keyword_overlap = len(query_keywords & memory_keywords) / max(len(query_keywords), 1)
        
        # Combine scores
        relevance_score = (
            semantic_score * 0.6 +
            entity_overlap * 0.25 +
            keyword_overlap * 0.15
        )
        
        return relevance_score
```

#### 4. Memory Integration with Tree

Integrate the enhanced memory system with the Tree class:

```python
# elysia/tree/memory_aware_tree.py
from elysia.tree.tree import Tree
from elysia.tree.persistent_environment import ContextAwareEnvironment

class MemoryAwareTree(Tree):
    """Tree with enhanced memory capabilities."""
    
    def __init__(self, memory_type: str = "context_aware", **kwargs):
        super().__init__(**kwargs)
        
        # Replace default environment with memory-aware version
        if memory_type == "context_aware":
            self.tree_data.environment = ContextAwareEnvironment(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                **kwargs
            )
        elif memory_type == "smart":
            self.tree_data.environment = SmartMemoryEnvironment(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                **kwargs
            )
        elif memory_type == "persistent":
            self.tree_data.environment = PersistentEnvironment(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                **kwargs
            )
    
    async def __call__(self, user_prompt: str, **kwargs) -> AsyncGenerator:
        """Enhanced call with memory context injection."""
        
        # Load previous conversation state
        if hasattr(self.tree_data.environment, 'load_state'):
            await self.tree_data.environment.load_state()
        
        # Get relevant context for the current query
        if hasattr(self.tree_data.environment, 'get_relevant_context'):
            relevant_context = await self.tree_data.environment.get_relevant_context(
                user_prompt, max_items=5
            )
            
            # Inject context into tree data for LLM consumption
            self.tree_data.environment.hidden_environment['query_context'] = relevant_context
        
        # Proceed with normal tree execution
        async for result in super().__call__(user_prompt, **kwargs):
            yield result
        
        # Save updated state
        if hasattr(self.tree_data.environment, 'save_state'):
            await self.tree_data.environment.save_state()
```

### Memory Best Practices

1. **Selective Memory**: Don't store everything - be selective about what gets remembered
2. **Context Relevance**: Prioritize recent and relevant memories over older ones
3. **Memory Limits**: Implement reasonable limits to prevent memory bloat
4. **Privacy Considerations**: Be mindful of sensitive information in persistent storage
5. **Performance**: Balance memory richness with retrieval performance

## Development Setup

### Prerequisites

- Python 3.10-3.12
- Poetry or pip for dependency management
- Access to a Weaviate instance (local or cloud)
- API keys for LLM providers (OpenAI, OpenRouter, etc.)

### Installation for Development

1. **Clone the repository:**
```bash
git clone https://github.com/weaviate/elysia
cd elysia
```

2. **Set up virtual environment:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -e .
pip install -e ".[dev]"  # Include development dependencies
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

Example `.env` configuration:
```env
# Weaviate Configuration
WCD_URL=https://your-cluster.weaviate.network
WCD_API_KEY=your-weaviate-api-key
WEAVIATE_IS_LOCAL=False

# LLM Configuration
OPENAI_API_KEY=your-openai-key
OPENROUTER_API_KEY=your-openrouter-key

# Development Settings
LOGGING_LEVEL=DEBUG
USE_FEEDBACK=False
```

5. **Install spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

### Directory Structure

```
elysia/
├── elysia/                 # Main package
│   ├── api/               # FastAPI backend
│   ├── tools/             # RAG tools (query, aggregate, etc.)
│   │   ├── retrieval/     # Retrieval-specific tools
│   │   ├── postprocessing/ # Post-processing tools
│   │   ├── text/          # Text generation tools
│   │   └── visualisation/ # Visualization tools
│   ├── tree/              # Decision tree logic
│   ├── preprocessing/     # Collection preprocessing
│   ├── util/              # Utility functions
│   └── config.py          # Configuration management
├── tests/                 # Test suite
├── docs/                  # Documentation
└── examples/              # Example scripts
```

### Configuration Management

Elysia uses a centralized configuration system:

```python
from elysia.config import Settings, settings

# Access current settings
print(settings.WCD_URL)
print(settings.OPENAI_API_KEY)

# Create custom settings
custom_settings = Settings(
    WCD_URL="http://localhost:8080",
    WEAVIATE_IS_LOCAL=True,
    LOGGING_LEVEL="INFO"
)
```

## Testing Guidelines

### Test Structure

Tests are organized by functionality:

```
tests/
├── no_reqs/              # Tests that don't require external services
│   ├── test_chunking.py
│   ├── test_memory.py
│   └── test_utils.py
└── requires_env/         # Tests requiring API keys/services
    ├── test_retrieval.py
    ├── test_preprocessing.py
    └── test_integration.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/no_reqs/

# Run with coverage
pytest --cov=elysia --cov-report=html

# Run specific test
pytest tests/no_reqs/test_chunking.py::test_sentence_chunking
```

### Writing Tests

#### 1. Unit Tests for Chunking

```python
# tests/no_reqs/test_chunking.py
import pytest
from elysia.tools.retrieval.chunk import Chunker

class TestChunker:
    def test_sentence_chunking(self):
        chunker = Chunker(chunking_strategy="sentences", num_sentences=2)
        document = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        chunks, spans = chunker.chunk(document)
        
        assert len(chunks) == 2
        assert "First sentence. Second sentence." in chunks[0]
        assert "Third sentence. Fourth sentence." in chunks[1]
    
    def test_token_chunking(self):
        chunker = Chunker(chunking_strategy="fixed", num_tokens=5)
        document = "This is a test document with many tokens for testing chunking."
        
        chunks, spans = chunker.chunk(document)
        
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 7 for chunk in chunks)  # Allow some overlap
    
    def test_chunk_overlap(self):
        chunker = Chunker(chunking_strategy="sentences", num_sentences=3)
        document = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        
        chunks, spans = chunker.chunk_by_sentences(document, overlap_sentences=1)
        
        # Check that chunks have expected overlap
        assert len(chunks) >= 2
        # Verify overlap exists between consecutive chunks
```

#### 2. Integration Tests for Retrieval

```python
# tests/requires_env/test_retrieval.py
import pytest
import asyncio
from elysia.tools.retrieval.query import Query
from elysia.tree.objects import TreeData, CollectionData
from elysia.util.client import ClientManager

@pytest.mark.asyncio
class TestRetrieval:
    @pytest.fixture
    async def setup_test_data(self):
        # Setup test collection with sample data
        client_manager = ClientManager()
        # ... setup code
        yield client_manager
        # ... cleanup code
    
    async def test_semantic_search(self, setup_test_data):
        client_manager = setup_test_data
        
        # Create test tree data
        tree_data = TreeData(
            user_prompt="Find information about AI",
            collection_data=CollectionData(collection_names=["test_collection"])
        )
        
        # Execute query
        query_tool = Query()
        inputs = {"collection_names": ["test_collection"]}
        
        results = []
        async for result in query_tool(
            tree_data=tree_data,
            inputs=inputs,
            base_lm=None,  # Mock LM
            complex_lm=None,  # Mock LM
            client_manager=client_manager
        ):
            results.append(result)
        
        # Verify results
        assert len(results) > 0
        # Add more specific assertions
```

#### 3. Memory System Tests

```python
# tests/no_reqs/test_memory.py
import pytest
from elysia.tree.objects import Environment

class TestEnvironment:
    def test_add_and_retrieve_results(self):
        env = Environment()
        
        # Add test results
        test_objects = [{"id": 1, "content": "test content"}]
        test_metadata = {"query": "test query", "timestamp": "2024-01-01"}
        
        env.add_objects("query", "test_collection", test_objects, test_metadata)
        
        # Retrieve results  
        results = env.find("query", "test_collection")
        
        assert len(results) == 1
        assert results[0]["objects"] == test_objects
        assert results[0]["metadata"] == test_metadata
    
    def test_memory_search(self):
        env = Environment()
        
        # Add multiple results
        env.add_objects("query", "collection1", [{"type": "document"}], {"topic": "AI"})
        env.add_objects("query", "collection2", [{"type": "article"}], {"topic": "ML"})
        env.add_objects("aggregate", "collection1", [{"count": 10}], {"operation": "count"})
        
        # Search by tool and result name
        ai_results = env.find("query", "collection1")
        assert len(ai_results) == 1
        
        # Get all query results from environment
        query_results = []
        if "query" in env.environment:
            for result_name in env.environment["query"]:
                query_results.extend(env.find("query", result_name))
        assert len(query_results) == 2
```

### Mocking External Services

For tests that interact with external services, use mocking:

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_weaviate_client():
    client = AsyncMock()
    client.collections.exists.return_value = True
    client.collections.get.return_value.query.fetch_objects.return_value.objects = []
    return client

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate.return_value = "Mocked LLM response"
    return llm
```

## Contributing Guidelines

### Code Style

Elysia follows Python best practices:

1. **PEP 8 compliance** with line length of 88 characters (Black default)
2. **Type hints** for all public functions and methods
3. **Docstrings** in Google style for all public APIs
4. **Async/await** for I/O operations

### Code Formatting

Use the provided development tools:

```bash
# Format code
black elysia/ tests/

# Sort imports
isort elysia/ tests/

# Type checking (if mypy is configured)
mypy elysia/
```

### Submitting Changes

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the guidelines above

3. **Add tests** for new functionality

4. **Run the test suite:**
```bash
pytest
```

5. **Update documentation** if needed

6. **Commit with descriptive messages:**
```bash
git commit -m "Add custom chunking strategy for PDF documents"
```

7. **Submit a pull request** with:
   - Clear description of changes
   - Test results
   - Documentation updates
   - Breaking change notes (if any)

### Common Contribution Areas

1. **New Chunking Strategies**: Implement domain-specific chunking
2. **Vector Database Adapters**: Add support for new vector databases
3. **Memory Enhancements**: Improve memory management and context handling
4. **Retrieval Algorithms**: Add new search and ranking algorithms
5. **Tool Extensions**: Create new tools for specific use cases
6. **Performance Optimizations**: Improve speed and memory usage
7. **Documentation**: Improve guides and examples

### Performance Considerations

When contributing, consider:

1. **Async Operations**: Use async/await for I/O bound operations
2. **Memory Usage**: Be mindful of memory consumption in long-running conversations
3. **Vectorization Costs**: Consider the cost of generating embeddings
4. **Caching**: Implement appropriate caching where beneficial
5. **Batch Operations**: Use batch operations for multiple documents

### Getting Help

- **Documentation**: https://weaviate.github.io/elysia/
- **GitHub Issues**: https://github.com/weaviate/elysia/issues
- **Discussions**: Use GitHub Discussions for questions and ideas

---

This guide provides a comprehensive foundation for contributing to Elysia's RAG system. The modular architecture makes it straightforward to customize and extend functionality while maintaining compatibility with the existing system.