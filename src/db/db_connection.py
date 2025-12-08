"""
Database connection module for PostgreSQL with pgvector support.

Requires environment variables:
- DB_USER: Database username
- DB_PASSWORD: Database password  
- DB_HOST: Database host (Cloud SQL IP)
- DB_PORT: Database port (default: 5432)
- DB_NAME: Database name (default: postgres)
"""

import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector


def get_db_config() -> dict:
    """Get database configuration from environment variables."""
    return {
        "user": os.environ.get("DB_USER"),
        "password": os.environ.get("DB_PASSWORD"),
        "host": os.environ.get("DB_HOST"),
        "port": os.environ.get("DB_PORT", "5432"),
        "dbname": os.environ.get("DB_NAME", "postgres"),
    }


def get_connection():
    """Create a new database connection."""
    config = get_db_config()
    
    # Validate required config
    required = ["user", "password", "host"]
    missing = [k for k in required if not config.get(k)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(f'DB_{k.upper()}' for k in missing)}")
    
    conn = psycopg2.connect(**config)
    register_vector(conn)  # Register pgvector type
    return conn


@contextmanager
def get_db_cursor(commit: bool = True):
    """Context manager for database operations.
    
    Args:
        commit: Whether to commit the transaction on success
        
    Yields:
        cursor: Database cursor
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        yield cursor
        if commit:
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def init_embeddings_table(
    table_name: str = "document_embeddings",
    embedding_dim: int = 768
) -> None:
    """Initialize the embeddings table with pgvector.
    
    Args:
        table_name: Name of the table to create
        embedding_dim: Dimension of the embedding vectors (768 for text-multilingual-embedding-002)
    """
    with get_db_cursor() as cursor:
        # Create table with vector column
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                chunk_id INTEGER,
                text TEXT NOT NULL,
                context TEXT,
                metadata JSONB,
                strategy VARCHAR(50),
                embedding vector({embedding_dim}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for vector similarity search (IVFFlat or HNSW)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx 
            ON {table_name} 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        
        print(f"✅ Table '{table_name}' initialized with {embedding_dim}-dimensional vectors")


def test_connection() -> bool:
    """Test the database connection."""
    try:
        with get_db_cursor(commit=False) as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print(f"✅ Database connection successful: {result}")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def insert_chunks_batch(
    chunks: list[dict],
    embeddings: list,
    table_name: str = "document_embeddings",
    batch_size: int = 100
) -> int:
    """Insert chunks with embeddings to database in batches.
    
    Args:
        chunks: List of chunk dictionaries (from JSONL files)
               Expected keys: chunk_id, text, context, metadata, strategy
        embeddings: List of embedding vectors (same order as chunks)
        table_name: Table name to insert into
        batch_size: Number of rows per batch insert
    
    Returns:
        Number of rows inserted
    """
    import json
    import numpy as np
    
    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
    
    total = len(chunks)
    inserted = 0
    
    with get_db_cursor() as cursor:
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            # Prepare data tuples
            data = []
            for chunk, emb in zip(batch_chunks, batch_embeddings):
                # Convert embedding to list if numpy array
                if isinstance(emb, np.ndarray):
                    emb = emb.tolist()
                
                # Convert metadata dict to JSON string
                metadata = chunk.get("metadata", {})
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)
                
                data.append((
                    chunk.get("chunk_id"),
                    chunk.get("text", ""),
                    chunk.get("context"),
                    metadata,
                    chunk.get("strategy"),
                    emb
                ))
            
            # Batch insert using execute_values (much faster than individual inserts)
            execute_values(
                cursor,
                f"""INSERT INTO {table_name} 
                    (chunk_id, text, context, metadata, strategy, embedding)
                    VALUES %s""",
                data,
                template="(%s, %s, %s, %s, %s, %s::vector)"
            )
            
            inserted += len(data)
            print(f"   ✓ Inserted {inserted}/{total} chunks")
    
    print(f"✅ Successfully inserted {inserted} chunks into '{table_name}'")
    return inserted


def clear_table(table_name: str = "document_embeddings") -> None:
    """Clear all rows from a table.
    
    Args:
        table_name: Table name to clear
    """
    with get_db_cursor() as cursor:
        cursor.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY")
        print(f"✅ Table '{table_name}' cleared")


def get_table_count(table_name: str = "document_embeddings") -> int:
    """Get the number of rows in a table.
    
    Args:
        table_name: Table name to count
    
    Returns:
        Number of rows
    """
    with get_db_cursor(commit=False) as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        return count


def search_similar(
    query_embedding: list,
    table_name: str = "document_embeddings",
    top_k: int = 5
) -> list[dict]:
    """Search for similar chunks using cosine similarity.
    
    Args:
        query_embedding: Query embedding vector
        table_name: Table to search in
        top_k: Number of results to return
    
    Returns:
        List of dicts with chunk data and similarity score
    """
    import numpy as np
    
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    
    with get_db_cursor(commit=False) as cursor:
        cursor.execute(
            f"""
            SELECT 
                id,
                chunk_id,
                text,
                context,
                metadata,
                strategy,
                1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k)
        )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "chunk_id": row[1],
                "text": row[2],
                "context": row[3],
                "metadata": row[4],
                "strategy": row[5],
                "similarity": float(row[6])
            })
        
        return results
