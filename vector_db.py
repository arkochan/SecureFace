import faiss
import numpy as np
import time
import os # Import os for path checking
# Import the DatabaseConnection class
from database.connection import DatabaseConnection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for the FAISS index and ID counter
index = None
dimension = 512  # ArcFace embedding dimension
next_embedding_id = 0  # Simple in-memory ID counter, consider persistence

def init_index(dim=512, index_path=None):
    """
    Initializes the FAISS index.
    If index_path is provided, it attempts to load the index from that path.
    Otherwise, it creates a new index.
    """
    global index, dimension, next_embedding_id
    dimension = dim
    if index_path and os.path.exists(index_path):
        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        # Determine the next ID to use. This is a simplification.
        # In practice, you might want to store the next ID persistently or
        # derive it from the index contents/max ID.
        # For IVF, you might need to check ntotal or stored vectors.
        next_embedding_id = index.ntotal
        logger.info(f"Loaded index with {index.ntotal} vectors. Next ID set to {next_embedding_id}")
    else:
        logger.info("Creating new FAISS index")
        # Using IndexFlatL2 for simplicity. Consider IndexIVFFlat for larger datasets.
        index = faiss.IndexFlatL2(dimension)
        next_embedding_id = 0

    # Ensure the metadata table exists in PostgreSQL
    _create_metadata_table()

def _create_metadata_table():
    """Creates the metadata table in PostgreSQL if it doesn't exist."""
    db_conn = DatabaseConnection()
    if not db_conn.connect():
        logger.error("Failed to connect to database for creating metadata table.")
        return

    try:
        create_table_query = """
        CREATE TABLE IF NOT EXISTS embeddings_metadata (
            embedding_id INTEGER PRIMARY KEY, -- FAISS internal ID
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            -- FOREIGN KEY (user_id) REFERENCES users(user_id) -- Uncomment if you have a users table
        );
        """
        if db_conn.execute_update(create_table_query):
            logger.info("Ensured embeddings_metadata table exists")
        else:
            logger.error("Failed to create embeddings_metadata table")
    finally:
        db_conn.disconnect()

def add_embedding(vector, user_id):
    """
    Adds a single embedding vector to the FAISS index and its metadata to PostgreSQL.

    Args:
        vector (np.ndarray): The 512-D embedding vector.
        user_id (int): The ID of the user this embedding belongs to.

    Returns:
        int: The ID assigned to the embedding by FAISS, or -1 on error.
    """
    global index, next_embedding_id

    if index is None:
        logger.error("FAISS index is not initialized. Call init_index() first.")
        return -1

    if vector.shape[0] != dimension:
        logger.error(f"Embedding dimension mismatch. Expected {dimension}, got {vector.shape[0]}")
        return -1

    try:
        # Add vector to FAISS index
        # FAISS assigns an internal ID based on the order of insertion (0, 1, 2, ...)
        faiss_id = int(index.ntotal) # ID that FAISS will assign - convert to Python int
        vector_float32 = vector.astype(np.float32).reshape(1, -1)
        index.add(vector_float32)
        logger.info(f"Added vector to FAISS index with ID {faiss_id}")

        # Store metadata in PostgreSQL
        db_conn = DatabaseConnection()
        if not db_conn.connect():
            logger.error("Failed to connect to database for adding embedding metadata.")
            return -1

        try:
            insert_metadata_query = """
            INSERT INTO embeddings_metadata (embedding_id, user_id, created_at)
            VALUES (%s, %s, %s);
            """
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            # Convert types to ensure compatibility with PostgreSQL
            faiss_id = int(faiss_id)
            user_id = int(user_id)
            if db_conn.execute_update(insert_metadata_query, (faiss_id, user_id, timestamp)):
                logger.info(f"Stored metadata for embedding ID {faiss_id}, user ID {user_id}")
                return faiss_id
            else:
                logger.error(f"Failed to store metadata for embedding ID {faiss_id}")
                return -1
        finally:
            db_conn.disconnect()

    except Exception as e:
        logger.error(f"Error adding vector to FAISS index: {e}")
        return -1

def search_embeddings(query_vector, k=5):
    """
    Searches for the k most similar embeddings in the FAISS index.

    Args:
        query_vector (np.ndarray): The 512-D query embedding vector.
        k (int): The number of nearest neighbors to search for.

    Returns:
        list: A list of tuples (faiss_id, distance, user_id, created_at) for the k nearest neighbors,
              or an empty list on error.
    """
    global index

    if index is None or index.ntotal == 0:
        logger.warning("FAISS index is not initialized or is empty.")
        return []

    if query_vector.shape[0] != dimension:
        logger.error(f"Query vector dimension mismatch. Expected {dimension}, got {query_vector.shape[0]}")
        return []

    try:
        query_vector_float32 = query_vector.astype(np.float32).reshape(1, -1)
        distances, indices = index.search(query_vector_float32, k)
        logger.info(f"Search completed, found {len(indices[0])} results")

        results = []
        db_conn = DatabaseConnection()
        if not db_conn.connect():
            logger.error("Failed to connect to database for fetching metadata during search.")
            return []

        try:
            # Fetch metadata for each returned FAISS ID
            for i in range(len(indices[0])):
                faiss_id = indices[0][i]
                distance = distances[0][i]
                
                # FAISS returns -1 for indices if k is larger than the number of vectors
                if faiss_id == -1:
                    continue 
                    
                # Convert numpy.int64 to regular Python int to avoid PostgreSQL adaptation issues
                faiss_id = int(faiss_id)
                    
                # Query metadata
                select_metadata_query = """
                SELECT user_id, created_at FROM embeddings_metadata WHERE embedding_id = %s;
                """
                rows = db_conn.execute_query(select_metadata_query, (faiss_id,))
                if rows and len(rows) > 0:
                    # Assuming RealDictCursor, rows are dict-like
                    user_id = rows[0]['user_id']
                    created_at = rows[0]['created_at']
                    results.append((faiss_id, distance, user_id, created_at))
                else:
                    logger.warning(f"No metadata found for FAISS ID {faiss_id}")
                    
            return results
        finally:
            db_conn.disconnect()
                
    except Exception as e:
        logger.error(f"Error searching FAISS index: {e}")
        return []

def save_index(path):
    """Saves the FAISS index to a file."""
    global index
    if index is None:
        logger.error("FAISS index is not initialized. Nothing to save.")
        return
    try:
        faiss.write_index(index, path)
        logger.info(f"FAISS index saved to {path}")
    except Exception as e:
        logger.error(f"Error saving FAISS index to {path}: {e}")

def get_index_stats():
    """Returns basic statistics about the FAISS index."""
    global index, next_embedding_id
    if index is None:
        return {"initialized": False}
    return {
        "initialized": True,
        "dimension": dimension,
        "total_vectors": index.ntotal,
        "next_embedding_id": next_embedding_id # This might not be perfectly accurate after loading
    }

def get_template_embedding(template_user_id=-1):
    """
    Retrieves the template embedding vector associated with a specific user_id (default -1).
    This function queries the database for the latest embedding with the given user_id
    and then reconstructs the vector from the FAISS index.

    Args:
        template_user_id (int): The user ID used to mark the template (default: -1).

    Returns:
        np.ndarray or None: The reconstructed template embedding vector, or None if not found or error.
    """
    global index

    if index is None:
        logger.error("FAISS index is not initialized.")
        return None
        
    if index.ntotal == 0:
         logger.warning("FAISS index is empty.")
         return None

    db_conn = DatabaseConnection()
    if not db_conn.connect():
        logger.error("Failed to connect to database for retrieving template embedding.")
        return None

    try:
        # Query for the embedding ID with the specified user_id, get the most recent one
        select_template_query = """
        SELECT embedding_id FROM embeddings_metadata WHERE user_id = %s ORDER BY created_at DESC LIMIT 1;
        """
        rows = db_conn.execute_query(select_template_query, (template_user_id,))
        if rows and len(rows) > 0:
            template_faiss_id = rows[0]['embedding_id']
            logger.info(f"Found template with FAISS ID: {template_faiss_id} for user_id: {template_user_id}")
            
            # Reconstruct the vector from the index using its internal ID
            # This works for IndexFlat. For IndexIVF, you might need to enable storage of original vectors
            # or use a different approach.
            if hasattr(index, 'reconstruct'):
                try:
                    reconstructed_vector = index.reconstruct(template_faiss_id)
                    logger.info("Template embedding reconstructed from FAISS index.")
                    return reconstructed_vector
                except Exception as reconstruct_error:
                    logger.error(f"Error reconstructing template embedding: {reconstruct_error}")
                    return None
            else:
                logger.error("FAISS index type does not support reconstruction.")
                return None
        else:
            logger.info(f"No template found for user_id: {template_user_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving template embedding: {e}")
        return None
    finally:
        db_conn.disconnect()
            
# Initialize the index when the module is imported, or let the main app do it explicitly
# init_index()