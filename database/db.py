from .connection import DatabaseConnection

class SecureFaceDB:
    """Main database class with methods for SecureFace operations"""
    
    def __init__(self):
        self.db = DatabaseConnection()
        self.db.connect()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.disconnect()
    
    # Users table methods
    def create_user(self, full_name, role_id=None, department=None, image_path=None):
        """Create a new user"""
        query = """
        INSERT INTO users (full_name, role_id, department, image_path)
        VALUES (%s, %s, %s, %s)
        RETURNING user_id
        """
        result = self.db.execute_insert_returning(query, (full_name, role_id, department, image_path))
        return result[0]['user_id'] if result else None
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        query = """
        SELECT user_id, full_name, role_id, department, image_path, created_at, updated_at
        FROM users
        WHERE user_id = %s
        """
        result = self.db.execute_query(query, (user_id,))
        return result[0] if result else None
    
    def get_all_users(self):
        """Get all users"""
        query = """
        SELECT user_id, full_name, role_id, department, image_path, created_at, updated_at
        FROM users
        """
        return self.db.execute_query(query)
    
    def update_user(self, user_id, full_name=None, role_id=None, department=None, image_path=None):
        """Update user information"""
        # Build dynamic query based on provided fields
        updates = []
        params = []
        
        if full_name is not None:
            updates.append("full_name = %s")
            params.append(full_name)
        if role_id is not None:
            updates.append("role_id = %s")
            params.append(role_id)
        if department is not None:
            updates.append("department = %s")
            params.append(department)
        if image_path is not None:
            updates.append("image_path = %s")
            params.append(image_path)
        
        # Always update the updated_at timestamp
        updates.append("updated_at = CURRENT_TIMESTAMP")
        
        if not updates:
            return False
        
        query = f"UPDATE users SET {', '.join(updates)} WHERE user_id = %s"
        params.append(user_id)
        
        return self.db.execute_update(query, params)
    
    def delete_user(self, user_id):
        """Delete a user"""
        query = "DELETE FROM users WHERE user_id = %s"
        return self.db.execute_update(query, (user_id,))
    
    # Roles table methods
    def create_role(self, role_name, access_level, description=None):
        """Create a new role"""
        query = """
        INSERT INTO roles (role_name, access_level, description)
        VALUES (%s, %s, %s)
        RETURNING role_id
        """
        result = self.db.execute_insert_returning(query, (role_name, access_level, description))
        return result[0]['role_id'] if result else None
    
    def get_role_by_id(self, role_id):
        """Get role by ID"""
        query = """
        SELECT role_id, role_name, access_level, description
        FROM roles
        WHERE role_id = %s
        """
        result = self.db.execute_query(query, (role_id,))
        return result[0] if result else None
    
    def get_all_roles(self):
        """Get all roles"""
        query = """
        SELECT role_id, role_name, access_level, description
        FROM roles
        """
        return self.db.execute_query(query)
    
    def update_role(self, role_id, role_name=None, access_level=None, description=None):
        """Update role information"""
        updates = []
        params = []
        
        if role_name is not None:
            updates.append("role_name = %s")
            params.append(role_name)
        if access_level is not None:
            updates.append("access_level = %s")
            params.append(access_level)
        if description is not None:
            updates.append("description = %s")
            params.append(description)
        
        if not updates:
            return False
        
        query = f"UPDATE roles SET {', '.join(updates)} WHERE role_id = %s"
        params.append(role_id)
        
        return self.db.execute_update(query, params)
    
    def delete_role(self, role_id):
        """Delete a role"""
        query = "DELETE FROM roles WHERE role_id = %s"
        return self.db.execute_update(query, (role_id,))
    
    # Recognition logs methods
    def create_recognition_log(self, user_id=None, camera_id=None, recognition_result=None, confidence_score=None):
        """Create a new recognition log entry"""
        query = """
        INSERT INTO recognition_logs (user_id, camera_id, recognition_result, confidence_score)
        VALUES (%s, %s, %s, %s)
        RETURNING log_id
        """
        result = self.db.execute_insert_returning(query, (user_id, camera_id, recognition_result, confidence_score))
        return result[0]['log_id'] if result else None
    
    def get_log_by_id(self, log_id):
        """Get recognition log by ID"""
        query = """
        SELECT log_id, user_id, camera_id, recognition_result, timestamp, confidence_score
        FROM recognition_logs
        WHERE log_id = %s
        """
        result = self.db.execute_query(query, (log_id,))
        return result[0] if result else None
    
    def get_logs_by_user(self, user_id):
        """Get all recognition logs for a specific user"""
        query = """
        SELECT log_id, user_id, camera_id, recognition_result, timestamp, confidence_score
        FROM recognition_logs
        WHERE user_id = %s
        ORDER BY timestamp DESC
        """
        return self.db.execute_query(query, (user_id,))
    
    def get_all_logs(self):
        """Get all recognition logs"""
        query = """
        SELECT log_id, user_id, camera_id, recognition_result, timestamp, confidence_score
        FROM recognition_logs
        ORDER BY timestamp DESC
        """
        return self.db.execute_query(query)
    
    # Additional utility methods
    def get_user_with_role(self, user_id):
        """Get user information with their role details"""
        query = """
        SELECT u.user_id, u.full_name, u.department, u.image_path, u.created_at, u.updated_at,
               r.role_name, r.access_level, r.description as role_description
        FROM users u
        LEFT JOIN roles r ON u.role_id = r.role_id
        WHERE u.user_id = %s
        """
        result = self.db.execute_query(query, (user_id,))
        return result[0] if result else None
        
    # Embeddings metadata methods
    def get_all_embeddings_metadata(self):
        """Get all embeddings metadata"""
        query = """
        SELECT embedding_id, user_id, created_at
        FROM embeddings_metadata
        ORDER BY embedding_id
        """
        return self.db.execute_query(query)
        
    def get_embeddings_by_user(self, user_id):
        """Get all embeddings for a specific user"""
        query = """
        SELECT embedding_id, user_id, created_at
        FROM embeddings_metadata
        WHERE user_id = %s
        ORDER BY created_at DESC
        """
        return self.db.execute_query(query, (user_id,))
        
    def delete_embedding_metadata(self, embedding_id):
        """Delete embedding metadata by embedding ID"""
        query = """
        DELETE FROM embeddings_metadata
        WHERE embedding_id = %s
        """
        return self.db.execute_update(query, (embedding_id,))
        
    def get_embedding_metadata(self, embedding_id):
        """Get metadata for a specific embedding"""
        query = """
        SELECT embedding_id, user_id, created_at
        FROM embeddings_metadata
        WHERE embedding_id = %s
        """
        result = self.db.execute_query(query, (embedding_id,))
        return result[0] if result else None