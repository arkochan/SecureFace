import psycopg2
from psycopg2.extras import RealDictCursor
from .config import DatabaseConfig

class DatabaseConnection:
    """Handles database connections and operations"""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Establishes a connection to the database"""
        try:
            self.connection = psycopg2.connect(
                host=DatabaseConfig.HOST,
                port=DatabaseConfig.PORT,
                database=DatabaseConfig.DATABASE,
                user=DatabaseConfig.USER,
                password=DatabaseConfig.PASSWORD,
                cursor_factory=RealDictCursor
            )
            self.cursor = self.connection.cursor()
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self):
        """Closes the database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
    
    def execute_query(self, query, params=None):
        """Executes a SELECT query and returns the results"""
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
    
    def execute_update(self, query, params=None):
        """Executes an INSERT/UPDATE/DELETE query"""
        try:
            self.cursor.execute(query, params)
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error executing update: {e}")
            self.connection.rollback()
            return False
    
    def execute_many(self, query, params_list):
        """Executes a query with multiple parameter sets"""
        try:
            self.cursor.executemany(query, params_list)
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error executing many: {e}")
            self.connection.rollback()
            return False