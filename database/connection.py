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
                cursor_factory=RealDictCursor,
            )
            self.cursor = self.connection.cursor()
            print("Database connection established successfully")
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            self.connection = None
            self.cursor = None
            return False

    def disconnect(self):
        """Closes the database connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            print("Database connection closed")
        except Exception as e:
            print(f"Error closing database connection: {e}")
        finally:
            self.cursor = None
            self.connection = None

    def _ensure_connection(self):
        """Ensures database connection exists"""
        if not self.connection or self.connection.closed:
            raise Exception("No active database connection. Call connect() first.")

    def execute_query(self, query, params=None):
        """Executes a SELECT query and returns the results"""
        try:
            self._ensure_connection()
            self.cursor.execute(query, params)
            result = self.cursor.fetchall()
            print(f"Query executed successfully, returned {len(result)} rows")
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Query: {query}")
            print(f"Params: {params}")
            return None

    def execute_update(self, query, params=None):
        """Executes an INSERT/UPDATE/DELETE query"""
        try:
            self._ensure_connection()
            self.cursor.execute(query, params)
            rows_affected = self.cursor.rowcount
            self.connection.commit()
            print(f"Update executed successfully, {rows_affected} rows affected")
            return rows_affected
        except Exception as e:
            print(f"Error executing update: {e}")
            print(f"Query: {query}")
            print(f"Params: {params}")
            if self.connection:
                self.connection.rollback()
            return False

    def execute_insert_returning(self, query, params=None):
        """Executes an INSERT ... RETURNING query"""
        try:
            self._ensure_connection()
            self.cursor.execute(query, params)
            result = self.cursor.fetchall()
            rows_affected = self.cursor.rowcount
            self.connection.commit()
            print(
                f"Insert with returning executed successfully, {rows_affected} rows inserted"
            )
            return result
        except Exception as e:
            print(f"Error executing insert returning: {e}")
            print(f"Query: {query}")
            print(f"Params: {params}")
            if self.connection:
                self.connection.rollback()
            return None

    def execute_many(self, query, params_list):
        """Executes a query with multiple parameter sets"""
        try:
            self._ensure_connection()
            self.cursor.executemany(query, params_list)
            rows_affected = self.cursor.rowcount
            self.connection.commit()
            print(
                f"Batch execute completed successfully, {rows_affected} rows affected"
            )
            return rows_affected
        except Exception as e:
            print(f"Error executing many: {e}")
            print(f"Query: {query}")
            print(f"Params list length: {len(params_list) if params_list else 0}")
            if self.connection:
                self.connection.rollback()
            return False

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
