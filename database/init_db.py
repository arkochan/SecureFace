import psycopg2
from .config import DatabaseConfig

def create_tables():
    """Create all necessary tables in the database"""
    
    # Connect to the database
    conn = psycopg2.connect(
        host=DatabaseConfig.HOST,
        port=DatabaseConfig.PORT,
        database=DatabaseConfig.DATABASE,
        user=DatabaseConfig.USER,
        password=DatabaseConfig.PASSWORD
    )
    
    cursor = conn.cursor()
    
    # Create roles table first (referenced by users)
    create_roles_table = """
    CREATE TABLE IF NOT EXISTS roles (
        role_id BIGSERIAL PRIMARY KEY,
        role_name VARCHAR(100) NOT NULL UNIQUE,
        access_level INT NOT NULL, -- Higher number = more privileges
        description TEXT
    );
    """
    
    # Create users table
    create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        user_id BIGSERIAL PRIMARY KEY,
        full_name VARCHAR(255) NOT NULL,
        role_id BIGINT REFERENCES roles(role_id),
        department VARCHAR(100),
        image_path VARCHAR(1024),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Create recognition_logs table
    create_logs_table = """
    CREATE TABLE IF NOT EXISTS recognition_logs (
        log_id BIGSERIAL PRIMARY KEY,
        user_id BIGINT REFERENCES users(user_id),
        camera_id BIGINT,
        recognition_result VARCHAR(50) NOT NULL, -- 'ALLOWED', 'BLOCKED', 'UNKNOWN'
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        confidence_score NUMERIC(5,4)
    );
    """
    
    try:
        # Execute table creation queries
        cursor.execute(create_roles_table)
        cursor.execute(create_users_table)
        cursor.execute(create_logs_table)
        
        # Commit changes
        conn.commit()
        print("Tables created successfully!")
        
        # Insert some default roles if they don't exist
        insert_default_roles = """
        INSERT INTO roles (role_name, access_level, description)
        VALUES 
        ('Admin', 100, 'Administrator with full access'),
        ('Security', 50, 'Security personnel with access to logs'),
        ('Employee', 10, 'Regular employee with basic access')
        ON CONFLICT (role_name) DO NOTHING;
        """
        
        cursor.execute(insert_default_roles)
        conn.commit()
        print("Default roles inserted successfully!")
        
    except Exception as e:
        print(f"Error creating tables: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    create_tables()