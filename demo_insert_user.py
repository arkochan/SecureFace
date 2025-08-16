#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.connection import DatabaseConnection

def insert_demo_user():
    """Demo script to insert a demo user using DatabaseConnection"""
    
    print("🔧 Demo: Inserting a demo user using DatabaseConnection")
    print("=" * 50)
    
    # Create database connection
    db_conn = DatabaseConnection()
    
    try:
        # Connect to database
        print("🔌 Connecting to database...")
        if not db_conn.connect():
            print("❌ Failed to connect to database")
            return False
            
        print("✅ Connected to database successfully")
        
        # Insert a demo user using execute_insert_returning
        print("\n📥 Inserting demo user...")
        insert_query = """
        INSERT INTO users (full_name, role_id, department, image_path)
        VALUES (%s, %s, %s, %s)
        RETURNING user_id, full_name, created_at
        """
        
        demo_params = ("Demo User", 1, "IT Department", "user_images/demo.jpg")
        
        result = db_conn.execute_insert_returning(insert_query, demo_params)
        
        if result:
            user_id = result[0]['user_id']
            full_name = result[0]['full_name']
            created_at = result[0]['created_at']
            print(f"✅ Demo user inserted successfully!")
            print(f"   User ID: {user_id}")
            print(f"   Name: {full_name}")
            print(f"   Created at: {created_at}")
            
            # Verify the user was actually saved by querying it back
            print("\n🔍 Verifying user was saved...")
            verify_query = "SELECT user_id, full_name, department FROM users WHERE user_id = %s"
            verify_result = db_conn.execute_query(verify_query, (user_id,))
            
            if verify_result and len(verify_result) > 0:
                print(f"✅ Verification successful!")
                print(f"   User ID: {verify_result[0]['user_id']}")
                print(f"   Name: {verify_result[0]['full_name']}")
                print(f"   Department: {verify_result[0]['department']}")
                return True
            else:
                print("❌ Verification failed - user not found in database")
                return False
        else:
            print("❌ Failed to insert demo user")
            return False
            
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Always disconnect
        print("\n🔌 Disconnecting from database...")
        db_conn.disconnect()
        print("✅ Disconnected from database")

def show_all_users():
    """Show all users in the database"""
    print("\n📋 All users in database:")
    print("-" * 30)
    
    db_conn = DatabaseConnection()
    
    try:
        if not db_conn.connect():
            print("❌ Failed to connect to database")
            return
            
        query = "SELECT user_id, full_name, department, created_at FROM users ORDER BY user_id"
        result = db_conn.execute_query(query)
        
        if result:
            for user in result:
                print(f"ID: {user['user_id']}, Name: {user['full_name']}, Dept: {user['department']}, Created: {user['created_at']}")
        else:
            print("No users found")
            
    except Exception as e:
        print(f"❌ Error querying users: {e}")
    finally:
        db_conn.disconnect()

if __name__ == "__main__":
    print("🚀 Database Connection Demo Script")
    print("=" * 40)
    
    # Insert demo user
    success = insert_demo_user()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        # Show all users
        show_all_users()
    else:
        print("\n💥 Demo failed!")
        show_all_users()