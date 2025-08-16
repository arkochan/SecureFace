#!/usr/bin/env python3

import sys
import os
import argparse
import json

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import vector_db
from database.db import SecureFaceDB

def initialize_faiss():
    """Initialize the FAISS index"""
    print("🔧 Initializing FAISS index...")
    try:
        vector_db.init_index(dim=512, index_path="faiss_index.bin")
        print("✅ FAISS index initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize FAISS index: {e}")
        return False

def view_faiss_stats():
    """View FAISS database statistics"""
    print("📊 FAISS Database Statistics")
    print("=" * 40)
    
    try:
        stats = vector_db.get_index_stats()
        if not stats.get("initialized", False):
            print("⚠️ FAISS index is not initialized")
            print("💡 Tip: Run 'python faiss_cli.py init' to initialize")
            return
            
        print(f"Status: {'Active' if stats['initialized'] else 'Inactive'}")
        print(f"Dimension: {stats.get('dimension', 'N/A')}")
        print(f"Total Vectors: {stats.get('total_vectors', 0)}")
        print(f"Next Embedding ID: {stats.get('next_embedding_id', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error getting FAISS stats: {e}")

def view_all_embeddings():
    """View all embeddings in the FAISS database"""
    print("🔍 All Embeddings in FAISS Database")
    print("=" * 50)
    
    try:
        stats = vector_db.get_index_stats()
        if not stats.get("initialized", False) or stats.get("total_vectors", 0) == 0:
            print("📭 No embeddings found in the database")
            return
            
        total_vectors = stats.get("total_vectors", 0)
        print(f"Total embeddings: {total_vectors}")
        print("\n📋 Embedding Details:")
        print("-" * 50)
        
        # Get metadata for all embeddings
        with SecureFaceDB() as db:
            all_embeddings = db.get_all_embeddings_metadata()
            
        if all_embeddings:
            for embedding in all_embeddings:
                print(f"FAISS ID: {embedding['embedding_id']}, User ID: {embedding['user_id']}, Created: {embedding['created_at']}")
        else:
            print("📭 No embedding metadata found")
            
    except Exception as e:
        print(f"❌ Error viewing embeddings: {e}")

def view_user_embeddings(user_id):
    """View embeddings for a specific user"""
    print(f"🔍 Embeddings for User ID: {user_id}")
    print("=" * 40)
    
    try:
        # Get embeddings for specific user from database
        with SecureFaceDB() as db:
            user_embeddings = db.get_embeddings_by_user(user_id)
            
        if user_embeddings:
            print(f"Found {len(user_embeddings)} embedding(s) for user {user_id}:")
            for embedding in user_embeddings:
                print(f"  FAISS ID: {embedding['embedding_id']}, Created: {embedding['created_at']}")
        else:
            print(f"📭 No embeddings found for user {user_id}")
            
    except Exception as e:
        print(f"❌ Error viewing user embeddings: {e}")

def remove_embedding(faiss_id):
    """Remove a specific embedding by FAISS ID (not directly supported by FAISS)"""
    print(f"⚠️ FAISS doesn't support removing individual vectors")
    print(f"💡 To remove embedding ID {faiss_id}, you would need to:")
    print(f"   1. Rebuild the index without that vector")
    print(f"   2. Remove the metadata from the database")
    print(f"   3. Save the new index")
    
    try:
        # Remove metadata from database
        with SecureFaceDB() as db:
            result = db.delete_embedding_metadata(faiss_id)
            
        if result:
            print(f"✅ Metadata for embedding {faiss_id} removed from database")
        else:
            print(f"❌ Failed to remove metadata for embedding {faiss_id}")
            
    except Exception as e:
        print(f"❌ Error removing embedding metadata: {e}")

def remove_user_embeddings(user_id):
    """Remove all embeddings for a specific user"""
    print(f"🗑️ Removing all embeddings for User ID: {user_id}")
    print("=" * 45)
    
    try:
        # Get all embeddings for this user
        with SecureFaceDB() as db:
            user_embeddings = db.get_embeddings_by_user(user_id)
            
        if not user_embeddings:
            print(f"📭 No embeddings found for user {user_id}")
            return
            
        print(f"Found {len(user_embeddings)} embedding(s) for user {user_id}")
        
        # Remove metadata for each embedding
        removed_count = 0
        for embedding in user_embeddings:
            faiss_id = embedding['embedding_id']
            result = db.delete_embedding_metadata(faiss_id)
            if result:
                removed_count += 1
                print(f"✅ Removed metadata for embedding {faiss_id}")
            else:
                print(f"❌ Failed to remove metadata for embedding {faiss_id}")
                
        print(f"\n📊 Summary: {removed_count}/{len(user_embeddings)} embeddings removed from database")
        print("💡 Note: FAISS index still contains the vectors. To fully remove them, rebuild the index.")
        
    except Exception as e:
        print(f"❌ Error removing user embeddings: {e}")

def export_faiss_index(output_path):
    """Export the FAISS index to a file"""
    print(f"💾 Exporting FAISS index to: {output_path}")
    print("=" * 40)
    
    try:
        vector_db.save_index(output_path)
        print("✅ FAISS index exported successfully")
    except Exception as e:
        print(f"❌ Error exporting FAISS index: {e}")

def import_faiss_index(input_path):
    """Import a FAISS index from a file"""
    print(f"📥 Importing FAISS index from: {input_path}")
    print("=" * 40)
    
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return
        
    try:
        # Reinitialize with the new index
        vector_db.init_index(dim=512, index_path=input_path)
        print("✅ FAISS index imported successfully")
        
        # Show stats
        stats = vector_db.get_index_stats()
        print(f"📊 New index has {stats.get('total_vectors', 0)} vectors")
        
    except Exception as e:
        print(f"❌ Error importing FAISS index: {e}")

def view_embedding_details(faiss_id):
    """View details of a specific embedding"""
    print(f"🔍 Details for Embedding ID: {faiss_id}")
    print("=" * 40)
    
    try:
        # Get metadata for specific embedding
        with SecureFaceDB() as db:
            embedding = db.get_embedding_metadata(faiss_id)
            
        if embedding:
            print(f"FAISS ID: {embedding['embedding_id']}")
            print(f"User ID: {embedding['user_id']}")
            print(f"Created At: {embedding['created_at']}")
            
            # Try to get user details
            try:
                user = db.get_user_by_id(embedding['user_id'])
                if user:
                    print(f"User Name: {user['full_name']}")
                    print(f"Department: {user['department']}")
                else:
                    print("User: Not found")
            except Exception:
                print("User: Error retrieving user details")
        else:
            print(f"📭 No embedding found with ID {faiss_id}")
            
    except Exception as e:
        print(f"❌ Error viewing embedding details: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="FAISS Database Management CLI for SecureFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python faiss_cli.py init                    # Initialize FAISS database
  python faiss_cli.py stats                   # View database statistics
  python faiss_cli.py view                    # View all embeddings
  python faiss_cli.py view --user 12          # View embeddings for user 12
  python faiss_cli.py view --id 5             # View details for embedding 5
  python faiss_cli.py remove --user 12        # Remove all embeddings for user 12
  python faiss_cli.py remove --id 5           # Remove metadata for embedding 5
  python faiss_cli.py export index.bin        # Export FAISS index
  python faiss_cli.py import index.bin        # Import FAISS index
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    subparsers.add_parser('init', help='Initialize FAISS database')
    
    # Stats command
    subparsers.add_parser('stats', help='View FAISS database statistics')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View embeddings')
    view_parser.add_argument('--user', type=int, help='View embeddings for specific user')
    view_parser.add_argument('--id', type=int, help='View details for specific embedding')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove embeddings')
    remove_parser.add_argument('--user', type=int, help='Remove all embeddings for specific user')
    remove_parser.add_argument('--id', type=int, help='Remove specific embedding by FAISS ID (metadata only)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export FAISS index')
    export_parser.add_argument('output', help='Output file path')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import FAISS index')
    import_parser.add_argument('input', help='Input file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize FAISS (try to load existing or create new)
    initialize_faiss()
    
    # Execute command
    if args.command == 'init':
        print("✅ FAISS database is ready")
        
    elif args.command == 'stats':
        view_faiss_stats()
        
    elif args.command == 'view':
        if args.user:
            view_user_embeddings(args.user)
        elif args.id:
            view_embedding_details(args.id)
        else:
            view_all_embeddings()
            
    elif args.command == 'remove':
        if args.user:
            remove_user_embeddings(args.user)
        elif args.id:
            remove_embedding(args.id)
        else:
            print("❌ Please specify --user or --id to remove")
            parser.print_help()
            
    elif args.command == 'export':
        export_faiss_index(args.output)
        
    elif args.command == 'import':
        import_faiss_index(args.input)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()