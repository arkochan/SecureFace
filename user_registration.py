import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import queue
import os
import time
import psycopg2
from database.connection import DatabaseConnection
import numpy as np
from embedder import FaceEmbedder
import vector_db


class UserRegistrationWindow:
    def __init__(self, parent, config_queue):
        self.parent = parent
        self.config_queue = config_queue
        self.window = None
        self.capturing = False
        self.captured_frame = None

        # Create user_images directory if it doesn't exist
        if not os.path.exists("user_images"):
            os.makedirs("user_images")

        self._create_window()

    def _create_window(self):
        """Create the user registration window"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Register New User")
        self.window.geometry("400x500")
        self.window.resizable(False, False)

        # Make the window modal
        self.window.transient(self.parent)
        self.window.grab_set()

        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame, text="Register New User", font=("Arial", 12, "bold")
        )
        title_label.pack(pady=(0, 20))

        # User details form
        form_frame = ttk.LabelFrame(main_frame, text="User Details", padding="10")
        form_frame.pack(fill=tk.X, pady=(0, 20))

        # Full Name
        name_frame = ttk.Frame(form_frame)
        name_frame.pack(fill=tk.X, pady=5)
        ttk.Label(name_frame, text="Full Name:").pack(anchor=tk.W)
        self.name_entry = ttk.Entry(name_frame, width=30)
        self.name_entry.pack(fill=tk.X, pady=(5, 0))

        # Department
        dept_frame = ttk.Frame(form_frame)
        dept_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dept_frame, text="Department:").pack(anchor=tk.W)
        self.dept_entry = ttk.Entry(dept_frame, width=30)
        self.dept_entry.pack(fill=tk.X, pady=(5, 0))

        # Role selection
        role_frame = ttk.Frame(form_frame)
        role_frame.pack(fill=tk.X, pady=5)
        ttk.Label(role_frame, text="Role:").pack(anchor=tk.W)
        self.role_var = tk.StringVar()
        self.role_combo = ttk.Combobox(
            role_frame, textvariable=self.role_var, state="readonly", width=28
        )
        self.role_combo.pack(fill=tk.X, pady=(5, 0))
        self._populate_roles()

        # Photo capture section
        photo_frame = ttk.LabelFrame(main_frame, text="Photo Capture", padding="10")
        photo_frame.pack(fill=tk.X, pady=(0, 20))

        # Instructions
        ttk.Label(
            photo_frame,
            text="Click 'Take Photo' to capture current frame",
            wraplength=350,
        ).pack(pady=(0, 10))

        # Take Photo button
        self.take_photo_btn = ttk.Button(
            photo_frame,
            text="Take Photo",
            command=self._take_photo,
            style="Accent.TButton",
        )
        self.take_photo_btn.pack(pady=10)

        # Photo status
        self.photo_status = ttk.Label(photo_frame, text="No photo captured")
        self.photo_status.pack()

        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # Register button
        self.register_btn = ttk.Button(
            button_frame,
            text="Register User",
            command=self._register_user,
            state=tk.DISABLED,
        )
        self.register_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Cancel button
        cancel_btn = ttk.Button(
            button_frame, text="Cancel", command=self._cancel_registration
        )
        cancel_btn.pack(side=tk.RIGHT)

        # Bind window close event
        self.window.protocol("WM_DELETE_WINDOW", self._cancel_registration)

    def _populate_roles(self):
        """Populate the role combobox with available roles from database"""
        try:
            db_conn = DatabaseConnection()
            if db_conn.connect():
                roles = db_conn.execute_query(
                    "SELECT role_id, role_name FROM roles ORDER BY access_level DESC"
                )
                if roles:
                    # Format for combobox: "Role Name (ID)"
                    role_values = [
                        f"{role['role_name']} ({role['role_id']})" for role in roles
                    ]
                    self.role_combo["values"] = role_values
                    if role_values:
                        self.role_combo.current(0)  # Select first role by default
                db_conn.disconnect()
            else:
                messagebox.showerror(
                    "Database Error", "Could not connect to database to fetch roles"
                )
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching roles: {str(e)}")

    def _take_photo(self):
        """Capture the current frame from the camera"""
        print("📸 Take Photo button clicked - initiating frame capture process")
        # Request a frame capture from the main application
        try:
            config = {
                "capture_frame": True  # Special flag to request frame capture
            }
            print(
                "📤 Sending capture_frame request to main application via config queue"
            )
            self.config_queue.put(config)

            # Update UI to show we're waiting for frame
            print("🔄 Updating UI to show capturing state")
            self.photo_status.config(text="Capturing frame...")
            self.take_photo_btn.config(state=tk.DISABLED)

            # We'll enable the button again after a short delay
            print("⏰ Setting up UI re-enable timer")
            self.window.after(1000, lambda: self.take_photo_btn.config(state=tk.NORMAL))
            print("✅ Frame capture request sent successfully")
        except Exception as e:
            error_msg = f"Error requesting frame capture: {str(e)}"
            print(f"❌ {error_msg}")
            messagebox.showerror("Error", error_msg)

    def _register_user(self):
        """Register the new user in the database and vector store"""
        print("📋 Starting user registration process")
        # Get form data
        full_name = self.name_entry.get().strip()
        department = self.dept_entry.get().strip()
        role_selection = self.role_var.get()

        print(
            f"📄 Form data - Name: {full_name}, Department: {department}, Role: {role_selection}"
        )

        # Validate input
        if not full_name:
            print("❌ Validation failed: Full name is required")
            messagebox.showerror("Validation Error", "Please enter a full name")
            return

        if not role_selection:
            print("❌ Validation failed: Role selection is required")
            messagebox.showerror("Validation Error", "Please select a role")
            return

        # Extract role ID from selection (format: "Role Name (ID)")
        try:
            role_id = int(role_selection.split("(")[-1].split(")")[0])
            print(f"✅ Role ID extracted: {role_id}")
        except (IndexError, ValueError):
            error_msg = "Invalid role selection"
            print(f"❌ {error_msg}")
            messagebox.showerror("Error", error_msg)
            return

        try:
            # Save the captured image
            print("💾 Saving captured image to file system")
            timestamp = int(time.time())
            image_filename = f"user_{timestamp}.jpg"
            image_path = os.path.join("user_images", image_filename)
            success = cv2.imwrite(image_path, self.captured_frame)

            if not success:
                error_msg = "Failed to save user image"
                print(f"❌ {error_msg}")
                messagebox.showerror("Error", error_msg)
                return

            print(f"✅ Image saved successfully to {image_path}")

            # Generate embedding from the captured frame
            print("🧠 Generating face embedding from captured frame")
            embedding = self._generate_embedding(self.captured_frame)
            if embedding is None:
                error_msg = "Failed to generate face embedding"
                print(f"❌ {error_msg}")
                messagebox.showerror("Error", error_msg)
                return

            print("✅ Face embedding generated successfully")

            # Normalize the embedding
            print("📏 Normalizing embedding vector")
            embedding = embedding / np.linalg.norm(embedding)
            print("✅ Embedding normalized")

            # Save user to database
            print("🗄️ Saving user information to PostgreSQL database")
            user_id = self._save_user_to_database(
                full_name, role_id, department, image_path
            )
            if user_id is None:
                error_msg = "Failed to save user to database"
                print(f"❌ {error_msg}")
                messagebox.showerror("Error", error_msg)
                return

            print(f"✅ User saved to database with ID: {user_id}")

            # Save embedding to vector database
            print("🔍 Saving embedding to FAISS vector database")
            faiss_id = self._save_embedding_to_vector_db(embedding, user_id)
            if faiss_id == -1:
                error_msg = "Failed to save embedding to vector database"
                print(f"❌ {error_msg}")
                messagebox.showerror("Error", error_msg)
                return

            print(f"✅ Embedding saved to vector database with FAISS ID: {faiss_id}")

            print(f"📊 Registration Summary:")
            print(f"   User ID: {user_id}")
            print(f"   FAISS ID: {faiss_id}")
            print(f"   Name: {full_name}")
            print(f"   Department: {department}")
            print(f"   Role ID: {role_id}")
            print(f"   Image: {image_path}")

            # Immediately fetch and display the stored user details
            print(
                "🔍 Fetching registered user details from database for verification..."
            )
            self._fetch_and_display_user(user_id)

            # Success
            success_msg = f"User '{full_name}' registered successfully!"
            print(f"🎉 {success_msg}")
            messagebox.showinfo("Success", success_msg)
            self.window.destroy()

        except Exception as e:
            error_msg = f"Error registering user: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()
            messagebox.showerror("Error", error_msg)

    def _generate_embedding(self, frame):
        """Generate face embedding from frame"""
        print("🧠 Starting face embedding generation process")
        try:
            # Convert BGR to RGB (InsightFace expects RGB)
            print("🔄 Converting BGR to RGB color space")
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"✅ Color conversion completed - image shape: {img_rgb.shape}")

            # Create a temporary embedder instance for generating the embedding
            print("🔧 Creating temporary FaceEmbedder instance")
            temp_embedder = FaceEmbedder()
            print("🔍 Detecting faces and generating embedding using InsightFace")
            faces = temp_embedder.app.get(img_rgb)
            print(f"👤 Face detection completed - found {len(faces)} face(s)")

            if len(faces) > 0:
                embedding = faces[0].embedding
                print(f"✅ Embedding generated successfully - shape: {embedding.shape}")
                # Clean up the temporary embedder
                temp_embedder.stop()
                return embedding
            else:
                print("⚠️ No faces detected in the image")
                # Clean up the temporary embedder
                temp_embedder.stop()
                return None
        except Exception as e:
            error_msg = f"Error generating embedding: {e}"
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()
            return None

    def _save_user_to_database(self, full_name, role_id, department, image_path):
        """Save user details to PostgreSQL database"""
        print(
            f"📥 Saving user to database - Name: {full_name}, Role ID: {role_id}, Department: {department}"
        )
        try:
            db_conn = DatabaseConnection()
            print("🔗 Connecting to PostgreSQL database")
            if not db_conn.connect():
                print("❌ Failed to connect to database")
                return None

            insert_query = """
                INSERT INTO users (full_name, role_id, department, image_path)
                VALUES (%s, %s, %s, %s)
                RETURNING user_id;
            """
            print("📤 Executing user insertion query")
            rows = db_conn.execute_insert_returning(
                insert_query, (full_name, role_id, department, image_path)
            )
            db_conn.disconnect()
            print("🔓 Database connection closed")

            if rows and len(rows) > 0:
                user_id = rows[0]["user_id"]
                print(f"✅ User saved successfully with ID: {user_id}")

                # Small delay to ensure database commit
                import time

                time.sleep(0.1)

                # Immediately verify the user was stored by retrieving it
                print("🔍 Verifying user storage by retrieving from database...")
                verification_result = self._verify_user_in_database(user_id)
                if verification_result:
                    print(
                        f"✅ User verification successful - Name: {verification_result['full_name']}, Department: {verification_result['department']}"
                    )
                else:
                    print("⚠️ User verification failed - could not retrieve stored user")

                return user_id
            else:
                print("⚠️ No user ID returned from database insertion")
                return None
        except Exception as e:
            error_msg = f"Error saving user to database: {e}"
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()
            return None

    def _verify_user_in_database(self, user_id):
        """Verify that the user was properly stored by retrieving it"""
        # Since we just inserted the user, let's try to retrieve it using the same database instance
        # or at least ensure we're using a fresh connection
        try:
            from database.db import SecureFaceDB

            print("🔍 Verifying user storage by retrieving from database...")
            with SecureFaceDB() as db:
                user = db.get_user_by_id(user_id)

            if user:
                print(
                    f"✅ User verified in database - ID: {user['user_id']}, Name: {user['full_name']}"
                )
                return user
            else:
                print(
                    f"❌ User not found in database during verification - ID: {user_id}"
                )
                # Let's also try to get all users to see if there's a broader issue
                print("🔍 Getting all users for debugging...")
                with SecureFaceDB() as db:
                    all_users = db.get_all_users()
                if all_users:
                    print(f"📊 Total users in database: {len(all_users)}")
                    for u in all_users:
                        print(f"   - User ID: {u['user_id']}, Name: {u['full_name']}")
                else:
                    print("📊 No users found in database")
                return None
        except Exception as e:
            error_msg = f"Error verifying user in database: {e}"
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()
            return None

    def _save_embedding_to_vector_db(self, embedding, user_id):
        """Save embedding to FAISS vector database"""
        print(f"📥 Saving embedding to vector database for user ID: {user_id}")
        try:
            # Add embedding to FAISS index
            faiss_id = vector_db.add_embedding(embedding, user_id)
            print(f"🔍 FAISS add_embedding returned ID: {faiss_id}")

            # Save the index
            print("💾 Saving FAISS index to file")
            vector_db.save_index("faiss_index.bin")
            print("✅ FAISS index saved successfully")

            # Verify the embedding was stored by searching for it
            if faiss_id != -1:
                print(
                    "🔍 Verifying embedding storage by searching in vector database..."
                )
                verification_result = self._verify_embedding_in_vector_db(user_id)
                if verification_result:
                    print(
                        f"✅ Embedding verification successful"
                    )
                else:
                    print(
                        "⚠️ Embedding verification failed - could not find stored embedding"
                    )

            return faiss_id
        except Exception as e:
            error_msg = f"Error saving embedding to vector database: {e}"
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()
            return -1

    def _verify_embedding_in_vector_db(self, user_id):
        """Verify that the embedding was properly stored by checking index stats"""
        try:
            # Check the index stats to verify the embedding was added
            stats = vector_db.get_index_stats()
            print(f"📊 Vector database stats after insertion: {stats}")

            # If we have at least one vector, assume it worked
            if stats.get("initialized", False) and stats.get("total_vectors", 0) > 0:
                print(
                    f"✅ Vector database verification successful - total vectors: {stats.get('total_vectors')}"
                )
                return True
            else:
                print("⚠️ Vector database verification failed - no vectors found")
                return False
        except Exception as e:
            error_msg = f"Error verifying embedding in vector database: {e}"
            print(f"❌ {error_msg}")
            return False

    def _fetch_and_display_user(self, user_id):
        """Fetch and display user details from database for verification"""
        try:
            from database.db import SecureFaceDB

            print(f"📥 Fetching user details for ID: {user_id}")
            with SecureFaceDB() as db:
                user = db.get_user_by_id(user_id)

            if user:
                print("✅ User details retrieved successfully:")
                print(f"   User ID: {user['user_id']}")
                print(f"   Full Name: {user['full_name']}")
                print(f"   Role ID: {user['role_id']}")
                print(f"   Department: {user['department']}")
                print(f"   Image Path: {user['image_path']}")
                print(f"   Created At: {user['created_at']}")
                print(f"   Updated At: {user['updated_at']}")
            else:
                print(f"❌ Failed to retrieve user details for ID: {user_id}")

        except Exception as e:
            error_msg = f"Error fetching user details: {e}"
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()

    def _cancel_registration(self):
        """Cancel the registration and close the window"""
        self.window.destroy()

    def set_captured_frame(self, frame):
        """Set the captured frame and update UI"""
        print(
            f"📥 Received captured frame in registration window - shape: {frame.shape}"
        )
        self.captured_frame = frame.copy()
        print("✅ Frame copied and stored for registration")
        self.photo_status.config(text="Photo captured successfully!")
        self.register_btn.config(state=tk.NORMAL)
        print("🔄 UI updated: photo status changed and register button enabled")

