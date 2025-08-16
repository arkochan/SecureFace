-- init.sql
-- This script initializes the database tables for SecureFace

-- Wrap in transaction for atomicity
BEGIN;

-- ======================================================
-- ROLES TABLE
-- Stores user roles with access levels
-- ======================================================
CREATE TABLE IF NOT EXISTS roles (
    role_id BIGSERIAL PRIMARY KEY,
    role_name VARCHAR(100) NOT NULL UNIQUE,
    access_level INT NOT NULL, -- Higher number = more privileges
    description TEXT
);

-- ======================================================
-- USERS TABLE
-- Stores user information and references roles
-- ======================================================
CREATE TABLE IF NOT EXISTS users (
    user_id BIGSERIAL PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    role_id BIGINT REFERENCES roles(role_id),
    department VARCHAR(100),
    image_path VARCHAR(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ======================================================
-- RECOGNITION LOGS TABLE
-- Tracks all face recognition events
-- ======================================================
CREATE TABLE IF NOT EXISTS recognition_logs (
    log_id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(user_id),
    camera_id BIGINT,
    recognition_result VARCHAR(50) NOT NULL, -- 'ALLOWED', 'BLOCKED', 'UNKNOWN'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence_score NUMERIC(5,4)
);

-- ======================================================
-- DEFAULT ROLES INSERTION
-- Adds standard roles if they don't exist
-- ======================================================
INSERT INTO roles (role_name, access_level, description)
VALUES 
    ('Admin', 100, 'Administrator with full access'),
    ('Security', 50, 'Security personnel with access to logs'),
    ('Employee', 10, 'Regular employee with basic access')
ON CONFLICT (role_name) DO NOTHING;

-- ======================================================
-- INDEXES FOR PERFORMANCE
-- Recommended indexes for frequently queried columns
-- ======================================================
-- CREATE INDEX IF NOT EXISTS idx_users_role_id ON users(role_id);
-- CREATE INDEX IF NOT EXISTS idx_recognition_logs_user_id ON recognition_logs(user_id);
-- CREATE INDEX IF NOT EXISTS idx_recognition_logs_timestamp ON recognition_logs(timestamp);

COMMIT;