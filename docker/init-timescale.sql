-- Initialize TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create hypertables for time-series data
-- This will be done by our application migrations, but we can prepare the database

-- Set up proper permissions
GRANT ALL PRIVILEGES ON DATABASE amperedata TO amperedata;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS timeseries;
CREATE SCHEMA IF NOT EXISTS metadata;
CREATE SCHEMA IF NOT EXISTS auth;

-- Grant permissions on schemas
GRANT ALL ON SCHEMA timeseries TO amperedata;
GRANT ALL ON SCHEMA metadata TO amperedata;
GRANT ALL ON SCHEMA auth TO amperedata;

-- Log initialization
SELECT 'TimescaleDB initialized successfully' as status;
