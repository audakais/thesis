-- Disable foreign key checks for a clean start if tables already exist
SET session_replication_role = 'replica';

-- Drop tables in a specific order to avoid foreign key constraints
DROP TABLE IF EXISTS Measurements;
DROP TABLE IF EXISTS Biomarkers;
DROP TABLE IF EXISTS Samples;
DROP TABLE IF EXISTS SampleTypes;
DROP TABLE IF EXISTS Patients;
DROP TABLE IF EXISTS Individuals;

-- Enable foreign key checks
SET session_replication_role = 'origin';


-- Individuals: demographic and status data for each study participant.
CREATE TABLE Individuals (
    individual_id SERIAL PRIMARY KEY,
    individual_uuid UUID UNIQUE NOT NULL,
    gender VARCHAR(20),
    age INT,
    vital_status VARCHAR(20),
    survival_months DECIMAL(10, 2),
    ethnicity VARCHAR(50),
    race VARCHAR(50),
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Patients: cancer-specific data for individuals who are patients.
CREATE TABLE Patients (
    patient_id SERIAL PRIMARY KEY,
    individual_id INT UNIQUE NOT NULL,
    cancer_type VARCHAR(255) NOT NULL,
    diagnosis_date DATE,
    CONSTRAINT fk_individual_patient FOREIGN KEY (individual_id) REFERENCES Individuals(individual_id) ON DELETE CASCADE
);

-- SampleTypes: table of sample types.
CREATE TABLE SampleTypes (
    type_id SERIAL PRIMARY KEY,
    type_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT
);

-- Samples: links biological samples to individuals.
CREATE TABLE Samples (
    sample_id SERIAL PRIMARY KEY,
    individual_id INT NOT NULL,
    sample_uuid UUID UNIQUE NOT NULL,
    submitter_id VARCHAR(255) UNIQUE NOT NULL,
    sample_type_id INT NOT NULL,
    tissue_site VARCHAR(100), -- Added missing column
    collection_date DATE,
    CONSTRAINT fk_individual_sample FOREIGN KEY (individual_id) REFERENCES Individuals(individual_id) ON DELETE CASCADE,
    CONSTRAINT fk_sample_type FOREIGN KEY (sample_type_id) REFERENCES SampleTypes(type_id) ON DELETE RESTRICT
);

-- Biomarkers: information about genes or other measured biological markers.
CREATE TABLE Biomarkers (
    biomarker_id SERIAL PRIMARY KEY,
    biomarker_name VARCHAR(255) NOT NULL,
    ensembl_id VARCHAR(50) UNIQUE,
    biomarker_type VARCHAR(50),
    description TEXT
);

-- Measurements: the actual measurement data.
CREATE TABLE Measurements (
    measurement_id SERIAL PRIMARY KEY,
    sample_id INT NOT NULL,
    biomarker_id INT NOT NULL,
    measurement_value DOUBLE PRECISION NOT NULL,
    measurement_type VARCHAR(50) NOT NULL,
    CONSTRAINT fk_sample_measurement FOREIGN KEY (sample_id) REFERENCES Samples(sample_id) ON DELETE CASCADE,
    CONSTRAINT fk_biomarker_measurement FOREIGN KEY (biomarker_id) REFERENCES Biomarkers(biomarker_id) ON DELETE CASCADE,
    CONSTRAINT unique_measurement_per_type UNIQUE (sample_id, biomarker_id, measurement_type)
);

-- Indices for faster data retrieval
CREATE INDEX idx_individual_uuid ON Individuals(individual_uuid);
CREATE INDEX idx_patient_individual_id ON Patients(individual_id);
CREATE INDEX idx_sample_uuid ON Samples(sample_uuid);
CREATE INDEX idx_sample_submitter_id ON Samples(submitter_id);
CREATE INDEX idx_biomarker_ensembl_id ON Biomarkers(ensembl_id);
CREATE INDEX idx_measurement_sample_biomarker ON Measurements(sample_id, biomarker_id);