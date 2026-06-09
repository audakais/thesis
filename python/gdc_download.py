import requests
import psycopg2
from psycopg2 import Error
from uuid import uuid4
import json
import pandas as pd
from datetime import datetime
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import io

# --- PostgreSQL Database Configuration ---
DB_HOST = "localhost"
DB_NAME = "cancerdata"
DB_USER = "postgres"  # Your PostgreSQL user
DB_PASSWORD = "password" 

# --- GDC Data Portal API Configuration ---
GDC_API_BASE = "https://api.gdc.cancer.gov/"
GDC_API_PROJECTS_ENDPOINT = GDC_API_BASE + "projects"
GDC_API_CASES_ENDPOINT = GDC_API_BASE + "cases"
GDC_API_FILES_ENDPOINT = GDC_API_BASE + "files"
GDC_API_DATA_ENDPOINT = GDC_API_BASE + "data"

HEADERS = {"Content-Type": "application/json"}

# --- Database Utility Functions ---

def connect_db():
    """
    Establishes a connection to the PostgreSQL database. Returns the connection object and the cursor.
    """
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.autocommit = False
        cursor = conn.cursor()
        print("PostgreSQL database connection established successfully.")
        return conn, cursor
    except Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None, None

def close_db(conn, cursor):
    """
    Closes the database connection and cursor.
    """
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    print("PostgreSQL database connection closed.")

def get_pk_column_name(table_name):
    """
    Returns the primary key column name for a given table. Handles special cases like 'sampletypes'.
    """
    if table_name == "sampletypes":
        return "sample_type_id"
    return f"{table_name[:-1]}_id"

def insert_data(cursor, table_name, columns, values):
    """
    Inserts a row of data into a specified table.
    """
    cols_str = ", ".join(columns)
    vals_str = ", ".join(["%s"] * len(values))
    
    query = f"INSERT INTO {table_name} ({cols_str}) VALUES ({vals_str})"
    
    if table_name in ["individuals", "patients", "samples", "biomarkers", "sampletypes", "measurements"]:
        pk_col_name = get_pk_column_name(table_name)
        query += f" RETURNING {pk_col_name}" 
    
    try:
        cursor.execute(query, values)
        if table_name in ["individuals", "patients", "samples", "biomarkers", "sampletypes", "measurements"]:
            inserted_id = cursor.fetchone()[0]
            return inserted_id
        return None
    except Error as e:
        if "duplicate key value violates unique constraint" in str(e):
            print(f"Warning: Duplicate detected during insertion into {table_name}. Skipped.")
        else:
            print(f"Error during insertion into {table_name}: {e}")
        cursor.connection.rollback()
        return None

def fetch_existing_id(cursor, table_name, column_name, value):
    """
    Searches for an existing ID in a table based on a column value.
    """
    pk_col_name = get_pk_column_name(table_name)
    query = f"SELECT {pk_col_name} FROM {table_name} WHERE {column_name} = %s"
    try:
        cursor.execute(query, (value,))
        result = cursor.fetchone()
        return result[0] if result else None
    except Error as e:
        print(f"Error during lookup in {table_name}: {e}")
        cursor.connection.rollback()
        return None

def fetch_individual_uuid(cursor, individual_id):
    """
    Retrieves the individual_uuid of an individual given their individual_id from the DB.
    """
    query = "SELECT individual_uuid FROM Individuals WHERE individual_id = %s"
    try:
        cursor.execute(query, (individual_id,))
        result = cursor.fetchone()
        return result[0] if result else "Unknown"
    except Error as e:
        print(f"Error retrieving individual_uuid for individual_id {individual_id}: {e}")
        cursor.connection.rollback()
        return "Unknown"

# --- Functions for Data Extraction from GDC Data Portal ---

def get_gdc_data(endpoint, filters=None, fields=None, size=100, limit=None):
    """
    Executes a generic query to the GDC Data Portal.
    """
    params = {
        "format": "json",
        "size": size,
        "from": 0
    }
    if filters:
        params["filters"] = json.dumps(filters)
    if fields:
        params["fields"] = ",".join(fields)

    all_data = []
    total_retrieved = 0
    total_available = "unknown"

    while True:
        try:
            response = requests.get(endpoint, headers=HEADERS, params=params)
            response.raise_for_status()
            data = response.json()

            hits = data.get("data", {}).get("hits", [])
            all_data.extend(hits)
            total_retrieved = len(all_data)

            if limit is not None and total_retrieved >= limit:
                print(f"Limit of {limit} reached for {endpoint}.")
                all_data = all_data[:limit]
                break

            pagination = data.get("data", {}).get("pagination", {})
            total_available = pagination.get("total", "unknown")
            
            print(f"Retrieved {total_retrieved} of {total_available} results from {endpoint}...")

            if pagination.get("page") * pagination.get("size") >= total_available or not hits:
                break 
            params["from"] += size

        except requests.exceptions.RequestException as e:
            print(f"Error during API request to {endpoint}: {e}")
            break
        except json.JSONDecodeError as e:
            print(f"JSON decoding error from {endpoint} response: {e}")
            print(f"Raw response: {response.text[:500]}...")
            break
    return all_data

def download_gdc_file(file_id, output_path):
    """
    Downloads a file from the GDC Data Portal.
    """
    download_url = f"{GDC_API_DATA_ENDPOINT}/{file_id}"
    print(f"Attempting to download {file_id} to {output_path}...")
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download of {file_id} completed.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error during file download {file_id}: {e}")
        return False

def extract_and_load_clinical_data(cursor, limit=None):
    """
    Extracts clinical data (cases) from GDC and loads it into the Individuals and Patients tables.
    """
    print("\n--- Extracting and loading clinical data (Individuals and Patients) ---")
    
    clinical_fields = [
        "case_id", "submitter_id", "demographic.gender", "demographic.age_at_diagnosis",
        "demographic.vital_status", "demographic.days_to_death", "demographic.days_to_last_follow_up",
        "demographic.ethnicity", "demographic.race",
        "diagnoses.primary_diagnosis", "diagnoses.tumor_stage", "diagnoses.diagnosis_datetime",
        "treatments.treatment_type"
    ]
    
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [
                "TCGA-BRCA", "TCGA-LUAD", "TCGA-KIRC", "TCGA-COAD", "TCGA-READ",
                "TCGA-LGG", "TCGA-GBM", "TCGA-OV", "TCGA-SKCM", "TCGA-THCA",
                "TCGA-LIHC", "TCGA-KID", "TCGA-PRAD", "TCGA-STAD", "TCGA-ESCA",
                "TCGA-BLCA", "TCGA-HNSC", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC"
            ]}}
        ]
    }

    cases_data = get_gdc_data(GDC_API_CASES_ENDPOINT, filters=filters, fields=clinical_fields, size=2000, limit=limit)

    if not cases_data:
        print("No clinical data retrieved. Check filters and API connection.")
        return False

    for case in cases_data:
        demographic = case.get("demographic", {})
        diagnoses = case.get("diagnoses", [{}])[0]
        treatments = case.get("treatments", []) 

        individual_uuid = case.get("case_id")
        submitter_id = case.get("submitter_id")

        survival_months = None
        if demographic.get("vital_status") == "Dead" and demographic.get("days_to_death") is not None:
            survival_months = round(demographic["days_to_death"] / 30.4375, 2)
        elif demographic.get("vital_status") == "Alive" and demographic.get("days_to_last_follow_up") is not None:
            survival_months = round(demographic["days_to_last_follow_up"] / 30.4375, 2)

        diagnosis_date = None
        if diagnoses.get("diagnosis_datetime"):
            try:
                diagnosis_date = datetime.strptime(diagnoses["diagnosis_datetime"].split('T')[0], '%Y-%m-%d').date()
            except ValueError:
                pass

        treatment_types = ", ".join(sorted(list(set([t.get("treatment_type", "Unknown") for t in treatments])))) if treatments else None
        if treatment_types == "Unknown":
            treatment_types = None

        ethnicity_val = demographic.get("race")
        if ethnicity_val:
            ethnicity_val = ethnicity_val.replace('_', ' ').title()
        else:
            ethnicity_val = demographic.get("ethnicity")
            if ethnicity_val:
                ethnicity_val = ethnicity_val.replace('_', ' ').title()
            else:
                ethnicity_val = "Unknown"

        cancer_type_val = diagnoses.get("primary_diagnosis")
        
        individual_id = fetch_existing_id(cursor, "individuals", "individual_uuid", individual_uuid)

        if not individual_id:
            individual_data = {
                "individual_uuid": individual_uuid,
                "gender": demographic.get("gender", "Unknown").capitalize(),
                "age": demographic.get("age_at_diagnosis"),
                "vital_status": demographic.get("vital_status"),
                "survival_months": survival_months,
                "ethnicity": ethnicity_val
            }
            
            cols = [k for k, v in individual_data.items() if v is not None]
            vals = [v for k, v in individual_data.items() if v is not None]

            individual_id = insert_data(cursor, "individuals", cols, vals)
            if individual_id:
                print(f"Inserted individual: {submitter_id if submitter_id else 'N/A'} (ID: {individual_id})")
        else:
            print(f"Individual {submitter_id if submitter_id else 'N/A'} (ID: {individual_id}) already exists. Skipped individual insertion.")

        if individual_id and cancer_type_val and cancer_type_val.strip() != "" and cancer_type_val != "Not Reported":
            existing_patient_record = fetch_existing_id(cursor, "patients", "individual_id", individual_id)
            if not existing_patient_record:
                patient_data = {
                    "individual_id": individual_id,
                    "cancer_type": cancer_type_val,
                    "diagnosis_date": diagnosis_date,
                    "tumor_stage": diagnoses.get("tumor_stage"),
                    "treatment_type": treatment_types
                }
                cols = [k for k, v in patient_data.items() if v is not None]
                vals = [v for k, v in patient_data.items() if v is not None]
        
                patient_record_id = insert_data(cursor, "patients", cols, vals)
                if patient_record_id:
                    print(f"  Inserted patient diagnosis record (ID: {patient_record_id}) for individual {submitter_id if submitter_id else 'N/A'}")
            else:
                print(f"  Patient diagnosis record already exists for individual {submitter_id if submitter_id else 'N/A'}. Skipped diagnosis insertion.")
        elif individual_id:
            print(f"  Individual {submitter_id if submitter_id else 'N/A'} has no primary diagnosis. Skipping patient diagnosis record insertion.")
    return True

def extract_and_load_biospecimen_data(cursor, limit=None):
    """
    Extracts biospecimen data (samples) from GDC and loads it into the DB.
    """
    print("\n--- Extracting and loading sample data (Samples) ---")
    biospecimen_fields = [
        "case_id", "samples.sample_id", "samples.submitter_id", "samples.sample_type",
        "samples.collection_datetime", "samples.tissue_type"
    ]
    
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [
                "TCGA-BRCA", "TCGA-LUAD", "TCGA-KIRC", "TCGA-COAD", "TCGA-READ",
                "TCGA-LGG", "TCGA-GBM", "TCGA-OV", "TCGA-SKCM", "TCGA-THCA",
                "TCGA-LIHC", "TCGA-KID", "TCGA-PRAD", "TCGA-STAD", "TCGA-ESCA",
                "TCGA-BLCA", "TCGA-HNSC", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC"
            ]}}
        ]
    }

    cases_data = get_gdc_data(GDC_API_CASES_ENDPOINT, filters=filters, fields=biospecimen_fields, size=2000, limit=limit)

    if not cases_data:
        print("No biospecimen data retrieved. Check filters and API connection.")
        return False

    for case in cases_data:
        individual_uuid = case.get("case_id")
        individual_id = fetch_existing_id(cursor, "individuals", "individual_uuid", individual_uuid)

        if not individual_id:
            print(f"Individual with UUID {individual_uuid} not found in DB for samples. Skipping associated samples.")
            continue
        
        individual_submitter_id = fetch_individual_uuid(cursor, individual_id)

        for sample in case.get("samples", []):
            sample_uuid = sample.get("sample_id")
            submitter_sample_id = sample.get("submitter_id")
            sample_type_name = sample.get("sample_type")
            collection_date = None
            if sample.get("collection_datetime"):
                try:
                    collection_date = datetime.strptime(sample["collection_datetime"].split('T')[0], '%Y-%m-%d').date()
                except ValueError:
                    pass

            tissue_site = sample.get("tissue_type")

            sample_type_id = fetch_existing_id(cursor, "sampletypes", "type_name", sample_type_name)
            if not sample_type_id:
                sample_type_id = insert_data(cursor, "sampletypes", ["type_name"], [sample_type_name])
                if sample_type_id:
                    print(f"Inserted sample type: {sample_type_name} (ID: {sample_type_id})")
                else:
                    sample_type_id = fetch_existing_id(cursor, "sampletypes", "type_name", sample_type_name)

            existing_sample_id = fetch_existing_id(cursor, "samples", "sample_uuid", sample_uuid)

            if not existing_sample_id and sample_type_id:
                sample_data = {
                    "sample_uuid": sample_uuid,
                    "individual_id": individual_id,
                    "sample_type_id": sample_type_id,
                    "collection_date": collection_date,
                    "tissue_site": tissue_site
                }
                cols = [k for k, v in sample_data.items() if v is not None]
                vals = [v for k, v in sample_data.items() if v is not None]

                sample_id = insert_data(cursor, "samples", cols, vals)
                if sample_id:
                    print(f"Inserted sample: {submitter_sample_id if submitter_sample_id else 'N/A'} (ID: {sample_id}) for individual {individual_submitter_id}")
            else:
                sample_id = existing_sample_id
                print(f"Sample {submitter_sample_id if submitter_sample_id else 'N/A'} (ID: {sample_id}) already exists. Skipped.")
    return True

def extract_and_load_molecular_data(cursor, limit=None):
    """
    Extracts molecular data (biomarkers and measurements) from GDC and loads it into the DB.
    """
    print("\n--- Extracting and loading molecular data (Biomarkers, Measurements) ---")

    file_filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [
                "TCGA-BRCA", "TCGA-LUAD", "TCGA-KIRC", "TCGA-COAD", "TCGA-READ",
                "TCGA-LGG", "TCGA-GBM", "TCGA-OV", "TCGA-SKCM", "TCGA-THCA",
                "TCGA-LIHC", "TCGA-KID", "TCGA-PRAD", "TCGA-STAD", "TCGA-ESCA",
                "TCGA-BLCA", "TCGA-HNSC", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC"
            ]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["TSV"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["RNA-Seq"]}}
        ]
    }
    file_fields = ["file_id", "file_name", "cases.samples.sample_id"]

    gene_expression_files_metadata = get_gdc_data(GDC_API_FILES_ENDPOINT, filters=file_filters, fields=file_fields, size=500, limit=limit)

    if not gene_expression_files_metadata:
        print("No expression file metadata retrieved. Check filters.")
        return False

    download_dir = "gdc_downloads"
    os.makedirs(download_dir, exist_ok=True)

    for file_meta in gene_expression_files_metadata:
        file_id = file_meta.get("file_id")
        file_name = file_meta.get("file_name")

        gdc_sample_id = None
        if file_meta.get("cases") and file_meta["cases"][0].get("samples"):
            for sample_gdc_info in file_meta["cases"][0]["samples"]:
                if sample_gdc_info.get("sample_id"):
                    gdc_sample_id = sample_gdc_info["sample_id"]
                    break

        if not gdc_sample_id:
            print(f"No GDC sample_id found for file {file_name}. Skipped.")
            continue

        sample_id_db = fetch_existing_id(cursor, "samples", "sample_uuid", gdc_sample_id)
        if not sample_id_db:
            print(f"GDC sample {gdc_sample_id} not found in DB. Skipping measurements for file {file_name}.")
            continue

        local_file_path = os.path.join(download_dir, file_name)
        print(f"Processing expression file: {file_name} (DB Sample ID: {sample_id_db})")

        if download_gdc_file(file_id, local_file_path):
            try:
                df_expression = pd.read_csv(local_file_path, sep='\t', header=0)

                df_expression = df_expression.rename(columns={
                    'gene_id': 'ensembl_id',
                    'tpm_unstranded': 'tpm',
                    'fpkm_unstranded': 'fpkm',
                    'unstranded_read_count': 'read_count'
                })
                
                df_expression['ensembl_id'] = df_expression['ensembl_id'].apply(lambda x: x.split('.')[0])
                
                if 'gene_type' not in df_expression.columns:
                    df_expression['gene_type'] = 'protein_coding'

                for col in ['tpm', 'fpkm', 'read_count']:
                    df_expression[col] = pd.to_numeric(df_expression[col], errors='coerce')
                
                df_expression.dropna(subset=['tpm', 'fpkm', 'read_count'], inplace=True)
                
                if df_expression.empty:
                    print(f"  Warning: File {file_name} has no valid numeric expression values after cleaning. Skipping.")
                    continue

                for index, row in df_expression.iterrows():
                    ensembl_id = row['ensembl_id']
                    gene_type = row['gene_type']
                    tpm_val = row['tpm']
                    fpkm_val = row['fpkm']
                    read_count_val = row['read_count']
                    
                    biomarker_id = fetch_existing_id(cursor, "biomarkers", "ensembl_id", ensembl_id)
                    if not biomarker_id:
                        biomarker_id = insert_data(
                            cursor, "biomarkers",
                            ["biomarker_name", "ensembl_id", "gene_type"],
                            [ensembl_id, ensembl_id, gene_type]
                        )
                        if biomarker_id:
                            print(f"  Inserted biomarker: {ensembl_id} (ID: {biomarker_id})")
                        else:
                            biomarker_id = fetch_existing_id(cursor, "biomarkers", "ensembl_id", ensembl_id)

                    if biomarker_id:
                        measurement_data = {
                            "sample_id": sample_id_db,
                            "biomarker_id": biomarker_id,
                            "tpm": tpm_val,
                            "fpkm": fpkm_val,
                            "read_count": read_count_val
                        }
                        cols = [k for k, v in measurement_data.items() if v is not None]
                        vals = [v for k, v in measurement_data.items() if v is not None]
                        
                        insert_data(cursor, "measurements", cols, vals)
                
                print(f"  Expression data from file {file_name} loaded into the DB.")

            except Exception as e:
                print(f"An error occurred during parsing or insertion of file {file_name}: {e}. Skipping this file.")
                continue
        else:
            print(f"Download of file {file_name} failed. Skipping parsing.")
    return True

# --- Function for exporting a single table to CSV ---
def export_table_to_csv(conn, table_name, output_filepath):
    """
    Exports a database table to a CSV file.
    """
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df.to_csv(output_filepath, index=False, encoding='utf-8')
        
        print(f"Table '{table_name}' successfully exported to '{output_filepath}'")
        return True
    except Error as e:
        print(f"Error during export of table '{table_name}': {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during export: {e}")
        return False

# --- New function for exporting the entire DB to a single CSV ---
def export_full_db_to_single_csv(conn, output_filepath):
    """
    Exports combined data from all relevant database tables into a single CSV file.
    """
    print(f"\n--- Exporting entire database to '{output_filepath}' ---")
    try:
        query = """
        SELECT
            i.individual_id,
            i.individual_uuid,
            i.gender,
            i.age,
            i.vital_status,
            i.survival_months,
            i.ethnicity,
            p.cancer_type,
            p.diagnosis_date,
            p.tumor_stage,
            p.treatment_type,
            s.sample_id,
            s.sample_uuid,
            st.type_name AS sample_type,
            s.collection_date,
            s.tissue_site,
            -- Pivot biomarker TPM values into columns
            MAX(CASE WHEN b.ensembl_id = 'ENSG00000105374' THEN m.tpm ELSE NULL END) AS tp53_tpm,
            MAX(CASE WHEN b.ensembl_id = 'ENSG00000146648' THEN m.tpm ELSE NULL END) AS egfr_tpm,
            MAX(CASE WHEN b.ensembl_id = 'ENSG00000157764' THEN m.tpm ELSE NULL END) AS braf_tpm
        FROM
            Individuals i
        JOIN
            Samples s ON i.individual_id = s.individual_id
        JOIN
            SampleTypes st ON s.sample_type_id = st.sample_type_id
        LEFT JOIN
            Patients p ON i.individual_id = p.individual_id
        LEFT JOIN
            Measurements m ON s.sample_id = m.sample_id
        LEFT JOIN
            Biomarkers b ON m.biomarker_id = b.biomarker_id
        GROUP BY
            i.individual_id, i.individual_uuid, i.gender, i.age, i.vital_status, i.survival_months, i.ethnicity,
            p.cancer_type, p.diagnosis_date, p.tumor_stage, p.treatment_type,
            s.sample_id, s.sample_uuid, st.type_name, s.collection_date, s.tissue_site
        ORDER BY
            i.individual_id, s.sample_id;
        """
        df_full_db = pd.read_sql_query(query, conn)
        df_full_db.to_csv(output_filepath, index=False, encoding='utf-8')
        
        print(f"Export of entire database completed successfully to '{output_filepath}'")
        return True
    except Error as e:
        print(f"Error during export of entire database: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during export of entire database: {e}")
        return False

# --- Main Function ---

def main():
    conn, cursor = connect_db()
    if not conn or not cursor:
        return

    # To process all data, set this to None.
    # To process a limited number of cases for testing, set this to an integer (e.g., 100).
    MAX_CASES_FOR_DEBUG = None 

    try:
        sample_types_to_insert = [
            ("Primary Tumor", "Tissue sample collected from the primary tumor site."),
            ("Normal Adjacent Tissue", "Normal tissue sample collected adjacent to the primary tumor."),
            ("Blood", "Blood sample."),
            ("Metastatic Tumor", "Tissue sample collected from a metastatic site."),
            ("Healthy Control", "Sample from an individual without a cancer diagnosis.")
        ]
        print("\n--- Inserting standard sample types ---")
        for st_name, st_desc in sample_types_to_insert:
            existing_id = fetch_existing_id(cursor, "sampletypes", "type_name", st_name)
            if not existing_id:
                insert_data(cursor, "sampletypes", ["type_name", "description"], [st_name, st_desc])
                print(f"Sample type '{st_name}' inserted.")
            else:
                print(f"Sample type '{st_name}' already exists. Skipped.")
        conn.commit()

        print("\nStarting clinical data loading...")
        if extract_and_load_clinical_data(cursor, limit=MAX_CASES_FOR_DEBUG):
            conn.commit()
            print("Clinical data loaded successfully.")
        else:
            conn.rollback()
            print("Clinical data loading failed. Rollback.")
            return

        print("\nStarting sample data loading...")
        if extract_and_load_biospecimen_data(cursor, limit=MAX_CASES_FOR_DEBUG):
            conn.commit()
            print("Sample data loaded successfully.")
        else:
            conn.rollback()
            print("Sample data loading failed. Rollback.")
            return

        print("\nStarting molecular data loading (simulated for now)...")
        if extract_and_load_molecular_data(cursor, limit=MAX_CASES_FOR_DEBUG):
            conn.commit()
            print("Molecular data loaded successfully.")
        else:
            conn.rollback()
            print("Molecular data loading failed.")
            return

        export_full_db_to_single_csv(conn, os.path.join(os.path.dirname(base_dir), "data", "full_cancer_data.csv"))
        print("CSV data export completed.")

    except Exception as e:
        print(f"\nAn general error occurred during main execution: {e}")
        conn.rollback()
    finally:
        close_db(conn, cursor)

if __name__ == "__main__":
    main()