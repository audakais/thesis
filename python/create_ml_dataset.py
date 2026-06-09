import psycopg2
import pandas as pd
import time
import os
import sys

def create_ml_dataset():
    """
    Creates an ML dataset by performing Feature Selection (Top N Variance)
    AND the PIVOT operation directly inside PostgreSQL using 'crosstab'.
    
    This solves both the pandas Out-of-Memory error AND the 
    PostgreSQL '1600 column limit' error.
    
    REQUIRES: The 'tablefunc' extension must be enabled in PostgreSQL.
    (Run: 'CREATE EXTENSION IF NOT EXISTS tablefunc;' in psql)
    """
    
    print("--- RUNNING SCRIPT v5 (FEATURE SELECTION + DB PIVOT) ---")
    
    # --- Database Configuration ---
    DB_HOST = "localhost"
    DB_NAME = "cancerdata"
    DB_USER = "postgres"
    DB_PASSWORD = "password"
    
    # --- Configuration Constants ---
    MIN_SAMPLES_FOR_PROJECT = 100
    MEASUREMENT_TYPE_TO_USE = 'TPM' # Correct name found from DB
    
    # How many features to select? 
    # MUST be < 1600 (PostgreSQL crosstab limit). 1500 is a safe number.
    TOP_N_FEATURES_TO_SELECT = 1500
    
    # --- Output Configuration ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "ml_dataset_project_batches")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    
    print(f"--- LOGIC: Filter projects (>= {MIN_SAMPLES_FOR_PROJECT} Normal/Tumor) ---")
    print(f"---         Selecting Top {TOP_N_FEATURES_TO_SELECT} most variant features ---")
    print(f"---         Pivoting using 'crosstab' in DB ---")
    
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        conn.autocommit = True
        print("Database connection established.")
        
        # --- STEP 1: Find valid 'project_id's ---
        print(f"1. Finding 'project_id's with >= {MIN_SAMPLES_FOR_PROJECT} samples...")
        cancer_types_query = """
        SELECT 
            p.project_id
        FROM Patients p
        JOIN Samples s ON p.individual_id = s.individual_id
        JOIN SampleTypes st ON s.sample_type_id = st.type_id
        WHERE st.type_name IN ('Solid Tissue Normal', 'Primary Tumor')
        GROUP BY p.project_id
        HAVING COUNT(CASE WHEN st.type_name = 'Solid Tissue Normal' THEN 1 END) >= %s
           AND COUNT(CASE WHEN st.type_name = 'Primary Tumor' THEN 1 END) >= %s
        """
        viable_cancers_df = pd.read_sql_query(cancer_types_query, conn, params=(MIN_SAMPLES_FOR_PROJECT, MIN_SAMPLES_FOR_PROJECT))
        cancer_types_list = viable_cancers_df['project_id'].tolist()
        
        print(f"Found {len(cancer_types_list)} valid projects.")
        print(cancer_types_list)
        
        if not cancer_types_list:
            print("ERROR: No projects found.")
            return None
        
        # --- STEP 2: Process each project ---
        print("\n2. Processing each project (in batches)...")
        
        # Query to get sample metadata
        samples_query = """
            (SELECT s.sample_id, s.sample_uuid, s.submitter_id, p.project_id as cancer_project, st.type_name as sample_type, 0 as target
             FROM Samples s JOIN Patients p ON s.individual_id = p.individual_id JOIN SampleTypes st ON s.sample_type_id = st.type_id
             WHERE p.project_id = %s AND st.type_name = 'Solid Tissue Normal'
             AND EXISTS (SELECT 1 FROM Measurements m WHERE m.sample_id = s.sample_id AND m.measurement_type = %s))
            UNION ALL
            (SELECT s.sample_id, s.sample_uuid, s.submitter_id, p.project_id as cancer_project, st.type_name as sample_type, 1 as target
             FROM Samples s JOIN Patients p ON s.individual_id = p.individual_id JOIN SampleTypes st ON s.sample_type_id = st.type_id
             WHERE p.project_id = %s AND st.type_name = 'Primary Tumor'
             AND EXISTS (SELECT 1 FROM Measurements m WHERE m.sample_id = s.sample_id AND m.measurement_type = %s))
        """

        total_samples_processed = 0

        for i, project_id in enumerate(cancer_types_list):
            print(f"\n--- Processing {i+1}/{len(cancer_types_list)}: {project_id} ---")
            
            # 2a. Get sample metadata
            print(f"  Fetching sample metadata...")
            meta_params = (project_id, MEASUREMENT_TYPE_TO_USE, project_id, MEASUREMENT_TYPE_TO_USE)
            meta_dataset = pd.read_sql_query(samples_query, conn, params=meta_params)
            
            if meta_dataset.empty:
                print(f"  WARNING: {project_id} has no samples with data for type '{MEASUREMENT_TYPE_TO_USE}'. Skipping.")
                continue
                
            print(f"  Found {len(meta_dataset)} samples.")
            sample_ids_tuple = tuple(meta_dataset['sample_id'].tolist())

            # 2b. === NEW FEATURE SELECTION ===
            # Find the TOP N most variable biomarkers for THIS project's samples
            print(f"  Selecting Top {TOP_N_FEATURES_TO_SELECT} most variant biomarkers...")
            
            variance_query = f"""
            SELECT
                biomarker_id,
                VAR_SAMP(LOG(measurement_value + 1) / LOG(2)) as variance
            FROM Measurements
            WHERE sample_id IN %s
              AND measurement_type = %s
            GROUP BY biomarker_id
            HAVING VAR_SAMP(LOG(measurement_value + 1) / LOG(2)) IS NOT NULL
               AND VAR_SAMP(LOG(measurement_value + 1) / LOG(2)) > 0
            ORDER BY variance DESC
            LIMIT %s;
            """
            
            try:
                top_biomarker_ids_df = pd.read_sql_query(variance_query, conn, params=(sample_ids_tuple, MEASUREMENT_TYPE_TO_USE, TOP_N_FEATURES_TO_SELECT))
                top_biomarker_ids_tuple = tuple(top_biomarker_ids_df['biomarker_id'].tolist())
                
                if not top_biomarker_ids_tuple:
                    print("  WARNING: Could not find any variable biomarkers for this project. Skipping.")
                    continue
                
                print(f"  Found {len(top_biomarker_ids_tuple)} top biomarkers.")
                
            except Exception as e:
                print(f"  ERROR during variance calculation: {e}. Skipping.")
                continue

            # 2c. Get the NAMES of these top N biomarkers (our columns)
            print("  Fetching names for top biomarkers...")
            biomarker_names_query = """
            SELECT biomarker_name
            FROM Biomarkers
            WHERE biomarker_id IN %s
            ORDER BY biomarker_name;
            """
            biomarker_names_df = pd.read_sql_query(biomarker_names_query, conn, params=(top_biomarker_ids_tuple,))
            biomarker_list = biomarker_names_df['biomarker_name'].tolist()
            
            if not biomarker_list:
                print(f"  WARNING: Could not find names for biomarkers. Skipping.")
                continue

            # 2d. Build the dynamic crosstab query (now much smaller)
            print("  Building dynamic database pivot query...")
            
            # Format column names: "GeneA" DOUBLE PRECISION, "GeneB" DOUBLE PRECISION, ...
            formatted_cols = ', '.join([f'"{name}" DOUBLE PRECISION' for name in biomarker_list])
            
            # Source SQL - now also filters for TOP biomarker IDs
            source_sql = f"""
            SELECT m.sample_id, b.biomarker_name, m.measurement_value
            FROM Measurements m
            JOIN Biomarkers b ON m.biomarker_id = b.biomarker_id
            WHERE m.sample_id IN {sample_ids_tuple}
              AND m.biomarker_id IN {top_biomarker_ids_tuple}
              AND m.measurement_type = '{MEASUREMENT_TYPE_TO_USE}'
            ORDER BY 1, 2;
            """
            source_sql_escaped = source_sql.replace("'", "''") # Escape quotes for SQL string

            # Category SQL - now also filters for TOP biomarker IDs
            category_sql = f"""
            SELECT b.biomarker_name
            FROM Biomarkers b
            WHERE b.biomarker_id IN {top_biomarker_ids_tuple}
            ORDER BY 1;
            """
            
            # Build the final query (now with < 1600 columns)
            pivot_query = f"""
            SELECT *
            FROM crosstab(
                '{source_sql_escaped}',
                $$ {category_sql} $$
            ) AS (sample_id INT, {formatted_cols});
            """

            # 2e. Execute the pivot query
            print(f"  Executing filtered pivot in database (this may take a minute)...")
            start_pivot_time = time.time()
            
            try:
                pivot_df = pd.read_sql_query(pivot_query, conn)
            except Exception as e:
                print(f"  ERROR during database pivot for {project_id}: {e}")
                print("  Skipping this project.")
                continue
                
            end_pivot_time = time.time()
            print(f"  Pivot complete in {end_pivot_time - start_pivot_time:.2f} seconds.")

            # 2f. Merge metadata and pivoted data
            print("  Merging metadata and pivoted data...")
            final_project_df = pd.merge(
                meta_dataset, 
                pivot_df, 
                on='sample_id', 
                how='inner'
            )
            
            # 2g. SAVE THE CSV FILE
            safe_project_id = project_id.replace('-', '_').replace('/', '_')
            output_filename = f"dataset_{safe_project_id}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            final_project_df.to_csv(output_path, index=False)
            
            total_samples_processed += len(final_project_df)
            print(f"  SUCCESS: Saved {project_id} to {output_path}")
            print(f"  Dimensions for this batch: {final_project_df.shape}")

        
        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"All files saved to: {output_dir}")
        print(f"Total projects processed: {len(cancer_types_list)}")
        print(f"Total samples extracted: {total_samples_processed}")
        
    except (Exception, psycopg2.Error) as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    create_ml_dataset()