import pandas as pd
import psycopg2

# DB connection for gene name lookup
db_conf = {"host": "localhost", "database": "cancerdata", "user": "postgres", "password": "password"}

def create_final_report():
    df = pd.read_csv("risultati_validati.csv")

    conn = psycopg2.connect(**db_conf)
    cur = conn.cursor()

    gene_map = {}
    unique_ensg = df['Feature_Importances'].unique()

    # Resolve gene names from the database
    for ensg in unique_ensg:
        cur.execute("SELECT biomarker_name FROM Biomarkers WHERE biomarker_name = %s LIMIT 1", (ensg,))
        row = cur.fetchone()
        gene_map[ensg] = row[0] if row else ensg

    df['Gene_Symbol'] = df['Feature_Importances'].map(gene_map)

    cols_to_fix = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'CV_Stability']
    df[cols_to_fix] = df[cols_to_fix].round(3)

    final_table = df[['Cancer_Type', 'F1_Score', 'CV_Stability', 'Recall', 'Gene_Symbol']]

    final_table.to_csv("tabella_finale_tesi_completa.csv", index=False)

    print("Final table saved: tabella_finale_tesi_completa.csv")
    print("\nPreview:")
    print(final_table.head())

    cur.close()
    conn.close()

if __name__ == "__main__":
    create_final_report()
