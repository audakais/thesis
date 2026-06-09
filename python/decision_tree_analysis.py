import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import seaborn as sns

def analyze_biomarkers_with_dt(dataset_filename=None):
    """
    Runs analysis and saves all metrics to a structured
    multi-sheet Excel file.

    dataset_filename: name of the CSV in BASE_DIR. If None, the default is used.
    """

    print("=== BIOMARKER ANALYSIS (DECISION TREE - EXCEL EXPORT) ===")

    # --- CONFIGURATION ---
    # Default used when dataset_filename is not provided
    DATASET_FILENAME = dataset_filename if dataset_filename else "dataset_TCGA_THCA.csv"

    # Path to the data folder
    BASE_DIR = os.path.join(base_dir, "ml_dataset_project_batches")

    # Path to save PLOTS and SPREADSHEETS
    OUTPUT_DIR = os.path.join(os.path.dirname(base_dir), "outputs")
    
    dataset_path = os.path.join(BASE_DIR, DATASET_FILENAME)
    
    # --- Setup Output Files ---
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Create /outputs folder
    
    # Plot file name
    plot_filename = f"feature_importance_{DATASET_FILENAME.replace('.csv', '')}.png"
    plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    
    # SPREADSHEET file name
    excel_filename = f"analysis_results_{DATASET_FILENAME.replace('.csv', '')}.xlsx"
    excel_path = os.path.join(OUTPUT_DIR, excel_filename)
    
    # ----------------------
    
    print(f"Analyzing dataset: {DATASET_FILENAME}")
    print(f"Loading dataset: {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"ERROR: File not found. \nCheck that '{DATASET_FILENAME}' exists in:\n{BASE_DIR}")
        return
            
    print(f"Dataset loaded. Shape: {df.shape}")

    # --- 1. Data Preparation ---
    y = df['target']
    meta_columns = ['sample_id', 'sample_uuid', 'submitter_id', 'cancer_project', 'sample_type', 'target']
    X = df.drop(columns=meta_columns, errors='ignore')
    
    if X.isnull().values.any():
        print("Found NaN values. Filling with 0...")
        X = X.fillna(0)
    
    print("\nClass Distribution (Imbalance):")
    print(y.value_counts())
    
    # Safety check for single-class datasets
    if y.nunique() < 2:
        print(f"\nWARNING: This dataset ('{DATASET_FILENAME}') contains only one class.")
        print("Cannot perform classification. Skipping this project.")
        print("\nAnalysis complete (skipped).")
        return 
    
    # --- 2. Data Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # --- 3. Model Training ---
    print("\nTraining Decision Tree...")
    start_time = time.time()
    
    dt_classifier = DecisionTreeClassifier(
        max_depth=7,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced'
    )
    
    dt_classifier.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    
    # --- 4. Model Evaluation ---
    y_pred = dt_classifier.predict(X_test)
    
    # Get all metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # IMPORTANT: Get classification report as a dictionary
    report_dict = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Tumor (1)'], output_dict=True)
    
    print(f"\nModel Performance on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # --- 5. Biomarker Identification ---
    print("\nIdentifying most important biomarkers...")
    
    # Build ENSG -> symbol map for label display
    _named_path = os.path.join(OUTPUT_DIR, "Biomarkers_Final_Named.csv")
    _symbol_map = {}
    if os.path.exists(_named_path):
        _nm = pd.read_csv(_named_path)[['Gene', 'symbol']].dropna()
        _symbol_map = dict(zip(_nm['Gene'], _nm['symbol']))
    # mygene-resolved fallbacks for genes not in Biomarkers_Final_Named.csv
    _symbol_map.update({
        "ENSG00000104951": "IL4I1",
        "ENSG00000234493": "RHOXF1P1",
    })

    # Get the FULL list of feature importances, map ENSG to symbol
    feature_importance_df = pd.DataFrame({
        'biomarker_name': [_symbol_map.get(g, g) for g in X.columns],
        'importance': dt_classifier.feature_importances_
    }).sort_values('importance', ascending=False)

    top_features = feature_importance_df[feature_importance_df['importance'] > 0]

    print(f"Model used {len(top_features)} out of {len(X.columns)} features.")

    # --- 6. Plot and Save Graph ---
    cohort_label = (DATASET_FILENAME
                    .replace("dataset_", "")
                    .replace("_vs_gtex.csv", " vs GTEx")
                    .replace(".csv", ""))
    print(f"Saving plot to {plot_path}...")
    plt.figure(figsize=(12, 8))
    top_plot = top_features.head(20)  # plot top 20

    sns.barplot(x='importance', y='biomarker_name', data=top_plot, palette='viridis')
    plt.xlabel('Feature Importance (Gini Importance)')
    plt.ylabel('Biomarker (Gene)')
    plt.title(f'Top {len(top_plot)} Biomarker Importance — {cohort_label}')
    plt.tight_layout()

    try:
        plt.savefig(plot_path, dpi=300)
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close()

    # --- 6b. Confusion Matrix PNG ---
    cm_tag = cohort_label.replace(' ', '_').replace('/', '_')
    cm_png = os.path.join(OUTPUT_DIR, f"confusion_matrix_DT_{cm_tag}.png")
    display_labels = ['Normal', 'Tumor'] if y.nunique() == 2 else [str(c) for c in sorted(y.unique())]
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=display_labels, yticklabels=display_labels)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title(f'Decision Tree — Confusion Matrix\n{cohort_label}', fontsize=11)
    plt.tight_layout()
    try:
        fig_cm.savefig(cm_png, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix PNG: {cm_png}")
    except Exception as e:
        print(f"Error saving confusion matrix PNG: {e}")
    plt.close(fig_cm)

    # --- 7. Save Results to Excel Spreadsheet ---
    print(f"Saving spreadsheet report to {excel_path}...")
    
    # Prepare DataFrames for each sheet
    
    # Sheet 1: Summary
    summary_data = {
        'Metric': ['Accuracy', 'F1-Score (Macro Avg)', 'F1-Score (Weighted Avg)', 'Total Features Used'],
        'Value': [
            report_dict['accuracy'],
            report_dict['macro avg']['f1-score'],
            report_dict['weighted avg']['f1-score'],
            len(top_features)
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Sheet 2: Classification Report
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Sheet 3: Feature Importance (the FULL 1500-gene list)
    # This is already feature_importance_df
    
    # Sheet 4: Confusion Matrix
    cm_df = pd.DataFrame(cm, 
                         index=['Actual Normal (0)', 'Actual Tumor (1)'], 
                         columns=['Predicted Normal (0)', 'Predicted Tumor (1)'])
    
    # Write all DataFrames to a single Excel file
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            report_df.to_excel(writer, sheet_name='Classification_Report')
            feature_importance_df.to_excel(writer, sheet_name='Feature_Importance (All)', index=False)
            cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
            
        print("Spreadsheet report saved successfully.")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

    print("\n--- Analysis complete ---")


if __name__ == "__main__":
    # Runs on all dataset_TCGA_*.csv files found in ml_dataset_project_batches.
    # Falls back to hardcoded default if the folder is empty.
    BASE_DIR = os.path.join(base_dir, "ml_dataset_project_batches")
    files = sorted(glob.glob(os.path.join(BASE_DIR, "dataset_TCGA_*.csv")))
    if files:
        for fp in files:
            print(f"\n{'='*60}\n{os.path.basename(fp)}\n{'='*60}")
            try:
                analyze_biomarkers_with_dt(os.path.basename(fp))
            except Exception as e:
                print(f"ERROR su {os.path.basename(fp)}: {e}")
                continue
    else:
        analyze_biomarkers_with_dt()
