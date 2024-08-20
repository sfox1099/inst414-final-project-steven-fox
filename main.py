from extract import extract
from transform import transform
from load import load
from model import prepare_data, train_model
from evaluate import evaluate_model
from visualization import plot_feature_distributions, plot_correlation_matrix, plot_model_performance
import os


def main():
    """
    Main function to execute the data pipeline workflow.
    
    Steps:
    1. Extract data
    2. Transform data
    3. Load data
    4. Build and train model
    5. Evaluate model
    6. Create visualizations

    Returns:
        None
    """
    try:
        # Step 1: Extract data
        file_path = "C:\\Users\\sfox1\\sfox_final_project\\inst414-final-project-steven-fox\\Data\\predictive_maintenance.csv"
        raw_data = extract(file_path)
        
        # Step 2: Transform data
        processed_data = transform(raw_data)
        
        # Ensure the directory exists
        processed_dir = 'data/processed'
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        # Step 3: Load data
        processed_file_path = os.path.join(processed_dir, 'processed_data.csv')
        load(processed_data, processed_file_path)
        
        # Step 4: Build and train model
        X, y = prepare_data(processed_data)
        model, X_test, y_test = train_model(X, y)
        
        # Step 5: Evaluate model
        accuracy, precision, recall, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
        
        # Step 6: Create visualizations
        plot_feature_distributions(processed_data)
        plot_correlation_matrix(processed_data)
        plot_model_performance(conf_matrix)
        
        print("Workflow executed successfully")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
