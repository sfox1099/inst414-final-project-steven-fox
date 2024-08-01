from etl.extract import extract
from etl.transform import transform
from etl.load import load
from analysis.model import prepare_data, train_model
from analysis.evaluate import evaluate_model
from vis.visualizations import plot_feature_distributions, plot_correlation_matrix, plot_model_performance

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
    # Step 1: Extract data
    file_path = 'data/extracted/sample_data.csv'
    raw_data = extract(file_path)
    
    # Step 2: Transform data
    processed_data = transform(raw_data)
    
    # Step 3: Load data
    processed_file_path = 'data/processed/processed_data.csv'
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

if __name__ == "__main__":
    main()


