from zenml.client import Client

from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    # print(Client().active_stack.experiment_tracker.local_path)
    train_pipeline(data_path="/home/mafaq/repos/Customer Satisfaction MLOPS Project/data/olist_customers_dataset.csv")
    
    # mlflow ui --backend-store-uri "/home/mafaq/.config/zenml/local_stores/411ff4a1-f1a3-4e72-8b4d-c830bb2b4de5/mlruns"