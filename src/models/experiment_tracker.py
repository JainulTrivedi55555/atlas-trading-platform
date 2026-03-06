import mlflow
import mlflow.sklearn
import mlflow.pytorch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

EXPERIMENT_NAME = 'ATLAS-Price-Prediction'

def setup_mlflow():
    """Initialize MLflow experiment."""
    mlflow.set_tracking_uri('mlruns')
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f'MLflow experiment: {EXPERIMENT_NAME}')
    print('View UI: mlflow ui  then open http://localhost:5000')

def log_model_run(model_name, params, metrics, artifacts=None):
    """Log a complete model run to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Log all hyperparameters
        mlflow.log_params(params)
        
        # Log all metrics
        mlflow.log_metrics(metrics)
        
        # Log any files (plots, reports) 
        if artifacts:
            for artifact_path in artifacts:
                if Path(artifact_path).exists():
                    mlflow.log_artifact(artifact_path)
                    
        print(f'Logged run: {model_name}')
        print(f'Metrics: {metrics}')

if __name__ == '__main__':
    setup_mlflow()
    
    # Test log
    log_model_run(
        model_name='test_run',
        params={'learning_rate': 0.01, 'n_estimators': 100},
        metrics={'auc': 0.52, 'accuracy': 0.54}
    )
    
    print('MLflow setup complete!')