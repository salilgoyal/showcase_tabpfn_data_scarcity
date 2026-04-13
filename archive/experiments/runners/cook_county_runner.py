"""
Experiment: Cook County subsampling experiment with multiple train sizes and seeds.
"""

import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.evelyn_preprocessing import load_and_prepare_data
from models import TabPFNModel, XGBoostModel
from evaluation import compute_metrics

logger = logging.getLogger(__name__)


class CookCountyRunner:
    """Runner for cook county subsampling experiment."""

    def __init__(self, config: Dict):
        """
        Initialize runner.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        
        # Extract data path (convert to absolute path)
        import os
        data_path = config['data']['cook_county_csv']
        if not os.path.isabs(data_path):
            # Convert relative path to absolute from project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.data_path = os.path.join(project_root, data_path)
        else:
            self.data_path = data_path
        self.target_column = config['data']['target_column']

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load cook county data and apply preprocessing.
        
        Returns:
            X_train_pool, y_train_pool, X_test, y_test
        """
        logger.info(f"Loading cook county data from: {self.data_path}")
        
        # Use the existing preprocessing function
        # Note: This function handles train/test split internally
        feature_config = self.preprocessing_config.get('features', {})
        step_config = self.preprocessing_config.get('steps', {})
        
        X_train_pool, y_train_pool, X_test, y_test, _ = load_and_prepare_data(
            data_path=self.data_path,
            feature_config=feature_config,
            step_config=step_config,
            cbg_column='block_group_id'  # Use default CBG column
        )
        
        logger.info(f"Features: {X_train_pool.shape[1]}")
        logger.info(f"Training pool: {len(X_train_pool)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train_pool, y_train_pool, X_test, y_test

    def sample_train_set(self, X_train_pool: pd.DataFrame, y_train_pool: pd.Series, 
                        train_size: int, seed: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Sample training set of specified size.
        
        Args:
            X_train_pool: Full training pool features
            y_train_pool: Full training pool targets
            train_size: Number of samples to draw
            seed: Random seed
            
        Returns:
            X_train, y_train
        """
        np.random.seed(seed)
        
        if train_size >= len(X_train_pool):
            logger.warning(f"Requested train_size {train_size} >= pool size {len(X_train_pool)}, using full pool")
            return X_train_pool, y_train_pool
        
        # Sample indices
        indices = np.random.choice(len(X_train_pool), size=train_size, replace=False)
        
        return X_train_pool.iloc[indices], y_train_pool.iloc[indices]

    def run_single_experiment(self, model_name: str, train_size: int, seed: int,
                             X_train_pool: pd.DataFrame, y_train_pool: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Run a single experiment with given model, train size, and seed.
        
        Args:
            model_name: Name of model to use ('tabpfn' or 'xgboost')
            train_size: Number of training samples
            seed: Random seed
            X_train_pool: Full training pool features
            y_train_pool: Full training pool targets  
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with experiment results
        """
        logger.info(f"Running {model_name} with train_size={train_size}, seed={seed}")
        
        try:
            # Sample training data
            X_train, y_train = self.sample_train_set(X_train_pool, y_train_pool, train_size, seed)
            
            # Initialize model
            if model_name == 'tabpfn':
                model = TabPFNModel(config=self.config.get('models', {}).get('tabpfn', {}))
            elif model_name == 'xgboost':
                model = XGBoostModel(config=self.config.get('models', {}).get('xgboost', {}))
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - start_time
            
            # Predict
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            # Compute metrics
            log_transformed = self.preprocessing_config.get('steps', {}).get('log_transform_target', False)
            metrics = compute_metrics(y_test, y_pred, log_transformed=log_transformed)
            
            # Create result record
            result = {
                'experiment_name': self.config['experiment']['name'],
                'experiment_description': self.config['experiment']['description'],
                'train_size': train_size,
                'test_size': len(y_test),
                'n_features': X_train.shape[1],
                'seed': seed,
                'model': model_name,
                'fit_time': fit_time,
                'pred_time': pred_time,
                'status': 'success',
                **metrics
            }
            
            logger.info(f"Success: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed: {str(e)}")
            
            # Return failed result
            result = {
                'experiment_name': self.config['experiment']['name'],
                'experiment_description': self.config['experiment']['description'],
                'train_size': train_size,
                'test_size': len(y_test),
                'n_features': 0,
                'seed': seed,
                'model': model_name,
                'fit_time': 0,
                'pred_time': 0,
                'status': f'failed: {str(e)}',
                'r2': np.nan,
                'mae': np.nan,
                'rmse': np.nan,
                'mse': np.nan
            }
            return result

    def run_experiment(self) -> pd.DataFrame:
        """
        Run the full cook county experiment.
        
        Returns:
            DataFrame with all results
        """
        logger.info("="*80)
        logger.info(f"COOK COUNTY EXPERIMENT: {self.config['experiment']['name']}")
        logger.info("="*80)
        
        # Load and prepare data
        X_train_pool, y_train_pool, X_test, y_test = self.load_and_prepare_data()
        
        # Get experiment parameters
        models = self.config.get('models', {})
        train_sizes = self.config.get('train_sizes', [100, 200, 500, 1000])
        seeds = self.config.get('seeds', list(range(10)))
        
        # Filter enabled models
        enabled_models = [name for name, config in models.items() 
                         if config.get('enabled', True)]
        
        if not enabled_models:
            raise ValueError("No models enabled in config")
        
        logger.info(f"Models: {enabled_models}")
        logger.info(f"Train sizes: {train_sizes}")
        logger.info(f"Seeds: {seeds}")
        
        total_experiments = len(enabled_models) * len(train_sizes) * len(seeds)
        logger.info(f"Total experiments: {total_experiments}")
        logger.info("="*80)
        
        # Run experiments
        results = []
        experiment_num = 0
        start_time = time.time()
        
        for model_name in enabled_models:
            for train_size in train_sizes:
                for seed in seeds:
                    experiment_num += 1
                    logger.info(f"\n--- Experiment {experiment_num}/{total_experiments} ---")
                    
                    result = self.run_single_experiment(
                        model_name, train_size, seed,
                        X_train_pool, y_train_pool, X_test, y_test
                    )
                    results.append(result)
        
        # Create results DataFrame
        df_results = pd.DataFrame(results)
        
        total_time = time.time() - start_time
        logger.info("="*80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("="*80)
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Total experiments: {len(results)}")
        logger.info(f"Successful: {sum(r['status'] == 'success' for r in results)}")
        
        return df_results


def main(config_path: str):
    """
    Main function to run cook county experiment.
    
    Args:
        config_path: Path to experiment config file
    """
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get experiment name for templating
    experiment_name = config.get('experiment', {}).get('name', 'default')
    experiment_type = 'cook_county'
    
    # Handle output directory templating
    output_dir = config['output']['results_dir']
    if '{experiment_type}' in output_dir or '{experiment_name}' in output_dir:
        output_dir = output_dir.format(
            experiment_type=experiment_type,
            experiment_name=experiment_name
        )
        config['output']['results_dir'] = output_dir
        logger.info(f"Output directory (after templating): {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(output_dir, 'cook_county_experiment.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Run experiment
    runner = CookCountyRunner(config)
    df_results = runner.run_experiment()
    
    # Save results
    output_file = os.path.join(output_dir, 'results.csv')
    df_results.to_csv(output_file, index=False)
    
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Logs saved to: {log_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Cook County Experiment')
    parser.add_argument('config', type=str, help='Path to experiment config file')
    
    args = parser.parse_args()
    main(args.config)