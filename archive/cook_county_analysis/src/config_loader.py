"""
Configuration loader for experiment configs.
Reads YAML config files and provides a standardized interface.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with experiment configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['experiment_name', 'preprocessing', 'models', 'experiment_params', 'paths', 'slurm']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    logger.info(f"Loaded config: {config['experiment_name']}")
    return config


def get_preprocessing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract preprocessing configuration from experiment config.
    
    Returns:
        Dictionary with preprocessing settings
    """
    return config['preprocessing']


def get_model_config(config: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific model.
    
    Args:
        config: Full experiment config
        model_name: Name of model ('tabpfn' or 'xgboost')
        
    Returns:
        Model config dict or None if model not enabled
    """
    for model in config['models']:
        if model['name'] == model_name and model.get('enabled', True):
            return model
    return None


def setup_preprocessing_from_config(config: Dict[str, Any]):
    """
    Set up preprocessing imports and settings based on config.
    This modifies the global namespace where it's called.
    
    Args:
        config: Experiment config dictionary
        
    Returns:
        Tuple of (load_and_prepare_data function, USE_LOG_TRANSFORMED bool)
    """
    prep_config = get_preprocessing_config(config)
    prep_type = prep_config['type']
    use_log_transform = prep_config.get('use_log_transform', False)
    
    if prep_type == 'original':
        from src.data_utils import load_and_prepare_data
        return load_and_prepare_data, False
        
    elif prep_type == 'evelyn':
        from src.evelyn_preprocessing import load_and_prepare_data_evelyn as _load_evelyn
        include_property_chars = prep_config.get('include_property_chars', False)
        
        def load_and_prepare_data(data_path, cbg_column='block_group_id'):
            return _load_evelyn(data_path, cbg_column, include_property_chars=include_property_chars)
        
        return load_and_prepare_data, True
        
    elif prep_type == 'evelyn_propertyonly':
        from src.evelyn_preprocessing_propertyonly import load_and_prepare_data_propertyonly_evelyn as load_and_prepare_data
        return load_and_prepare_data, True
        
    else:
        raise ValueError(f"Unknown preprocessing type: {prep_type}")


def get_slurm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get SLURM configuration from experiment config."""
    return config['slurm']


def get_experiment_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get experiment parameters (train_sizes, seeds, etc.) from config."""
    return config['experiment_params']


def get_paths_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get paths configuration from experiment config."""
    return config['paths']

