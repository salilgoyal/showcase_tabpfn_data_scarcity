"""
Within-county experiment: Repeated k-fold cross-validation within a single county.

This experiment type evaluates model performance using repeated stratified k-fold
cross-validation on data from a single county.

The experiment:
- Loads a single county's data
- Performs repeated k-fold CV
- Reports per-fold metrics
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterator
from sklearn.model_selection import RepeatedKFold

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import BaseExperimentRunner, ExperimentMetadata
from src.data import CountyDataLoader

logger = logging.getLogger(__name__)


class WithinCountyExperiment(BaseExperimentRunner):
    """
    Experiment that performs repeated k-fold CV within a single county.

    Evaluates model performance using stratified cross-validation to understand
    how well models perform on data from the same county.
    """

    def __init__(self, config: Dict):
        """
        Initialize within-county experiment.

        Args:
            config: Configuration dictionary with:
                - data: Dataset paths and settings
                - models: Model configurations
                - preprocessing: Preprocessing settings
        """
        super().__init__(config)
        self.metadata = ExperimentMetadata(config)

        # Initialize data loader
        preprocessing_config = config.get('preprocessing')
        self.data_loader = CountyDataLoader(
            county_csvs_dir=config['data']['county_csvs_dir'],
            target_column=config['data']['target_column'],
            preprocessing_config=preprocessing_config
        )

    def create_repeated_kfold_splits(
        self,
        n_samples: int,
        k_folds: int,
        n_repeats: int,
        random_state: int = 42
    ) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
        """
        Generate repeated k-fold splits.

        Args:
            n_samples: Number of samples
            k_folds: Number of folds
            n_repeats: Number of repetitions
            random_state: Random state for reproducibility

        Yields:
            Tuples of (repetition, fold, train_indices, test_indices)
        """
        splitter = RepeatedKFold(
            n_splits=k_folds,
            n_repeats=n_repeats,
            random_state=random_state
        )

        rep = 0
        fold = 0
        for train_idx, test_idx in splitter.split(range(n_samples)):
            yield rep, fold, train_idx, test_idx

            fold += 1
            if fold >= k_folds:
                fold = 0
                rep += 1

    def run_single_fold(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        fips: int,
        bin_name: str,
        repetition: int,
        fold: int
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Run one fold for one model.

        Args:
            model_name: Model to use
            X: Full feature matrix
            y: Full target vector
            train_idx: Training indices
            test_idx: Test indices
            fips: County FIPS code
            bin_name: Size bin name
            repetition: Repetition number
            fold: Fold number

        Returns:
            Tuple of (result_dict, calibration_data, prediction_data)
        """
        try:
            # Split data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            logger.debug(
                f"  Rep {repetition}, Fold {fold}: "
                f"train={len(X_train)}, test={len(X_test)}"
            )

            # Check if predictions should be saved
            save_predictions = self.config.get('predictions', {}).get('save_predictions', False)

            # Train and predict
            result, cal_data, pred_data = self.train_and_predict(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                save_predictions=save_predictions
            )

            # Add experiment-specific metadata
            result.update({
                'fips': fips,
                'bin_name': bin_name,
                'repetition': repetition,
                'fold': fold,
                'status': 'success'
            })
            result = self.metadata.add_to_result(result)

            if cal_data is not None:
                cal_data['fips'] = fips
                cal_data['repetition'] = repetition
                cal_data['fold'] = fold

            if pred_data is not None:
                pred_data['fips'] = fips
                pred_data['repetition'] = repetition
                pred_data['fold'] = fold

            return result, cal_data, pred_data

        except Exception as e:
            logger.error(
                f"Failed for rep {repetition}, fold {fold}, model {model_name}: {e}",
                exc_info=True
            )

            # Return failed result
            result = self.metadata.add_to_result({
                'fips': fips,
                'bin_name': bin_name,
                'repetition': repetition,
                'fold': fold,
                'model': model_name,
                'train_size': 0,
                'test_size': 0,
                'n_features': X.shape[1] if X is not None else 0,
                'fit_time': 0,
                'pred_time': 0,
                'status': f'failed: {str(e)}',
                'r2': np.nan,
                'mae': np.nan,
                'rmse': np.nan,
                'mse': np.nan
            })

            return result, None, None

    def run_county(
        self,
        fips: int,
        bin_name: str,
        k_folds: int,
        n_repeats: int
    ) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
        """
        Run experiment for a single county.

        Args:
            fips: County FIPS code
            bin_name: Size bin name (for reference)
            k_folds: Number of CV folds
            n_repeats: Number of repetitions

        Returns:
            Tuple of (results_df, calibration_data, predictions_data)
        """
        logger.info("=" * 80)
        logger.info(f"WITHIN-COUNTY EXPERIMENT: County {fips} (bin: {bin_name})")
        logger.info("=" * 80)

        # Load and preprocess data
        logger.info(f"Loading county {fips}...")
        df = self.data_loader.load_county(fips, drop_missing_target=True)
        X, y = self.data_loader.preprocess_for_training(df)

        logger.info(f"County {fips}: {len(X)} samples, {X.shape[1]} features")

        # Get enabled models
        enabled_models = self.get_enabled_models()

        total_splits = k_folds * n_repeats
        logger.info(f"Models: {enabled_models}")
        logger.info(f"CV: {k_folds} folds × {n_repeats} repetitions = {total_splits} splits")
        logger.info(f"Total experiments: {total_splits * len(enabled_models)}")
        logger.info("=" * 80)

        # Results storage
        all_results = []
        all_calibration_data = []
        all_predictions_data = []

        experiment_num = 0
        import time
        start_time = time.time()

        # Run repeated k-fold CV
        for rep, fold, train_idx, test_idx in self.create_repeated_kfold_splits(
            n_samples=len(X),
            k_folds=k_folds,
            n_repeats=n_repeats,
            random_state=self.config['experiment']['random_seed']
        ):
            split_num = rep * k_folds + fold + 1
            logger.info(f"\n--- Split {split_num}/{total_splits} (rep={rep}, fold={fold}) ---")

            for model_name in enabled_models:
                experiment_num += 1

                result, cal_data, pred_data = self.run_single_fold(
                    model_name=model_name,
                    X=X,
                    y=y,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    fips=fips,
                    bin_name=bin_name,
                    repetition=rep,
                    fold=fold
                )

                all_results.append(result)

                if cal_data is not None:
                    all_calibration_data.append(cal_data)

                if pred_data is not None:
                    all_predictions_data.append(pred_data)

        # Create results DataFrame
        df_results = pd.DataFrame(all_results)

        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time / 60:.2f} minutes")
        logger.info(f"Total experiments: {len(all_results)}")
        logger.info(f"Successful: {sum(r['status'] == 'success' for r in all_results)}")

        return df_results, all_calibration_data or None, all_predictions_data or None

    def run_experiment(self) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
        """
        Run the experiment.

        Note: For within-county experiments, this is typically called via
        run_county() with specific parameters. This method exists for
        compatibility with BaseExperimentRunner interface.

        Returns:
            Tuple of (results_df, calibration_data, predictions_data)
        """
        raise NotImplementedError(
            "Within-county experiments should be run via run_county(fips, bin_name, k_folds, n_repeats). "
            "Use run_experiment.py with --experiment_type within_county."
        )
