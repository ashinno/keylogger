"""Model Manager for ML anomaly detection system - handles training, persistence, and lifecycle management."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import pickle
import json
import joblib
from pathlib import Path
import threading
import time
import hashlib

# ML imports
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ModelManager:
    """Centralized model management for ML anomaly detection system."""
    
    def __init__(self, config):
        self.config = config
        
        # Configuration
        self.models_dir = Path(config.get('ml.models_dir', 'models'))
        self.models_dir.mkdir(exist_ok=True)
        
        self.auto_retrain = config.get('ml.model_manager.auto_retrain', True)
        self.retrain_interval = config.get('ml.model_manager.retrain_interval', 3600)  # 1 hour
        self.min_samples_retrain = config.get('ml.model_manager.min_samples_retrain', 100)
        self.model_validation = config.get('ml.model_manager.validation', True)
        
        # Model registry
        self.models = {}
        self.model_metadata = {}
        self.model_performance = {}
        self.model_versions = defaultdict(list)
        
        # Training data management
        self.training_data = defaultdict(list)
        self.validation_data = defaultdict(list)
        self.data_locks = defaultdict(threading.Lock)
        
        # Model lifecycle tracking
        self.model_status = {}  # 'training', 'ready', 'updating', 'error'
        self.last_training = {}
        self.training_history = defaultdict(list)
        
        # Auto-retraining
        self.retrain_thread = None
        self.retrain_active = False
        
        # Performance monitoring
        self.performance_metrics = defaultdict(dict)
        self.drift_detection = defaultdict(dict)
        
        # Statistics
        self.stats = {
            'models_registered': 0,
            'training_sessions': 0,
            'successful_trainings': 0,
            'failed_trainings': 0,
            'auto_retrains': 0,
            'model_deployments': 0
        }
        
        # Load existing models
        self._load_existing_models()
        
        # Start auto-retraining if enabled
        if self.auto_retrain:
            self._start_auto_retrain()
        
        logger.info("ModelManager initialized")
    
    def register_model(self, model_name: str, model: BaseEstimator, 
                      model_type: str = 'classifier', 
                      metadata: Dict[str, Any] = None) -> bool:
        """Register a new model with the manager."""
        try:
            # Validate model
            if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
                raise ValueError("Model must have fit() and predict() methods")
            
            # Store model
            self.models[model_name] = model
            
            # Store metadata
            self.model_metadata[model_name] = {
                'type': model_type,
                'registered_at': datetime.now().isoformat(),
                'class_name': model.__class__.__name__,
                'parameters': model.get_params() if hasattr(model, 'get_params') else {},
                'custom_metadata': metadata or {}
            }
            
            # Initialize status
            self.model_status[model_name] = 'registered'
            self.last_training[model_name] = None
            
            # Initialize performance tracking
            self.performance_metrics[model_name] = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'training_time': [],
                'prediction_time': []
            }
            
            self.stats['models_registered'] += 1
            
            logger.info(f"Model '{model_name}' registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model '{model_name}': {e}")
            return False
    
    def add_training_data(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                         validation: bool = False) -> bool:
        """Add training data for a model."""
        try:
            if model_name not in self.models:
                logger.warning(f"Model '{model_name}' not registered")
                return False
            
            with self.data_locks[model_name]:
                if validation:
                    self.validation_data[model_name].append((X.copy(), y.copy()))
                else:
                    self.training_data[model_name].append((X.copy(), y.copy()))
            
            logger.debug(f"Added {'validation' if validation else 'training'} data for '{model_name}': "
                        f"{X.shape[0]} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error adding training data for '{model_name}': {e}")
            return False
    
    def train_model(self, model_name: str, hyperparameter_tuning: bool = False, 
                   cross_validation: bool = True) -> Dict[str, Any]:
        """Train a registered model."""
        if model_name not in self.models:
            return {'success': False, 'error': f"Model '{model_name}' not registered"}
        
        try:
            self.model_status[model_name] = 'training'
            start_time = time.time()
            
            # Prepare training data
            training_result = self._prepare_training_data(model_name)
            if not training_result['success']:
                self.model_status[model_name] = 'error'
                return training_result
            
            X_train, y_train, X_val, y_val = training_result['data']
            
            # Get model
            model = self.models[model_name]
            
            # Hyperparameter tuning
            if hyperparameter_tuning:
                model = self._tune_hyperparameters(model, X_train, y_train)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Validate model
            validation_results = {}
            if self.model_validation and X_val is not None:
                validation_results = self._validate_model(model, X_val, y_val)
            
            # Cross-validation
            cv_results = {}
            if cross_validation and len(X_train) > 10:
                cv_results = self._cross_validate_model(model, X_train, y_train)
            
            # Update model
            self.models[model_name] = model
            
            # Record training
            training_time = time.time() - start_time
            self._record_training(model_name, training_time, validation_results, cv_results)
            
            # Save model
            self._save_model(model_name)
            
            self.model_status[model_name] = 'ready'
            self.stats['successful_trainings'] += 1
            
            result = {
                'success': True,
                'training_time': training_time,
                'training_samples': len(X_train),
                'validation_results': validation_results,
                'cv_results': cv_results,
                'model_path': str(self.models_dir / f"{model_name}.pkl")
            }
            
            logger.info(f"Model '{model_name}' trained successfully in {training_time:.2f}s")
            return result
            
        except Exception as e:
            self.model_status[model_name] = 'error'
            self.stats['failed_trainings'] += 1
            logger.error(f"Error training model '{model_name}': {e}")
            return {'success': False, 'error': str(e)}
    
    def _prepare_training_data(self, model_name: str) -> Dict[str, Any]:
        """Prepare training and validation data for a model."""
        try:
            with self.data_locks[model_name]:
                training_data = self.training_data[model_name]
                validation_data = self.validation_data[model_name]
            
            if not training_data:
                return {'success': False, 'error': 'No training data available'}
            
            # Combine training data
            X_train_list = []
            y_train_list = []
            
            for X, y in training_data:
                X_train_list.append(X)
                y_train_list.append(y)
            
            X_train = np.vstack(X_train_list)
            y_train = np.hstack(y_train_list)
            
            # Combine validation data
            X_val = None
            y_val = None
            
            if validation_data:
                X_val_list = []
                y_val_list = []
                
                for X, y in validation_data:
                    X_val_list.append(X)
                    y_val_list.append(y)
                
                X_val = np.vstack(X_val_list)
                y_val = np.hstack(y_val_list)
            
            # Check minimum samples
            if len(X_train) < self.min_samples_retrain:
                return {
                    'success': False, 
                    'error': f'Insufficient training data: {len(X_train)} < {self.min_samples_retrain}'
                }
            
            return {
                'success': True,
                'data': (X_train, y_train, X_val, y_val)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Error preparing data: {str(e)}'}
    
    def _tune_hyperparameters(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """Perform hyperparameter tuning."""
        try:
            # Define parameter grids for common models
            param_grids = {
                'RandomForestClassifier': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'SVC': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.1, 1],
                    'kernel': ['rbf', 'linear']
                },
                'GradientBoostingClassifier': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
            
            model_class = model.__class__.__name__
            
            if model_class in param_grids:
                param_grid = param_grids[model_class]
                
                # Perform grid search
                grid_search = GridSearchCV(
                    model, param_grid, 
                    cv=3, scoring='accuracy', 
                    n_jobs=-1, verbose=0
                )
                
                grid_search.fit(X, y)
                
                logger.info(f"Hyperparameter tuning completed for {model_class}. "
                           f"Best score: {grid_search.best_score_:.3f}")
                
                return grid_search.best_estimator_
            
            else:
                logger.info(f"No hyperparameter grid defined for {model_class}")
                return model
            
        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed: {e}")
            return model
    
    def _validate_model(self, model: BaseEstimator, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Validate model performance."""
        try:
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            results = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0),
                'samples': len(y_val)
            }
            
            # Classification report
            if len(np.unique(y_val)) > 1:
                results['classification_report'] = classification_report(
                    y_val, y_pred, output_dict=True, zero_division=0
                )
            
            return results
            
        except Exception as e:
            logger.warning(f"Model validation failed: {e}")
            return {'error': str(e)}
    
    def _cross_validate_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation."""
        try:
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
            results = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
            return results
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {'error': str(e)}
    
    def _record_training(self, model_name: str, training_time: float, 
                        validation_results: Dict[str, Any], cv_results: Dict[str, Any]):
        """Record training session details."""
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'training_time': training_time,
            'validation_results': validation_results,
            'cv_results': cv_results
        }
        
        self.training_history[model_name].append(training_record)
        self.last_training[model_name] = datetime.now()
        
        # Update performance metrics
        if 'accuracy' in validation_results:
            self.performance_metrics[model_name]['accuracy'].append(validation_results['accuracy'])
        if 'precision' in validation_results:
            self.performance_metrics[model_name]['precision'].append(validation_results['precision'])
        if 'recall' in validation_results:
            self.performance_metrics[model_name]['recall'].append(validation_results['recall'])
        if 'f1_score' in validation_results:
            self.performance_metrics[model_name]['f1_score'].append(validation_results['f1_score'])
        
        self.performance_metrics[model_name]['training_time'].append(training_time)
        
        self.stats['training_sessions'] += 1
    
    def predict(self, model_name: str, X: np.ndarray, 
               return_probabilities: bool = False) -> Dict[str, Any]:
        """Make predictions using a trained model."""
        try:
            if model_name not in self.models:
                return {'success': False, 'error': f"Model '{model_name}' not found"}
            
            if self.model_status.get(model_name) != 'ready':
                return {'success': False, 'error': f"Model '{model_name}' not ready"}
            
            model = self.models[model_name]
            start_time = time.time()
            
            # Make predictions
            predictions = model.predict(X)
            
            result = {
                'success': True,
                'predictions': predictions,
                'prediction_time': time.time() - start_time,
                'samples': len(X)
            }
            
            # Add probabilities if requested and available
            if return_probabilities and hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X)
                    result['probabilities'] = probabilities
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")
            
            # Record prediction time
            self.performance_metrics[model_name]['prediction_time'].append(result['prediction_time'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions with '{model_name}': {e}")
            return {'success': False, 'error': str(e)}
    
    def _save_model(self, model_name: str) -> bool:
        """Save a trained model to disk."""
        try:
            model = self.models[model_name]
            metadata = self.model_metadata[model_name]
            
            # Create model package
            model_package = {
                'model': model,
                'metadata': metadata,
                'training_history': self.training_history[model_name],
                'performance_metrics': self.performance_metrics[model_name],
                'saved_at': datetime.now().isoformat()
            }
            
            # Save with versioning
            version = len(self.model_versions[model_name]) + 1
            model_file = self.models_dir / f"{model_name}_v{version}.pkl"
            
            with open(model_file, 'wb') as f:
                pickle.dump(model_package, f)
            
            # Create symlink to latest version
            latest_file = self.models_dir / f"{model_name}.pkl"
            if latest_file.exists():
                latest_file.unlink()
            
            try:
                latest_file.symlink_to(model_file.name)
            except OSError:
                # Fallback for systems that don't support symlinks
                import shutil
                shutil.copy2(model_file, latest_file)
            
            # Update version tracking
            self.model_versions[model_name].append({
                'version': version,
                'file': str(model_file),
                'saved_at': datetime.now().isoformat()
            })
            
            self.stats['model_deployments'] += 1
            
            logger.info(f"Model '{model_name}' saved as version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model '{model_name}': {e}")
            return False
    
    def load_model(self, model_name: str, version: Optional[int] = None) -> bool:
        """Load a model from disk."""
        try:
            if version is None:
                # Load latest version
                model_file = self.models_dir / f"{model_name}.pkl"
            else:
                # Load specific version
                model_file = self.models_dir / f"{model_name}_v{version}.pkl"
            
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                return False
            
            with open(model_file, 'rb') as f:
                model_package = pickle.load(f)
            
            # Restore model and metadata
            self.models[model_name] = model_package['model']
            self.model_metadata[model_name] = model_package['metadata']
            self.training_history[model_name] = model_package.get('training_history', [])
            self.performance_metrics[model_name] = model_package.get('performance_metrics', {})
            
            self.model_status[model_name] = 'ready'
            
            logger.info(f"Model '{model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {e}")
            return False
    
    def _load_existing_models(self):
        """Load all existing models from disk."""
        try:
            model_files = list(self.models_dir.glob("*.pkl"))
            
            for model_file in model_files:
                # Skip versioned files, only load latest
                if '_v' in model_file.stem:
                    continue
                
                model_name = model_file.stem
                if self.load_model(model_name):
                    logger.info(f"Loaded existing model: {model_name}")
            
        except Exception as e:
            logger.warning(f"Error loading existing models: {e}")
    
    def _start_auto_retrain(self):
        """Start automatic retraining thread."""
        if self.retrain_thread and self.retrain_thread.is_alive():
            return
        
        self.retrain_active = True
        self.retrain_thread = threading.Thread(
            target=self._auto_retrain_loop,
            daemon=True
        )
        self.retrain_thread.start()
        
        logger.info("Auto-retraining thread started")
    
    def _auto_retrain_loop(self):
        """Automatic retraining loop."""
        while self.retrain_active:
            try:
                for model_name in list(self.models.keys()):
                    if self._should_retrain(model_name):
                        logger.info(f"Auto-retraining model: {model_name}")
                        
                        result = self.train_model(model_name)
                        if result['success']:
                            self.stats['auto_retrains'] += 1
                            logger.info(f"Auto-retrain successful for {model_name}")
                        else:
                            logger.warning(f"Auto-retrain failed for {model_name}: {result.get('error')}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in auto-retrain loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _should_retrain(self, model_name: str) -> bool:
        """Determine if a model should be retrained."""
        try:
            # Check if enough time has passed
            last_train = self.last_training.get(model_name)
            if last_train:
                time_since_train = (datetime.now() - last_train).total_seconds()
                if time_since_train < self.retrain_interval:
                    return False
            
            # Check if we have enough new training data
            with self.data_locks[model_name]:
                training_data = self.training_data[model_name]
            
            if not training_data:
                return False
            
            total_samples = sum(len(X) for X, y in training_data)
            if total_samples < self.min_samples_retrain:
                return False
            
            # Check for performance degradation
            if self._detect_performance_drift(model_name):
                logger.info(f"Performance drift detected for {model_name}")
                return True
            
            # Check for data drift
            if self._detect_data_drift(model_name):
                logger.info(f"Data drift detected for {model_name}")
                return True
            
            # Regular retraining interval
            return True
            
        except Exception as e:
            logger.warning(f"Error checking retrain condition for {model_name}: {e}")
            return False
    
    def _detect_performance_drift(self, model_name: str) -> bool:
        """Detect performance drift in model."""
        try:
            metrics = self.performance_metrics.get(model_name, {})
            accuracy_history = metrics.get('accuracy', [])
            
            if len(accuracy_history) < 5:
                return False
            
            # Compare recent performance to historical average
            recent_accuracy = np.mean(accuracy_history[-3:])
            historical_accuracy = np.mean(accuracy_history[:-3])
            
            # Significant drop in performance
            drift_threshold = 0.05  # 5% drop
            return (historical_accuracy - recent_accuracy) > drift_threshold
            
        except Exception as e:
            logger.warning(f"Error detecting performance drift for {model_name}: {e}")
            return False
    
    def _detect_data_drift(self, model_name: str) -> bool:
        """Detect data drift using statistical tests."""
        try:
            # This is a simplified implementation
            # In practice, you'd use more sophisticated drift detection methods
            
            with self.data_locks[model_name]:
                training_data = self.training_data[model_name]
            
            if len(training_data) < 2:
                return False
            
            # Compare recent data distribution to historical
            recent_X = training_data[-1][0]
            historical_X = np.vstack([X for X, y in training_data[:-1]])
            
            # Simple statistical test (Kolmogorov-Smirnov)
            from scipy.stats import ks_2samp
            
            drift_detected = False
            for feature_idx in range(min(recent_X.shape[1], 10)):  # Check first 10 features
                recent_feature = recent_X[:, feature_idx]
                historical_feature = historical_X[:, feature_idx]
                
                statistic, p_value = ks_2samp(recent_feature, historical_feature)
                
                if p_value < 0.05:  # Significant difference
                    drift_detected = True
                    break
            
            return drift_detected
            
        except Exception as e:
            logger.warning(f"Error detecting data drift for {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a model."""
        if model_name not in self.models:
            return {'error': f"Model '{model_name}' not found"}
        
        info = {
            'name': model_name,
            'status': self.model_status.get(model_name, 'unknown'),
            'metadata': self.model_metadata.get(model_name, {}),
            'last_training': self.last_training.get(model_name),
            'training_history': len(self.training_history.get(model_name, [])),
            'performance_metrics': self._get_performance_summary(model_name),
            'versions': len(self.model_versions.get(model_name, [])),
            'training_data_size': self._get_training_data_size(model_name)
        }
        
        return info
    
    def _get_performance_summary(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics summary for a model."""
        metrics = self.performance_metrics.get(model_name, {})
        
        summary = {}
        for metric_name, values in metrics.items():
            if values:
                summary[metric_name] = {
                    'latest': values[-1],
                    'average': np.mean(values),
                    'best': max(values) if metric_name != 'training_time' else min(values),
                    'trend': 'improving' if len(values) > 1 and values[-1] > values[-2] else 'stable'
                }
        
        return summary
    
    def _get_training_data_size(self, model_name: str) -> Dict[str, int]:
        """Get training data size information."""
        with self.data_locks[model_name]:
            training_data = self.training_data[model_name]
            validation_data = self.validation_data[model_name]
        
        training_samples = sum(len(X) for X, y in training_data)
        validation_samples = sum(len(X) for X, y in validation_data)
        
        return {
            'training_samples': training_samples,
            'validation_samples': validation_samples,
            'training_batches': len(training_data),
            'validation_batches': len(validation_data)
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models with basic information."""
        models_list = []
        
        for model_name in self.models.keys():
            model_info = {
                'name': model_name,
                'status': self.model_status.get(model_name, 'unknown'),
                'type': self.model_metadata.get(model_name, {}).get('type', 'unknown'),
                'last_training': self.last_training.get(model_name),
                'performance': self._get_latest_performance(model_name)
            }
            models_list.append(model_info)
        
        return models_list
    
    def _get_latest_performance(self, model_name: str) -> Dict[str, float]:
        """Get latest performance metrics for a model."""
        metrics = self.performance_metrics.get(model_name, {})
        
        latest = {}
        for metric_name, values in metrics.items():
            if values:
                latest[metric_name] = values[-1]
        
        return latest
    
    def delete_model(self, model_name: str, delete_files: bool = True) -> bool:
        """Delete a model and optionally its files."""
        try:
            # Remove from memory
            if model_name in self.models:
                del self.models[model_name]
            if model_name in self.model_metadata:
                del self.model_metadata[model_name]
            if model_name in self.model_status:
                del self.model_status[model_name]
            if model_name in self.last_training:
                del self.last_training[model_name]
            if model_name in self.training_history:
                del self.training_history[model_name]
            if model_name in self.performance_metrics:
                del self.performance_metrics[model_name]
            
            # Clear training data
            with self.data_locks[model_name]:
                if model_name in self.training_data:
                    del self.training_data[model_name]
                if model_name in self.validation_data:
                    del self.validation_data[model_name]
            
            # Delete files
            if delete_files:
                # Delete all versions
                for version_info in self.model_versions.get(model_name, []):
                    file_path = Path(version_info['file'])
                    if file_path.exists():
                        file_path.unlink()
                
                # Delete latest symlink
                latest_file = self.models_dir / f"{model_name}.pkl"
                if latest_file.exists():
                    latest_file.unlink()
            
            if model_name in self.model_versions:
                del self.model_versions[model_name]
            
            logger.info(f"Model '{model_name}' deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model '{model_name}': {e}")
            return False
    
    def export_model(self, model_name: str, export_path: str, 
                    format: str = 'pickle') -> bool:
        """Export a model to a specific format."""
        try:
            if model_name not in self.models:
                logger.error(f"Model '{model_name}' not found")
                return False
            
            model = self.models[model_name]
            export_path = Path(export_path)
            
            if format.lower() == 'pickle':
                with open(export_path, 'wb') as f:
                    pickle.dump(model, f)
            
            elif format.lower() == 'joblib':
                joblib.dump(model, export_path)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Model '{model_name}' exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting model '{model_name}': {e}")
            return False
    
    def clear_training_data(self, model_name: str, keep_latest: int = 0) -> bool:
        """Clear training data for a model, optionally keeping latest batches."""
        try:
            with self.data_locks[model_name]:
                if model_name in self.training_data:
                    if keep_latest > 0:
                        self.training_data[model_name] = self.training_data[model_name][-keep_latest:]
                    else:
                        self.training_data[model_name].clear()
                
                if model_name in self.validation_data:
                    if keep_latest > 0:
                        self.validation_data[model_name] = self.validation_data[model_name][-keep_latest:]
                    else:
                        self.validation_data[model_name].clear()
            
            logger.info(f"Training data cleared for '{model_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing training data for '{model_name}': {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model manager statistics."""
        stats = self.stats.copy()
        
        stats.update({
            'registered_models': len(self.models),
            'ready_models': sum(1 for status in self.model_status.values() if status == 'ready'),
            'training_models': sum(1 for status in self.model_status.values() if status == 'training'),
            'error_models': sum(1 for status in self.model_status.values() if status == 'error'),
            'auto_retrain_active': self.retrain_active,
            'total_training_data': sum(len(data) for data in self.training_data.values()),
            'total_validation_data': sum(len(data) for data in self.validation_data.values())
        })
        
        return stats
    
    def stop_auto_retrain(self):
        """Stop automatic retraining."""
        self.retrain_active = False
        
        if self.retrain_thread and self.retrain_thread.is_alive():
            self.retrain_thread.join(timeout=10)
        
        logger.info("Auto-retraining stopped")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_auto_retrain()