"""ML Model Interpretability Engine for Enhanced Keylogger System.

This module provides comprehensive interpretability features for machine learning models,
including SHAP values, LIME explanations, feature importance analysis, decision path
visualization, and confidence indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import json
import pickle
from pathlib import Path
from collections import defaultdict
import warnings

# ML interpretability imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    warnings.warn("Visualization libraries not available. Install matplotlib and seaborn")

# ML imports
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logger = logging.getLogger(__name__)


class ModelInterpretabilityEngine:
    """Comprehensive ML model interpretability engine."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the interpretability engine.
        
        Args:
            config: Configuration dictionary containing interpretability settings
        """
        self.config = config
        self.interpretability_config = config.get('ml', {}).get('interpretability', {})
        
        # Configuration parameters
        self.enabled = bool(self.interpretability_config.get('enabled', True))
        # Enable flags should reflect requested configuration regardless of library availability; fallbacks handled in setup methods
        shap_requested = self.interpretability_config.get('shap_enabled', True)
        lime_requested = self.interpretability_config.get('lime_enabled', True)
        self.shap_enabled = bool(shap_requested)
        self.lime_enabled = bool(lime_requested)
        self.feature_importance_enabled = bool(self.interpretability_config.get('feature_importance_enabled', True))
        self.decision_paths_enabled = bool(self.interpretability_config.get('decision_paths_enabled', True))
        
        # Storage for explanations
        self.explanations_cache = {}
        self.feature_names = []
        self.model_metadata = {}
        
        # SHAP explainers
        self.shap_explainers = {}
        
        # LIME explainers
        self.lime_explainers = {}
        
        # Statistics
        self.stats = {
            'explanations_generated': 0,
            'shap_explanations': 0,
            'lime_explanations': 0,
            'feature_importance_calculated': 0,
            'decision_paths_extracted': 0
        }
        
        logger.info(f"ModelInterpretabilityEngine initialized - SHAP: {self.shap_enabled}, LIME: {self.lime_enabled}")
    
    def setup_explainers(self, model: Any, X_train: np.ndarray, feature_names: List[str],
                        model_type: str = 'classifier') -> Dict[str, Any]:
        """Setup explainers for a given model.
        
        Args:
            model: Trained ML model
            X_train: Training data for background/reference
            feature_names: List of feature names
            model_type: Type of model ('classifier' or 'regressor')
            
        Returns:
            Dictionary containing setup status and explainer information
        """
        if not self.enabled:
            return {'status': 'disabled', 'explainers': []}
        
        self.feature_names = feature_names
        model_id = id(model)
        
        explainers_setup = []
        
        # Setup SHAP explainer
        if self.shap_enabled:
            try:
                shap_explainer = self._setup_shap_explainer(model, X_train, model_type)
                if shap_explainer:
                    self.shap_explainers[model_id] = shap_explainer
                    explainers_setup.append('shap')
                    logger.info(f"SHAP explainer setup for model {model_id}")
            except Exception as e:
                logger.warning(f"Failed to setup SHAP explainer: {e}")
        
        # Setup LIME explainer
        if self.lime_enabled:
            try:
                lime_explainer = self._setup_lime_explainer(X_train, feature_names, model_type)
                if lime_explainer:
                    self.lime_explainers[model_id] = lime_explainer
                    explainers_setup.append('lime')
                    logger.info(f"LIME explainer setup for model {model_id}")
            except Exception as e:
                logger.warning(f"Failed to setup LIME explainer: {e}")
        
        # Store model metadata
        self.model_metadata[model_id] = {
            'model_type': model_type,
            'feature_names': feature_names,
            'setup_time': datetime.now(),
            'explainers': explainers_setup
        }
        
        return {
            'status': 'success',
            'model_id': model_id,
            'explainers': explainers_setup,
            'feature_count': len(feature_names)
        }

    def explain(self, model: Any, X: np.ndarray, instance_index: Optional[int] = None) -> Dict[str, Any]:
        """Generate explanations with robust fallbacks when SHAP/LIME are unavailable.

        Returns a dictionary with keys: 'method', 'values', 'feature_importance', 'notes'.
        """
        result: Dict[str, Any] = {'method': None, 'values': None, 'feature_importance': None, 'notes': ''}
        if not self.enabled:
            result['notes'] = 'Interpretability disabled by configuration'
            return result

        try:
            model_id = id(model)
            # Prefer SHAP when available and configured
            if self.shap_enabled and SHAP_AVAILABLE and model_id in self.shap_explainers:
                explainer = self.shap_explainers[model_id]
                shap_values = explainer.shap_values(X)
                result['method'] = 'shap'
                result['values'] = shap_values
                self.stats['shap_explanations'] += 1
                return result
        except Exception as e:
            logger.debug(f"SHAP explanation failed: {e}")

        try:
            # Fallback to LIME when available
            if self.lime_enabled and LIME_AVAILABLE and model_id in self.lime_explainers and instance_index is not None:
                explainer = self.lime_explainers[model_id]
                exp = explainer.explain_instance(
                    X[instance_index],
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=min(10, X.shape[1])
                )
                result['method'] = 'lime'
                result['values'] = exp.as_list()
                self.stats['lime_explanations'] += 1
                return result
        except Exception as e:
            logger.debug(f"LIME explanation failed: {e}")

        # Hardened fallback: feature importance via model attribute or permutation
        try:
            importances = None
            if hasattr(model, 'feature_importances_'):
                importances = np.array(getattr(model, 'feature_importances_'))
                method = 'feature_importances_'
            else:
                # Permutation importance as a generic fallback
                pi = permutation_importance(model, X, model.predict(X), n_repeats=5, random_state=42)
                importances = pi.importances_mean
                method = 'permutation_importance'
            result['method'] = method
            result['feature_importance'] = importances.tolist() if importances is not None else None
            self.stats['feature_importance_calculated'] += 1
        except Exception as e:
            result['notes'] = f'No interpretability libraries available and feature importance failed: {e}'
            logger.debug(result['notes'])
        return result
    
    def _setup_shap_explainer(self, model: Any, X_train: np.ndarray, model_type: str) -> Optional[Any]:
        """Setup SHAP explainer based on model type."""
        try:
            if not SHAP_AVAILABLE:
                return None
            # Determine appropriate SHAP explainer
            if hasattr(model, 'predict_proba'):
                # Tree-based models
                if hasattr(model, 'estimators_') or 'RandomForest' in str(type(model)):
                    explainer = shap.TreeExplainer(model)
                else:
                    # Use KernelExplainer for other models
                    explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])  # Sample for efficiency
            else:
                # Regression or other models
                explainer = shap.KernelExplainer(model.predict, X_train[:100])
            
            return explainer
        except Exception as e:
            logger.error(f"Error setting up SHAP explainer: {e}")
            return None
    
    def _setup_lime_explainer(self, X_train: np.ndarray, feature_names: List[str], 
                             model_type: str) -> Optional[Any]:
        """Setup LIME explainer."""
        try:
            if not LIME_AVAILABLE:
                return None
            if model_type == 'classifier':
                explainer = lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    class_names=['Normal', 'Anomaly'],
                    mode='classification',
                    discretize_continuous=True
                )
            else:
                explainer = lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    mode='regression',
                    discretize_continuous=True
                )
            
            return explainer
        except Exception as e:
            logger.error(f"Error setting up LIME explainer: {e}")
            return None
    
    def explain_prediction(self, model: Any, instance: np.ndarray, 
                          explanation_types: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive explanation for a single prediction.
        
        Args:
            model: Trained ML model
            instance: Single instance to explain (1D array)
            explanation_types: List of explanation types to generate
            
        Returns:
            Dictionary containing various explanations
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        if explanation_types is None:
            explanation_types = ['shap', 'lime', 'feature_importance', 'confidence']
        
        model_id = id(model)
        instance = instance.reshape(1, -1) if instance.ndim == 1 else instance
        
        explanations = {
            'timestamp': datetime.now().isoformat(),
            'model_id': model_id,
            'prediction': None,
            'confidence': None,
            'explanations': {}
        }
        
        try:
            # Get prediction and confidence
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(instance)[0]
                prediction = model.predict(instance)[0]
                confidence = np.max(prediction_proba)
                explanations['prediction'] = int(prediction)
                explanations['confidence'] = float(confidence)
                explanations['prediction_probabilities'] = prediction_proba.tolist()
            else:
                prediction = model.predict(instance)[0]
                explanations['prediction'] = float(prediction)
            
            # Generate SHAP explanation
            if 'shap' in explanation_types and self.shap_enabled and model_id in self.shap_explainers:
                shap_explanation = self._generate_shap_explanation(model_id, instance)
                if shap_explanation:
                    explanations['explanations']['shap'] = shap_explanation
                    self.stats['shap_explanations'] += 1
            
            # Generate LIME explanation
            if 'lime' in explanation_types and self.lime_enabled and model_id in self.lime_explainers:
                lime_explanation = self._generate_lime_explanation(model, model_id, instance)
                if lime_explanation:
                    explanations['explanations']['lime'] = lime_explanation
                    self.stats['lime_explanations'] += 1
            
            # Generate feature importance
            if 'feature_importance' in explanation_types and self.feature_importance_enabled:
                feature_importance = self._generate_feature_importance(model, instance)
                if feature_importance:
                    explanations['explanations']['feature_importance'] = feature_importance
                    self.stats['feature_importance_calculated'] += 1
            
            # Generate decision path (for tree-based models)
            if 'decision_path' in explanation_types and self.decision_paths_enabled:
                decision_path = self._generate_decision_path(model, instance)
                if decision_path:
                    explanations['explanations']['decision_path'] = decision_path
                    self.stats['decision_paths_extracted'] += 1
            
            # Generate uncertainty quantification
            if 'uncertainty' in explanation_types:
                uncertainty = self._calculate_uncertainty(model, instance)
                if uncertainty:
                    explanations['explanations']['uncertainty'] = uncertainty
            
            self.stats['explanations_generated'] += 1
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            explanations['error'] = str(e)
        
        return explanations
    
    def _generate_shap_explanation(self, model_id: int, instance: np.ndarray) -> Optional[Dict[str, Any]]:
        """Generate SHAP explanation for an instance."""
        try:
            explainer = self.shap_explainers[model_id]
            shap_values = explainer.shap_values(instance)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class case - use positive class
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            if shap_values.ndim > 1:
                shap_values = shap_values[0]  # Take first instance
            
            # Create feature importance ranking
            feature_importance = []
            for i, (feature, value) in enumerate(zip(self.feature_names, shap_values)):
                feature_importance.append({
                    'feature': feature,
                    'shap_value': float(value),
                    'abs_importance': float(abs(value)),
                    'rank': i + 1
                })
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: x['abs_importance'], reverse=True)
            
            # Update ranks
            for i, item in enumerate(feature_importance):
                item['rank'] = i + 1
            
            return {
                'method': 'SHAP',
                'feature_importance': feature_importance,
                'total_impact': float(np.sum(np.abs(shap_values))),
                'base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return None
    
    def _generate_lime_explanation(self, model: Any, model_id: int, instance: np.ndarray) -> Optional[Dict[str, Any]]:
        """Generate LIME explanation for an instance."""
        try:
            explainer = self.lime_explainers[model_id]
            
            # Generate explanation
            if hasattr(model, 'predict_proba'):
                explanation = explainer.explain_instance(
                    instance[0], 
                    model.predict_proba,
                    num_features=len(self.feature_names)
                )
            else:
                explanation = explainer.explain_instance(
                    instance[0], 
                    model.predict,
                    num_features=len(self.feature_names)
                )
            
            # Extract feature importance
            feature_importance = []
            for feature, importance in explanation.as_list():
                feature_importance.append({
                    'feature': feature,
                    'lime_value': float(importance),
                    'abs_importance': float(abs(importance))
                })
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: x['abs_importance'], reverse=True)
            
            # Update ranks
            for i, item in enumerate(feature_importance):
                item['rank'] = i + 1
            
            return {
                'method': 'LIME',
                'feature_importance': feature_importance,
                'score': float(explanation.score) if hasattr(explanation, 'score') else None,
                'intercept': float(explanation.intercept) if hasattr(explanation, 'intercept') else None
            }
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return None
    
    def _generate_feature_importance(self, model: Any, instance: np.ndarray) -> Optional[Dict[str, Any]]:
        """Generate feature importance explanation."""
        try:
            feature_importance = []
            
            # Try to get built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, (feature, importance) in enumerate(zip(self.feature_names, importances)):
                    feature_importance.append({
                        'feature': feature,
                        'importance': float(importance),
                        'rank': i + 1
                    })
                
                # Sort by importance
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                
                # Update ranks
                for i, item in enumerate(feature_importance):
                    item['rank'] = i + 1
                
                return {
                    'method': 'Built-in Feature Importance',
                    'feature_importance': feature_importance,
                    'total_importance': float(np.sum(importances))
                }
            
            # For models without built-in feature importance, use coefficients
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef[0]  # Take first class for multi-class
                
                for i, (feature, coef_val) in enumerate(zip(self.feature_names, coef)):
                    feature_importance.append({
                        'feature': feature,
                        'coefficient': float(coef_val),
                        'abs_coefficient': float(abs(coef_val)),
                        'rank': i + 1
                    })
                
                # Sort by absolute coefficient
                feature_importance.sort(key=lambda x: x['abs_coefficient'], reverse=True)
                
                # Update ranks
                for i, item in enumerate(feature_importance):
                    item['rank'] = i + 1
                
                return {
                    'method': 'Model Coefficients',
                    'feature_importance': feature_importance
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating feature importance: {e}")
            return None
    
    def _generate_decision_path(self, model: Any, instance: np.ndarray) -> Optional[Dict[str, Any]]:
        """Generate decision path for tree-based models."""
        try:
            # Only works for tree-based models
            if not hasattr(model, 'decision_path'):
                return None
            
            # Get decision path
            leaf_id = model.apply(instance)
            feature = model.tree_.feature
            threshold = model.tree_.threshold
            
            # Extract path
            path_info = []
            node_indicator = model.decision_path(instance)
            leaf_id = leaf_id[0]
            
            for node_id in node_indicator.indices:
                if leaf_id == node_id:
                    continue
                
                if instance[0][feature[node_id]] <= threshold[node_id]:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"
                
                path_info.append({
                    'node_id': int(node_id),
                    'feature': self.feature_names[feature[node_id]] if feature[node_id] >= 0 else 'leaf',
                    'threshold': float(threshold[node_id]),
                    'threshold_sign': threshold_sign,
                    'value': float(instance[0][feature[node_id]]) if feature[node_id] >= 0 else None
                })
            
            return {
                'method': 'Decision Path',
                'path': path_info,
                'leaf_id': int(leaf_id)
            }
            
        except Exception as e:
            logger.error(f"Error generating decision path: {e}")
            return None
    
    def _calculate_uncertainty(self, model: Any, instance: np.ndarray) -> Optional[Dict[str, Any]]:
        """Calculate prediction uncertainty."""
        try:
            uncertainty_info = {}
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(instance)[0]
                
                # Entropy-based uncertainty
                entropy = -np.sum(proba * np.log(proba + 1e-10))
                uncertainty_info['entropy'] = float(entropy)
                
                # Max probability (confidence)
                max_prob = np.max(proba)
                uncertainty_info['confidence'] = float(max_prob)
                uncertainty_info['uncertainty'] = float(1 - max_prob)
                
                # Margin (difference between top two predictions)
                sorted_proba = np.sort(proba)[::-1]
                if len(sorted_proba) > 1:
                    margin = sorted_proba[0] - sorted_proba[1]
                    uncertainty_info['margin'] = float(margin)
                
                # Prediction distribution
                uncertainty_info['prediction_distribution'] = proba.tolist()
            
            return uncertainty_info
            
        except Exception as e:
            logger.error(f"Error calculating uncertainty: {e}")
            return None
    
    def generate_global_explanations(self, model: Any, X_sample: np.ndarray, 
                                   y_sample: np.ndarray = None) -> Dict[str, Any]:
        """Generate global model explanations.
        
        Args:
            model: Trained ML model
            X_sample: Sample of data for global analysis
            y_sample: Sample of labels (optional)
            
        Returns:
            Dictionary containing global explanations
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        model_id = id(model)
        global_explanations = {
            'timestamp': datetime.now().isoformat(),
            'model_id': model_id,
            'sample_size': len(X_sample),
            'explanations': {}
        }
        
        try:
            # Global feature importance
            if self.feature_importance_enabled:
                global_importance = self._generate_global_feature_importance(model, X_sample, y_sample)
                if global_importance:
                    global_explanations['explanations']['global_feature_importance'] = global_importance
            
            # SHAP summary
            if self.shap_enabled and model_id in self.shap_explainers:
                shap_summary = self._generate_shap_summary(model_id, X_sample)
                if shap_summary:
                    global_explanations['explanations']['shap_summary'] = shap_summary
            
            # Model performance summary
            if y_sample is not None:
                performance_summary = self._generate_performance_summary(model, X_sample, y_sample)
                if performance_summary:
                    global_explanations['explanations']['performance'] = performance_summary
            
        except Exception as e:
            logger.error(f"Error generating global explanations: {e}")
            global_explanations['error'] = str(e)
        
        return global_explanations
    
    def _generate_global_feature_importance(self, model: Any, X_sample: np.ndarray, 
                                          y_sample: np.ndarray = None) -> Optional[Dict[str, Any]]:
        """Generate global feature importance analysis."""
        try:
            importance_data = {}
            
            # Built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = []
                
                for i, (feature, importance) in enumerate(zip(self.feature_names, importances)):
                    feature_importance.append({
                        'feature': feature,
                        'importance': float(importance),
                        'rank': i + 1
                    })
                
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                for i, item in enumerate(feature_importance):
                    item['rank'] = i + 1
                
                importance_data['built_in'] = feature_importance
            
            # Permutation importance (if labels available)
            if y_sample is not None:
                try:
                    perm_importance = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=42)
                    
                    perm_feature_importance = []
                    for i, (feature, importance, std) in enumerate(zip(self.feature_names, 
                                                                      perm_importance.importances_mean,
                                                                      perm_importance.importances_std)):
                        perm_feature_importance.append({
                            'feature': feature,
                            'importance_mean': float(importance),
                            'importance_std': float(std),
                            'rank': i + 1
                        })
                    
                    perm_feature_importance.sort(key=lambda x: x['importance_mean'], reverse=True)
                    for i, item in enumerate(perm_feature_importance):
                        item['rank'] = i + 1
                    
                    importance_data['permutation'] = perm_feature_importance
                    
                except Exception as e:
                    logger.warning(f"Could not calculate permutation importance: {e}")
            
            return importance_data if importance_data else None
            
        except Exception as e:
            logger.error(f"Error generating global feature importance: {e}")
            return None
    
    def _generate_shap_summary(self, model_id: int, X_sample: np.ndarray) -> Optional[Dict[str, Any]]:
        """Generate SHAP summary for global explanations."""
        try:
            explainer = self.shap_explainers[model_id]
            
            # Calculate SHAP values for sample
            sample_size = min(100, len(X_sample))  # Limit for performance
            X_sample_subset = X_sample[:sample_size]
            
            shap_values = explainer.shap_values(X_sample_subset)
            
            # Handle different formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Calculate mean absolute SHAP values
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            
            # Create summary
            shap_summary = []
            for i, (feature, mean_shap) in enumerate(zip(self.feature_names, mean_shap_values)):
                shap_summary.append({
                    'feature': feature,
                    'mean_abs_shap': float(mean_shap),
                    'rank': i + 1
                })
            
            shap_summary.sort(key=lambda x: x['mean_abs_shap'], reverse=True)
            for i, item in enumerate(shap_summary):
                item['rank'] = i + 1
            
            return {
                'sample_size': sample_size,
                'feature_importance': shap_summary,
                'total_importance': float(np.sum(mean_shap_values))
            }
            
        except Exception as e:
            logger.error(f"Error generating SHAP summary: {e}")
            return None
    
    def _generate_performance_summary(self, model: Any, X_sample: np.ndarray, 
                                    y_sample: np.ndarray) -> Optional[Dict[str, Any]]:
        """Generate model performance summary."""
        try:
            predictions = model.predict(X_sample)
            
            performance = {
                'sample_size': len(X_sample),
                'accuracy': float(np.mean(predictions == y_sample))
            }
            
            # Add probability-based metrics if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_sample)
                max_probabilities = np.max(probabilities, axis=1)
                
                performance['mean_confidence'] = float(np.mean(max_probabilities))
                performance['confidence_std'] = float(np.std(max_probabilities))
                performance['low_confidence_ratio'] = float(np.mean(max_probabilities < 0.7))
            
            return performance
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return None
    
    def get_explanation_summary(self, model_id: int = None) -> Dict[str, Any]:
        """Get summary of explanations generated."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats.copy(),
            'configuration': {
                'enabled': self.enabled,
                'shap_enabled': self.shap_enabled,
                'lime_enabled': self.lime_enabled,
                'feature_importance_enabled': self.feature_importance_enabled,
                'decision_paths_enabled': self.decision_paths_enabled
            },
            'available_explainers': {
                'shap_models': len(self.shap_explainers),
                'lime_models': len(self.lime_explainers)
            }
        }
        
        if model_id and model_id in self.model_metadata:
            summary['model_info'] = self.model_metadata[model_id]
        
        return summary
    
    def clear_cache(self):
        """Clear explanation cache."""
        self.explanations_cache.clear()
        logger.info("Explanation cache cleared")
    
    def save_explainers(self, filepath: str):
        """Save explainers to file."""
        try:
            explainer_data = {
                'shap_explainers': {},  # SHAP explainers can't be easily pickled
                'lime_explainers': {},  # LIME explainers can't be easily pickled
                'model_metadata': self.model_metadata,
                'feature_names': self.feature_names,
                'stats': self.stats,
                'config': self.interpretability_config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(explainer_data, f)
            
            logger.info(f"Explainer metadata saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving explainers: {e}")
    
    def load_explainers(self, filepath: str):
        """Load explainers from file."""
        try:
            with open(filepath, 'rb') as f:
                explainer_data = pickle.load(f)
            
            self.model_metadata = explainer_data.get('model_metadata', {})
            self.feature_names = explainer_data.get('feature_names', [])
            self.stats = explainer_data.get('stats', self.stats)
            
            logger.info(f"Explainer metadata loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading explainers: {e}")