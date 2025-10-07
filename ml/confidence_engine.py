"""Confidence and Uncertainty Quantification Engine for ML Models.

This module provides comprehensive confidence assessment and uncertainty quantification
for machine learning predictions, including calibration, ensemble uncertainty,
and confidence interval estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
import json
from collections import defaultdict, deque
import warnings

# ML imports
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats

# Statistical imports
try:
    from scipy import stats
    from scipy.special import softmax
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some uncertainty features will be limited.")

logger = logging.getLogger(__name__)


class ConfidenceEngine:
    """Advanced confidence and uncertainty quantification engine."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the confidence engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.confidence_config = config.get('ml', {}).get('confidence', {})
        
        # Configuration parameters
        self.enabled = self.confidence_config.get('enabled', True)
        self.calibration_enabled = self.confidence_config.get('calibration_enabled', True)
        self.uncertainty_methods = self.confidence_config.get('uncertainty_methods', 
                                                            ['entropy', 'margin', 'variance'])
        self.confidence_threshold = self.confidence_config.get('confidence_threshold', 0.8)
        self.calibration_method = self.confidence_config.get('calibration_method', 'isotonic')
        
        # Calibration models
        self.calibration_models = {}
        self.calibration_history = defaultdict(list)
        
        # Uncertainty tracking
        self.uncertainty_history = deque(maxlen=1000)
        self.confidence_statistics = {
            'total_predictions': 0,
            'high_confidence_predictions': 0,
            'low_confidence_predictions': 0,
            'calibration_score': None,
            'average_confidence': 0.0,
            'confidence_distribution': defaultdict(int)
        }
        
        # Ensemble tracking for uncertainty estimation
        self.ensemble_predictions = defaultdict(list)
        
        logger.info(f"ConfidenceEngine initialized with methods: {self.uncertainty_methods}")
    
    def calibrate_model(self, model: Any, X_cal: np.ndarray, y_cal: np.ndarray, 
                       model_id: str = None) -> Dict[str, Any]:
        """Calibrate model probabilities for better confidence estimation.
        
        Args:
            model: Trained ML model
            X_cal: Calibration features
            y_cal: Calibration labels
            model_id: Unique identifier for the model
            
        Returns:
            Dictionary containing calibration results
        """
        if not self.enabled or not self.calibration_enabled:
            return {'status': 'disabled'}
        
        if model_id is None:
            model_id = str(id(model))
        
        try:
            # Get uncalibrated predictions
            if hasattr(model, 'predict_proba'):
                uncalibrated_probs = model.predict_proba(X_cal)
            else:
                logger.warning("Model does not support probability prediction")
                return {'status': 'unsupported', 'message': 'Model does not support probabilities'}
            
            # Create calibrated classifier
            if self.calibration_method == 'isotonic':
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            else:  # sigmoid
                calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            
            # Fit calibration
            calibrated_model.fit(X_cal, y_cal)
            
            # Get calibrated predictions
            calibrated_probs = calibrated_model.predict_proba(X_cal)
            
            # Calculate calibration metrics
            calibration_metrics = self._calculate_calibration_metrics(
                y_cal, uncalibrated_probs, calibrated_probs
            )
            
            # Store calibrated model
            self.calibration_models[model_id] = calibrated_model
            
            # Update calibration history
            self.calibration_history[model_id].append({
                'timestamp': datetime.now(),
                'calibration_size': len(X_cal),
                'metrics': calibration_metrics
            })
            
            logger.info(f"Model {model_id} calibrated with {self.calibration_method} method")
            
            return {
                'status': 'success',
                'model_id': model_id,
                'calibration_method': self.calibration_method,
                'metrics': calibration_metrics,
                'calibration_size': len(X_cal)
            }
            
        except Exception as e:
            logger.error(f"Error calibrating model: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def assess_prediction_confidence(self, model: Any, instance: np.ndarray, 
                                   model_id: str = None, 
                                   ensemble_predictions: List[np.ndarray] = None) -> Dict[str, Any]:
        """Assess confidence and uncertainty for a single prediction.
        
        Args:
            model: Trained ML model
            instance: Single instance to assess (1D or 2D array)
            model_id: Model identifier for calibration lookup
            ensemble_predictions: List of predictions from ensemble models
            
        Returns:
            Dictionary containing comprehensive confidence assessment
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        if model_id is None:
            model_id = str(id(model))
        
        instance = instance.reshape(1, -1) if instance.ndim == 1 else instance
        
        try:
            confidence_assessment = {
                'timestamp': datetime.now().isoformat(),
                'model_id': model_id,
                'prediction': None,
                'raw_probabilities': None,
                'calibrated_probabilities': None,
                'confidence_metrics': {},
                'uncertainty_metrics': {},
                'confidence_level': 'unknown',
                'reliability_score': 0.0
            }
            
            # Get base prediction and probabilities
            if hasattr(model, 'predict_proba'):
                raw_probs = model.predict_proba(instance)[0]
                prediction = model.predict(instance)[0]
                confidence_assessment['prediction'] = int(prediction) if isinstance(prediction, (np.integer, int)) else float(prediction)
                confidence_assessment['raw_probabilities'] = raw_probs.tolist()
                
                # Get calibrated probabilities if available
                if model_id in self.calibration_models:
                    calibrated_probs = self.calibration_models[model_id].predict_proba(instance)[0]
                    confidence_assessment['calibrated_probabilities'] = calibrated_probs.tolist()
                    working_probs = calibrated_probs
                else:
                    working_probs = raw_probs
                
                # Calculate confidence metrics
                confidence_metrics = self._calculate_confidence_metrics(working_probs)
                confidence_assessment['confidence_metrics'] = confidence_metrics
                
                # Calculate uncertainty metrics
                uncertainty_metrics = self._calculate_uncertainty_metrics(
                    working_probs, ensemble_predictions
                )
                confidence_assessment['uncertainty_metrics'] = uncertainty_metrics
                
                # Determine confidence level
                max_prob = np.max(working_probs)
                if max_prob >= 0.9:
                    confidence_level = 'very_high'
                elif max_prob >= 0.8:
                    confidence_level = 'high'
                elif max_prob >= 0.6:
                    confidence_level = 'medium'
                elif max_prob >= 0.4:
                    confidence_level = 'low'
                else:
                    confidence_level = 'very_low'
                
                confidence_assessment['confidence_level'] = confidence_level
                
                # Calculate reliability score
                reliability_score = self._calculate_reliability_score(
                    confidence_metrics, uncertainty_metrics
                )
                confidence_assessment['reliability_score'] = reliability_score
                
            else:
                # Regression or models without probabilities
                prediction = model.predict(instance)[0]
                confidence_assessment['prediction'] = float(prediction)
                
                # For regression, use prediction intervals or ensemble variance
                if ensemble_predictions:
                    ensemble_array = np.array(ensemble_predictions)
                    prediction_std = np.std(ensemble_array)
                    prediction_mean = np.mean(ensemble_array)
                    
                    confidence_assessment['uncertainty_metrics'] = {
                        'prediction_std': float(prediction_std),
                        'prediction_variance': float(prediction_std ** 2),
                        'coefficient_of_variation': float(prediction_std / abs(prediction_mean)) if prediction_mean != 0 else float('inf')
                    }
            
            # Update statistics
            self._update_confidence_statistics(confidence_assessment)
            
            # Store in history
            self.uncertainty_history.append({
                'timestamp': datetime.now(),
                'model_id': model_id,
                'confidence_level': confidence_assessment['confidence_level'],
                'reliability_score': confidence_assessment['reliability_score']
            })
            
            return confidence_assessment
            
        except Exception as e:
            logger.error(f"Error assessing prediction confidence: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _calculate_calibration_metrics(self, y_true: np.ndarray, 
                                     uncalibrated_probs: np.ndarray, 
                                     calibrated_probs: np.ndarray) -> Dict[str, float]:
        """Calculate calibration quality metrics."""
        try:
            metrics = {}
            
            # Brier score (lower is better)
            if uncalibrated_probs.shape[1] == 2:  # Binary classification
                uncal_brier = brier_score_loss(y_true, uncalibrated_probs[:, 1])
                cal_brier = brier_score_loss(y_true, calibrated_probs[:, 1])
                
                metrics['uncalibrated_brier_score'] = float(uncal_brier)
                metrics['calibrated_brier_score'] = float(cal_brier)
                metrics['brier_score_improvement'] = float(uncal_brier - cal_brier)
            
            # Log loss (lower is better)
            try:
                uncal_logloss = log_loss(y_true, uncalibrated_probs)
                cal_logloss = log_loss(y_true, calibrated_probs)
                
                metrics['uncalibrated_log_loss'] = float(uncal_logloss)
                metrics['calibrated_log_loss'] = float(cal_logloss)
                metrics['log_loss_improvement'] = float(uncal_logloss - cal_logloss)
            except Exception as e:
                logger.warning(f"Could not calculate log loss: {e}")
            
            # Expected Calibration Error (ECE)
            ece_uncal = self._calculate_ece(y_true, uncalibrated_probs)
            ece_cal = self._calculate_ece(y_true, calibrated_probs)
            
            metrics['uncalibrated_ece'] = float(ece_uncal)
            metrics['calibrated_ece'] = float(ece_cal)
            metrics['ece_improvement'] = float(ece_uncal - ece_cal)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating calibration metrics: {e}")
            return {}
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        try:
            if y_prob.shape[1] == 2:  # Binary classification
                y_prob_max = y_prob[:, 1]
                y_pred = (y_prob_max > 0.5).astype(int)
            else:  # Multi-class
                y_prob_max = np.max(y_prob, axis=1)
                y_pred = np.argmax(y_prob, axis=1)
            
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob_max > bin_lower) & (y_prob_max <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
                    avg_confidence_in_bin = y_prob_max[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
            
        except Exception as e:
            logger.error(f"Error calculating ECE: {e}")
            return 0.0
    
    def _calculate_confidence_metrics(self, probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate various confidence metrics from probabilities."""
        try:
            metrics = {}
            
            # Basic confidence (max probability)
            max_prob = np.max(probabilities)
            metrics['max_probability'] = float(max_prob)
            metrics['confidence'] = float(max_prob)
            
            # Margin (difference between top two predictions)
            sorted_probs = np.sort(probabilities)[::-1]
            if len(sorted_probs) > 1:
                margin = sorted_probs[0] - sorted_probs[1]
                metrics['margin'] = float(margin)
                metrics['normalized_margin'] = float(margin / sorted_probs[0]) if sorted_probs[0] > 0 else 0.0
            
            # Entropy-based uncertainty
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            max_entropy = np.log(len(probabilities))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            metrics['entropy'] = float(entropy)
            metrics['normalized_entropy'] = float(normalized_entropy)
            metrics['entropy_confidence'] = float(1 - normalized_entropy)
            
            # Gini coefficient (concentration measure)
            sorted_probs_asc = np.sort(probabilities)
            n = len(probabilities)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_probs_asc))) / (n * np.sum(sorted_probs_asc)) - (n + 1) / n
            metrics['gini_coefficient'] = float(gini)
            
            # Variance of probabilities
            prob_variance = np.var(probabilities)
            metrics['probability_variance'] = float(prob_variance)
            
            # Top-k confidence (sum of top k probabilities)
            for k in [2, 3]:
                if len(probabilities) >= k:
                    top_k_sum = np.sum(sorted_probs[:k])
                    metrics[f'top_{k}_confidence'] = float(top_k_sum)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating confidence metrics: {e}")
            return {}
    
    def _calculate_uncertainty_metrics(self, probabilities: np.ndarray, 
                                     ensemble_predictions: List[np.ndarray] = None) -> Dict[str, float]:
        """Calculate uncertainty metrics."""
        try:
            metrics = {}
            
            # Aleatoric uncertainty (from single model probabilities)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            metrics['aleatoric_uncertainty'] = float(entropy)
            
            # Epistemic uncertainty (from ensemble if available)
            if ensemble_predictions and len(ensemble_predictions) > 1:
                ensemble_array = np.array(ensemble_predictions)
                
                # Mean prediction across ensemble
                mean_prediction = np.mean(ensemble_array, axis=0)
                
                # Variance across ensemble predictions
                prediction_variance = np.var(ensemble_array, axis=0)
                metrics['epistemic_uncertainty'] = float(np.mean(prediction_variance))
                
                # Total uncertainty
                total_uncertainty = entropy + np.mean(prediction_variance)
                metrics['total_uncertainty'] = float(total_uncertainty)
                
                # Mutual information (epistemic uncertainty for classification)
                if len(mean_prediction.shape) == 1 and len(mean_prediction) > 1:  # Classification
                    mean_entropy = -np.sum(mean_prediction * np.log(mean_prediction + 1e-10))
                    expected_entropy = np.mean([
                        -np.sum(pred * np.log(pred + 1e-10)) for pred in ensemble_array
                    ])
                    mutual_info = mean_entropy - expected_entropy
                    metrics['mutual_information'] = float(mutual_info)
                
                # Ensemble agreement
                if len(ensemble_array.shape) == 2:  # Classification probabilities
                    # Calculate agreement as inverse of variance in max predictions
                    max_predictions = np.argmax(ensemble_array, axis=1)
                    agreement = 1.0 - (np.var(max_predictions) / max(len(np.unique(max_predictions)), 1))
                    metrics['ensemble_agreement'] = float(agreement)
                
                # Prediction interval width (for regression-like outputs)
                if len(ensemble_array.shape) == 1 or ensemble_array.shape[1] == 1:
                    prediction_std = np.std(ensemble_array)
                    # 95% prediction interval
                    interval_width = 2 * 1.96 * prediction_std
                    metrics['prediction_interval_width'] = float(interval_width)
            
            # Predictive uncertainty measures
            max_prob = np.max(probabilities)
            metrics['predictive_uncertainty'] = float(1 - max_prob)
            
            # Differential entropy for continuous approximation
            if SCIPY_AVAILABLE:
                try:
                    # Approximate differential entropy
                    kde_entropy = stats.entropy(probabilities + 1e-10)
                    metrics['differential_entropy'] = float(kde_entropy)
                except Exception:
                    pass
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating uncertainty metrics: {e}")
            return {}
    
    def _calculate_reliability_score(self, confidence_metrics: Dict[str, float], 
                                   uncertainty_metrics: Dict[str, float]) -> float:
        """Calculate overall reliability score combining confidence and uncertainty."""
        try:
            # Base confidence score
            confidence_score = confidence_metrics.get('confidence', 0.0)
            
            # Penalty for high uncertainty
            entropy_penalty = uncertainty_metrics.get('aleatoric_uncertainty', 0.0) / np.log(10)  # Normalize
            
            # Bonus for low epistemic uncertainty (if available)
            epistemic_bonus = 0.0
            if 'epistemic_uncertainty' in uncertainty_metrics:
                epistemic_uncertainty = uncertainty_metrics['epistemic_uncertainty']
                epistemic_bonus = max(0, 0.1 - epistemic_uncertainty)  # Small bonus for low epistemic uncertainty
            
            # Margin bonus
            margin_bonus = confidence_metrics.get('normalized_margin', 0.0) * 0.1
            
            # Calculate final reliability score
            reliability_score = confidence_score - entropy_penalty + epistemic_bonus + margin_bonus
            reliability_score = max(0.0, min(1.0, reliability_score))  # Clamp to [0, 1]
            
            return float(reliability_score)
            
        except Exception as e:
            logger.error(f"Error calculating reliability score: {e}")
            return 0.0
    
    def _update_confidence_statistics(self, confidence_assessment: Dict[str, Any]):
        """Update running confidence statistics."""
        try:
            self.confidence_statistics['total_predictions'] += 1
            
            confidence_level = confidence_assessment.get('confidence_level', 'unknown')
            reliability_score = confidence_assessment.get('reliability_score', 0.0)
            
            # Update confidence level counts
            if confidence_level in ['high', 'very_high']:
                self.confidence_statistics['high_confidence_predictions'] += 1
            elif confidence_level in ['low', 'very_low']:
                self.confidence_statistics['low_confidence_predictions'] += 1
            
            # Update confidence distribution
            self.confidence_statistics['confidence_distribution'][confidence_level] += 1
            
            # Update average confidence (running average)
            total = self.confidence_statistics['total_predictions']
            current_avg = self.confidence_statistics['average_confidence']
            new_avg = ((current_avg * (total - 1)) + reliability_score) / total
            self.confidence_statistics['average_confidence'] = new_avg
            
        except Exception as e:
            logger.error(f"Error updating confidence statistics: {e}")
    
    def get_confidence_summary(self, model_id: str = None) -> Dict[str, Any]:
        """Get comprehensive confidence and uncertainty summary.
        
        Args:
            model_id: Optional model ID to filter results
            
        Returns:
            Dictionary containing confidence summary
        """
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'statistics': self.confidence_statistics.copy(),
                'configuration': {
                    'enabled': self.enabled,
                    'calibration_enabled': self.calibration_enabled,
                    'uncertainty_methods': self.uncertainty_methods,
                    'confidence_threshold': self.confidence_threshold,
                    'calibration_method': self.calibration_method
                },
                'calibrated_models': len(self.calibration_models),
                'uncertainty_history_size': len(self.uncertainty_history)
            }
            
            # Add model-specific information
            if model_id and model_id in self.calibration_history:
                summary['model_calibration_history'] = self.calibration_history[model_id]
            
            # Add recent uncertainty trends
            if self.uncertainty_history:
                recent_history = list(self.uncertainty_history)[-100:]  # Last 100 predictions
                
                confidence_levels = [h['confidence_level'] for h in recent_history]
                reliability_scores = [h['reliability_score'] for h in recent_history]
                
                summary['recent_trends'] = {
                    'confidence_level_distribution': {
                        level: confidence_levels.count(level) for level in set(confidence_levels)
                    },
                    'average_reliability_score': float(np.mean(reliability_scores)),
                    'reliability_score_std': float(np.std(reliability_scores)),
                    'trend_period': len(recent_history)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating confidence summary: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def detect_confidence_drift(self, window_size: int = 100) -> Dict[str, Any]:
        """Detect drift in confidence patterns over time.
        
        Args:
            window_size: Size of the sliding window for drift detection
            
        Returns:
            Dictionary containing drift analysis
        """
        try:
            if len(self.uncertainty_history) < window_size * 2:
                return {
                    'status': 'insufficient_data',
                    'message': f'Need at least {window_size * 2} predictions for drift detection'
                }
            
            recent_history = list(self.uncertainty_history)
            
            # Split into two windows
            older_window = recent_history[-window_size*2:-window_size]
            recent_window = recent_history[-window_size:]
            
            # Extract reliability scores
            older_scores = [h['reliability_score'] for h in older_window]
            recent_scores = [h['reliability_score'] for h in recent_window]
            
            # Statistical tests for drift
            drift_analysis = {
                'status': 'success',  # Explicit success status for test expectations
                'timestamp': datetime.now().isoformat(),
                'window_size': window_size,
                'older_window_stats': {
                    'mean': float(np.mean(older_scores)),
                    'std': float(np.std(older_scores)),
                    'median': float(np.median(older_scores))
                },
                'recent_window_stats': {
                    'mean': float(np.mean(recent_scores)),
                    'std': float(np.std(recent_scores)),
                    'median': float(np.median(recent_scores))
                }
            }
            
            # Calculate drift metrics
            mean_drift = np.mean(recent_scores) - np.mean(older_scores)
            std_drift = np.std(recent_scores) - np.std(older_scores)
            
            drift_analysis['drift_metrics'] = {
                'mean_drift': float(mean_drift),
                'std_drift': float(std_drift),
                'relative_mean_drift': float(mean_drift / np.mean(older_scores)) if np.mean(older_scores) != 0 else 0.0
            }
            
            # Statistical significance test
            if SCIPY_AVAILABLE:
                try:
                    t_stat, p_value = stats.ttest_ind(older_scores, recent_scores)
                    drift_analysis['statistical_test'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant_drift': p_value < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Could not perform statistical test: {e}")
            
            # Drift severity assessment
            if abs(mean_drift) > 0.1:
                severity = 'high'
            elif abs(mean_drift) > 0.05:
                severity = 'medium'
            else:
                severity = 'low'
            
            drift_analysis['drift_severity'] = severity
            drift_analysis['drift_direction'] = 'increasing' if mean_drift > 0 else 'decreasing'
            
            return drift_analysis
            
        except Exception as e:
            logger.error(f"Error detecting confidence drift: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def recommend_recalibration(self, model_id: str = None) -> Dict[str, Any]:
        """Recommend whether model recalibration is needed.
        
        Args:
            model_id: Model ID to analyze
            
        Returns:
            Dictionary containing recalibration recommendation
        """
        try:
            # Analyze recent confidence patterns
            drift_analysis = self.detect_confidence_drift()
            
            recommendation = {
                'timestamp': datetime.now().isoformat(),
                'model_id': model_id,
                'recalibration_needed': False,
                'reasons': [],
                'priority': 'low',
                'confidence_score': 0.0
            }
            
            # Check for significant drift
            if drift_analysis.get('status') == 'success':
                if drift_analysis.get('drift_severity') == 'high':
                    recommendation['recalibration_needed'] = True
                    recommendation['reasons'].append('High confidence drift detected')
                    recommendation['priority'] = 'high'
                
                if drift_analysis.get('statistical_test', {}).get('significant_drift', False):
                    recommendation['recalibration_needed'] = True
                    recommendation['reasons'].append('Statistically significant confidence change')
                    if recommendation['priority'] == 'low':
                        recommendation['priority'] = 'medium'
            
            # Check calibration history
            if model_id and model_id in self.calibration_history:
                last_calibration = self.calibration_history[model_id][-1]
                days_since_calibration = (datetime.now() - last_calibration['timestamp']).days
                
                if days_since_calibration > 30:  # Arbitrary threshold
                    recommendation['reasons'].append(f'Last calibration was {days_since_calibration} days ago')
                    if not recommendation['recalibration_needed']:
                        recommendation['priority'] = 'medium'
            
            # Check overall confidence statistics
            total_predictions = self.confidence_statistics['total_predictions']
            if total_predictions > 0:
                low_confidence_ratio = self.confidence_statistics['low_confidence_predictions'] / total_predictions
                
                if low_confidence_ratio > 0.3:  # More than 30% low confidence
                    recommendation['recalibration_needed'] = True
                    recommendation['reasons'].append(f'High ratio of low-confidence predictions: {low_confidence_ratio:.2%}')
                    recommendation['priority'] = 'high'
            
            # Calculate overall confidence in recommendation
            if recommendation['recalibration_needed']:
                confidence_score = min(1.0, len(recommendation['reasons']) * 0.3)
            else:
                confidence_score = 0.8  # High confidence in "no recalibration needed"
            
            recommendation['confidence_score'] = confidence_score
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recalibration recommendation: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def clear_history(self):
        """Clear confidence and uncertainty history."""
        self.uncertainty_history.clear()
        self.confidence_statistics = {
            'total_predictions': 0,
            'high_confidence_predictions': 0,
            'low_confidence_predictions': 0,
            'calibration_score': None,
            'average_confidence': 0.0,
            'confidence_distribution': defaultdict(int)
        }
        logger.info("Confidence history cleared")