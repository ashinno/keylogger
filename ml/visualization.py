"""Visualization components for ML interpretability features.

This module provides comprehensive visualization capabilities for machine learning
model interpretability, including feature importance plots, decision path diagrams,
SHAP visualizations, and confidence indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
import base64
from io import BytesIO
from datetime import datetime
import warnings

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle, FancyBboxPatch
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib seaborn")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

logger = logging.getLogger(__name__)


class InterpretabilityVisualizer:
    """Comprehensive visualization engine for ML interpretability."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the visualization engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.viz_config = config.get('ml', {}).get('visualization', {})
        
        # Configuration
        self.enabled = self.viz_config.get('enabled', True)
        self.use_plotly = self.viz_config.get('use_plotly', True) and PLOTLY_AVAILABLE
        self.use_matplotlib = self.viz_config.get('use_matplotlib', True) and MATPLOTLIB_AVAILABLE
        self.figure_size = self.viz_config.get('figure_size', (12, 8))
        self.dpi = self.viz_config.get('dpi', 100)
        self.color_palette = self.viz_config.get('color_palette', 'viridis')
        
        # Style configuration
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'available') and 'seaborn-v0_8' in plt.style.available else 'default')
            sns.set_palette(self.color_palette)
        
        logger.info(f"InterpretabilityVisualizer initialized - Plotly: {self.use_plotly}, Matplotlib: {self.use_matplotlib}")
    
    def create_feature_importance_plot(self, explanation_data: Dict[str, Any], 
                                     plot_type: str = 'bar', top_n: int = 20) -> Dict[str, Any]:
        """Create feature importance visualization.
        
        Args:
            explanation_data: Explanation data containing feature importance
            plot_type: Type of plot ('bar', 'horizontal_bar', 'waterfall')
            top_n: Number of top features to display
            
        Returns:
            Dictionary containing plot data and metadata
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        try:
            # Extract feature importance data
            feature_data = self._extract_feature_importance_data(explanation_data, top_n)
            if not feature_data:
                return {'status': 'no_data', 'message': 'No feature importance data found'}
            
            plots = {}
            
            # Create Plotly visualization
            if self.use_plotly:
                plotly_fig = self._create_plotly_feature_importance(feature_data, plot_type)
                if plotly_fig:
                    plots['plotly'] = {
                        'html': plotly_fig.to_html(include_plotlyjs='cdn'),
                        'json': plotly_fig.to_json()
                    }
            
            # Create Matplotlib visualization
            if self.use_matplotlib:
                matplotlib_fig = self._create_matplotlib_feature_importance(feature_data, plot_type)
                if matplotlib_fig:
                    plots['matplotlib'] = {
                        'base64': self._fig_to_base64(matplotlib_fig),
                        'svg': self._fig_to_svg(matplotlib_fig)
                    }
                    plt.close(matplotlib_fig)
            
            return {
                'status': 'success',
                'plots': plots,
                'metadata': {
                    'feature_count': len(feature_data['features']),
                    'plot_type': plot_type,
                    'top_n': top_n,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def create_decision_path_visualization(self, decision_path_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create decision path visualization.
        
        Args:
            decision_path_data: Decision path data from interpretability engine
            
        Returns:
            Dictionary containing visualization data
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        try:
            if 'decision_path' not in decision_path_data.get('explanations', {}):
                return {'status': 'no_data', 'message': 'No decision path data found'}
            
            path_info = decision_path_data['explanations']['decision_path']['path']
            
            plots = {}
            
            # Create Plotly decision tree visualization
            if self.use_plotly:
                plotly_fig = self._create_plotly_decision_path(path_info)
                if plotly_fig:
                    plots['plotly'] = {
                        'html': plotly_fig.to_html(include_plotlyjs='cdn'),
                        'json': plotly_fig.to_json()
                    }
            
            # Create Matplotlib decision path
            if self.use_matplotlib:
                matplotlib_fig = self._create_matplotlib_decision_path(path_info)
                if matplotlib_fig:
                    plots['matplotlib'] = {
                        'base64': self._fig_to_base64(matplotlib_fig),
                        'svg': self._fig_to_svg(matplotlib_fig)
                    }
                    plt.close(matplotlib_fig)
            
            return {
                'status': 'success',
                'plots': plots,
                'metadata': {
                    'path_length': len(path_info),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating decision path visualization: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def create_confidence_indicator(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create confidence level indicator visualization.
        
        Args:
            prediction_data: Prediction data with confidence information
            
        Returns:
            Dictionary containing confidence visualization
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        try:
            confidence = prediction_data.get('confidence', 0.0)
            prediction = prediction_data.get('prediction')
            probabilities = prediction_data.get('prediction_probabilities', [])
            
            plots = {}
            
            # Create Plotly confidence gauge
            if self.use_plotly:
                plotly_fig = self._create_plotly_confidence_gauge(confidence, prediction, probabilities)
                if plotly_fig:
                    plots['plotly'] = {
                        'html': plotly_fig.to_html(include_plotlyjs='cdn'),
                        'json': plotly_fig.to_json()
                    }
            
            # Create Matplotlib confidence indicator
            if self.use_matplotlib:
                matplotlib_fig = self._create_matplotlib_confidence_indicator(confidence, prediction, probabilities)
                if matplotlib_fig:
                    plots['matplotlib'] = {
                        'base64': self._fig_to_base64(matplotlib_fig),
                        'svg': self._fig_to_svg(matplotlib_fig)
                    }
                    plt.close(matplotlib_fig)
            
            return {
                'status': 'success',
                'plots': plots,
                'metadata': {
                    'confidence': confidence,
                    'prediction': prediction,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating confidence indicator: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def create_shap_waterfall_plot(self, shap_data: Dict[str, Any], top_n: int = 15) -> Dict[str, Any]:
        """Create SHAP waterfall plot showing feature contributions.
        
        Args:
            shap_data: SHAP explanation data
            top_n: Number of top features to show
            
        Returns:
            Dictionary containing waterfall plot
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        try:
            if 'shap' not in shap_data.get('explanations', {}):
                return {'status': 'no_data', 'message': 'No SHAP data found'}
            
            shap_explanation = shap_data['explanations']['shap']
            feature_importance = shap_explanation['feature_importance'][:top_n]
            base_value = shap_explanation.get('base_value', 0.0)
            
            plots = {}
            
            # Create Plotly waterfall chart
            if self.use_plotly:
                plotly_fig = self._create_plotly_shap_waterfall(feature_importance, base_value)
                if plotly_fig:
                    plots['plotly'] = {
                        'html': plotly_fig.to_html(include_plotlyjs='cdn'),
                        'json': plotly_fig.to_json()
                    }
            
            # Create Matplotlib waterfall chart
            if self.use_matplotlib:
                matplotlib_fig = self._create_matplotlib_shap_waterfall(feature_importance, base_value)
                if matplotlib_fig:
                    plots['matplotlib'] = {
                        'base64': self._fig_to_base64(matplotlib_fig),
                        'svg': self._fig_to_svg(matplotlib_fig)
                    }
                    plt.close(matplotlib_fig)
            
            return {
                'status': 'success',
                'plots': plots,
                'metadata': {
                    'feature_count': len(feature_importance),
                    'base_value': base_value,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating SHAP waterfall plot: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def create_uncertainty_visualization(self, uncertainty_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create uncertainty visualization.
        
        Args:
            uncertainty_data: Uncertainty data from interpretability engine
            
        Returns:
            Dictionary containing uncertainty visualization
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        try:
            if 'uncertainty' not in uncertainty_data.get('explanations', {}):
                return {'status': 'no_data', 'message': 'No uncertainty data found'}
            
            uncertainty_info = uncertainty_data['explanations']['uncertainty']
            
            plots = {}
            
            # Create Plotly uncertainty visualization
            if self.use_plotly:
                plotly_fig = self._create_plotly_uncertainty_plot(uncertainty_info)
                if plotly_fig:
                    plots['plotly'] = {
                        'html': plotly_fig.to_html(include_plotlyjs='cdn'),
                        'json': plotly_fig.to_json()
                    }
            
            # Create Matplotlib uncertainty visualization
            if self.use_matplotlib:
                matplotlib_fig = self._create_matplotlib_uncertainty_plot(uncertainty_info)
                if matplotlib_fig:
                    plots['matplotlib'] = {
                        'base64': self._fig_to_base64(matplotlib_fig),
                        'svg': self._fig_to_svg(matplotlib_fig)
                    }
                    plt.close(matplotlib_fig)
            
            return {
                'status': 'success',
                'plots': plots,
                'metadata': {
                    'entropy': uncertainty_info.get('entropy'),
                    'confidence': uncertainty_info.get('confidence'),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating uncertainty visualization: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _extract_feature_importance_data(self, explanation_data: Dict[str, Any], top_n: int) -> Optional[Dict[str, Any]]:
        """Extract and format feature importance data."""
        try:
            # Try different explanation types
            for explanation_type in ['shap', 'lime', 'feature_importance']:
                if explanation_type in explanation_data.get('explanations', {}):
                    explanation = explanation_data['explanations'][explanation_type]
                    
                    if 'feature_importance' in explanation:
                        features = explanation['feature_importance'][:top_n]
                        
                        # Determine value key based on explanation type
                        if explanation_type == 'shap':
                            value_key = 'shap_value'
                            abs_value_key = 'abs_importance'
                        elif explanation_type == 'lime':
                            value_key = 'lime_value'
                            abs_value_key = 'abs_importance'
                        else:
                            value_key = 'importance'
                            abs_value_key = 'importance'
                        
                        return {
                            'features': [f['feature'] for f in features],
                            'values': [f[value_key] for f in features],
                            'abs_values': [f.get(abs_value_key, abs(f[value_key])) for f in features],
                            'explanation_type': explanation_type
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting feature importance data: {e}")
            return None
    
    def _create_plotly_feature_importance(self, feature_data: Dict[str, Any], plot_type: str):
        """Create Plotly feature importance plot."""
        try:
            features = feature_data['features']
            values = feature_data['values']
            abs_values = feature_data['abs_values']
            
            # Create color scale based on positive/negative values
            colors = ['red' if v < 0 else 'blue' for v in values]
            
            if plot_type == 'horizontal_bar':
                fig = go.Figure(data=[
                    go.Bar(
                        y=features,
                        x=values,
                        orientation='h',
                        marker_color=colors,
                        text=[f'{v:.3f}' for v in values],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title=f'Feature Importance ({feature_data["explanation_type"].upper()})',
                    xaxis_title='Importance Value',
                    yaxis_title='Features',
                    height=max(400, len(features) * 25),
                    showlegend=False
                )
                
            else:  # vertical bar
                fig = go.Figure(data=[
                    go.Bar(
                        x=features,
                        y=abs_values,
                        marker_color=colors,
                        text=[f'{v:.3f}' for v in values],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title=f'Feature Importance ({feature_data["explanation_type"].upper()})',
                    xaxis_title='Features',
                    yaxis_title='Absolute Importance',
                    xaxis_tickangle=-45,
                    height=600,
                    showlegend=False
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Plotly feature importance plot: {e}")
            return None
    
    def _create_matplotlib_feature_importance(self, feature_data: Dict[str, Any], plot_type: str):
        """Create Matplotlib feature importance plot."""
        try:
            features = feature_data['features']
            values = feature_data['values']
            abs_values = feature_data['abs_values']
            
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Create color array
            colors = ['red' if v < 0 else 'blue' for v in values]
            
            if plot_type == 'horizontal_bar':
                bars = ax.barh(features, values, color=colors, alpha=0.7)
                ax.set_xlabel('Importance Value')
                ax.set_ylabel('Features')
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, values)):
                    ax.text(bar.get_width() + (0.01 if value >= 0 else -0.01), 
                           bar.get_y() + bar.get_height()/2, 
                           f'{value:.3f}', 
                           ha='left' if value >= 0 else 'right', 
                           va='center')
                
            else:  # vertical bar
                bars = ax.bar(range(len(features)), abs_values, color=colors, alpha=0.7)
                ax.set_xlabel('Features')
                ax.set_ylabel('Absolute Importance')
                ax.set_xticks(range(len(features)))
                ax.set_xticklabels(features, rotation=45, ha='right')
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, values)):
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + 0.01, 
                           f'{value:.3f}', 
                           ha='center', 
                           va='bottom')
            
            ax.set_title(f'Feature Importance ({feature_data["explanation_type"].upper()})')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            red_patch = mpatches.Patch(color='red', alpha=0.7, label='Negative Impact')
            blue_patch = mpatches.Patch(color='blue', alpha=0.7, label='Positive Impact')
            ax.legend(handles=[red_patch, blue_patch])
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Matplotlib feature importance plot: {e}")
            return None
    
    def _create_plotly_decision_path(self, path_info: List[Dict[str, Any]]):
        """Create Plotly decision path visualization."""
        try:
            # Create a flowchart-style decision path
            fig = go.Figure()
            
            y_positions = list(range(len(path_info), 0, -1))
            
            for i, (step, y_pos) in enumerate(zip(path_info, y_positions)):
                if step['feature'] == 'leaf':
                    continue
                
                # Create decision node
                fig.add_trace(go.Scatter(
                    x=[i],
                    y=[y_pos],
                    mode='markers+text',
                    marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
                    text=f"{step['feature']}<br>{step['threshold_sign']} {step['threshold']:.3f}",
                    textposition="middle center",
                    showlegend=False,
                    hovertemplate=f"<b>{step['feature']}</b><br>" +
                                f"Condition: {step['threshold_sign']} {step['threshold']:.3f}<br>" +
                                f"Value: {step['value']:.3f}<extra></extra>"
                ))
                
                # Add arrow to next step
                if i < len(path_info) - 1:
                    fig.add_annotation(
                        x=i, y=y_pos,
                        ax=i+1, ay=y_positions[i+1],
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='gray'
                    )
            
            fig.update_layout(
                title='Decision Path Visualization',
                xaxis_title='Decision Steps',
                yaxis_title='Path Flow',
                showlegend=False,
                height=max(400, len(path_info) * 60),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Plotly decision path: {e}")
            return None
    
    def _create_matplotlib_decision_path(self, path_info: List[Dict[str, Any]]):
        """Create Matplotlib decision path visualization."""
        try:
            fig, ax = plt.subplots(figsize=(12, max(6, len(path_info) * 0.8)), dpi=self.dpi)
            
            y_positions = list(range(len(path_info), 0, -1))
            
            for i, (step, y_pos) in enumerate(zip(path_info, y_positions)):
                if step['feature'] == 'leaf':
                    continue
                
                # Create decision box
                box = FancyBboxPatch(
                    (i-0.4, y_pos-0.3), 0.8, 0.6,
                    boxstyle="round,pad=0.1",
                    facecolor='lightblue',
                    edgecolor='darkblue',
                    linewidth=2
                )
                ax.add_patch(box)
                
                # Add text
                ax.text(i, y_pos, f"{step['feature']}\n{step['threshold_sign']} {step['threshold']:.3f}",
                       ha='center', va='center', fontsize=10, weight='bold')
                
                # Add arrow to next step
                if i < len(path_info) - 1:
                    ax.annotate('', xy=(i+1, y_positions[i+1]), xytext=(i, y_pos),
                               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
            
            ax.set_xlim(-0.5, len(path_info) - 0.5)
            ax.set_ylim(0.5, len(path_info) + 0.5)
            ax.set_xlabel('Decision Steps')
            ax.set_title('Decision Path Visualization')
            ax.set_xticks(range(len(path_info)))
            ax.set_xticklabels([f'Step {i+1}' for i in range(len(path_info))])
            ax.set_yticks([])
            ax.grid(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Matplotlib decision path: {e}")
            return None
    
    def _create_plotly_confidence_gauge(self, confidence: float, prediction: Any, probabilities: List[float]):
        """Create Plotly confidence gauge."""
        try:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Prediction Confidence (%)"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            # Add prediction information
            fig.add_annotation(
                text=f"Prediction: {prediction}<br>Confidence: {confidence:.3f}",
                xref="paper", yref="paper",
                x=0.5, y=0.1,
                showarrow=False,
                font=dict(size=14)
            )
            
            fig.update_layout(height=400)
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Plotly confidence gauge: {e}")
            return None
    
    def _create_matplotlib_confidence_indicator(self, confidence: float, prediction: Any, probabilities: List[float]):
        """Create Matplotlib confidence indicator."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
            
            # Confidence gauge
            theta = np.linspace(0, np.pi, 100)
            r = 1
            
            # Background arc
            ax1.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=10)
            
            # Confidence arc
            confidence_theta = theta[:int(confidence * 100)]
            color = 'green' if confidence > 0.8 else 'yellow' if confidence > 0.5 else 'red'
            ax1.plot(r * np.cos(confidence_theta), r * np.sin(confidence_theta), color, linewidth=10)
            
            # Add text
            ax1.text(0, -0.3, f'Confidence\n{confidence:.1%}', ha='center', va='center', 
                    fontsize=16, weight='bold')
            ax1.text(0, -0.6, f'Prediction: {prediction}', ha='center', va='center', fontsize=12)
            
            ax1.set_xlim(-1.2, 1.2)
            ax1.set_ylim(-0.8, 1.2)
            ax1.set_aspect('equal')
            ax1.axis('off')
            ax1.set_title('Prediction Confidence')
            
            # Probability distribution (if available)
            if probabilities:
                classes = [f'Class {i}' for i in range(len(probabilities))]
                bars = ax2.bar(classes, probabilities, alpha=0.7)
                ax2.set_ylabel('Probability')
                ax2.set_title('Class Probabilities')
                ax2.set_ylim(0, 1)
                
                # Highlight predicted class
                max_idx = np.argmax(probabilities)
                bars[max_idx].set_color('red')
                
                # Add value labels
                for bar, prob in zip(bars, probabilities):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{prob:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Matplotlib confidence indicator: {e}")
            return None
    
    def _create_plotly_shap_waterfall(self, feature_importance: List[Dict[str, Any]], base_value: float):
        """Create Plotly SHAP waterfall chart."""
        try:
            features = [f['feature'] for f in feature_importance]
            values = [f['shap_value'] for f in feature_importance]
            
            # Calculate cumulative values for waterfall
            cumulative = [base_value]
            for value in values:
                cumulative.append(cumulative[-1] + value)
            
            # Create waterfall chart
            fig = go.Figure()
            
            # Base value
            fig.add_trace(go.Bar(
                name='Base Value',
                x=['Base'],
                y=[base_value],
                marker_color='gray',
                text=[f'{base_value:.3f}'],
                textposition='auto'
            ))
            
            # Feature contributions
            for i, (feature, value) in enumerate(zip(features, values)):
                color = 'red' if value < 0 else 'blue'
                fig.add_trace(go.Bar(
                    name=feature,
                    x=[feature],
                    y=[value],
                    base=[cumulative[i]],
                    marker_color=color,
                    text=[f'{value:.3f}'],
                    textposition='auto',
                    showlegend=False
                ))
            
            # Final prediction
            fig.add_trace(go.Bar(
                name='Prediction',
                x=['Prediction'],
                y=[cumulative[-1]],
                marker_color='green',
                text=[f'{cumulative[-1]:.3f}'],
                textposition='auto'
            ))
            
            fig.update_layout(
                title='SHAP Waterfall Plot - Feature Contributions',
                xaxis_title='Features',
                yaxis_title='Contribution Value',
                xaxis_tickangle=-45,
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Plotly SHAP waterfall: {e}")
            return None
    
    def _create_matplotlib_shap_waterfall(self, feature_importance: List[Dict[str, Any]], base_value: float):
        """Create Matplotlib SHAP waterfall chart."""
        try:
            features = ['Base'] + [f['feature'] for f in feature_importance] + ['Prediction']
            values = [base_value] + [f['shap_value'] for f in feature_importance]
            
            # Calculate positions and colors
            cumulative = [base_value]
            colors = ['gray']
            
            for value in values[1:]:
                cumulative.append(cumulative[-1] + value)
                colors.append('red' if value < 0 else 'blue')
            
            colors.append('green')  # Final prediction
            
            fig, ax = plt.subplots(figsize=(max(10, len(features) * 0.8), 6), dpi=self.dpi)
            
            # Create waterfall bars
            for i, (feature, value, color) in enumerate(zip(features[:-1], values, colors[:-1])):
                if i == 0:  # Base value
                    ax.bar(i, value, color=color, alpha=0.7)
                    ax.text(i, value/2, f'{value:.3f}', ha='center', va='center', weight='bold')
                else:  # Feature contributions
                    bottom = cumulative[i-1] if value > 0 else cumulative[i]
                    height = abs(value)
                    ax.bar(i, height, bottom=bottom, color=color, alpha=0.7)
                    ax.text(i, bottom + height/2, f'{value:.3f}', ha='center', va='center', weight='bold')
            
            # Final prediction bar
            final_value = cumulative[-1]
            ax.bar(len(features)-1, final_value, color=colors[-1], alpha=0.7)
            ax.text(len(features)-1, final_value/2, f'{final_value:.3f}', ha='center', va='center', weight='bold')
            
            # Formatting
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.set_ylabel('Contribution Value')
            ax.set_title('SHAP Waterfall Plot - Feature Contributions')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            red_patch = mpatches.Patch(color='red', alpha=0.7, label='Negative Contribution')
            blue_patch = mpatches.Patch(color='blue', alpha=0.7, label='Positive Contribution')
            gray_patch = mpatches.Patch(color='gray', alpha=0.7, label='Base Value')
            green_patch = mpatches.Patch(color='green', alpha=0.7, label='Final Prediction')
            ax.legend(handles=[red_patch, blue_patch, gray_patch, green_patch])
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Matplotlib SHAP waterfall: {e}")
            return None
    
    def _create_plotly_uncertainty_plot(self, uncertainty_info: Dict[str, Any]):
        """Create Plotly uncertainty visualization."""
        try:
            # Create subplot with multiple uncertainty metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Confidence vs Uncertainty', 'Entropy', 'Prediction Distribution', 'Uncertainty Metrics'),
                specs=[[{"type": "scatter"}, {"type": "indicator"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            confidence = uncertainty_info.get('confidence', 0)
            uncertainty = uncertainty_info.get('uncertainty', 0)
            entropy = uncertainty_info.get('entropy', 0)
            distribution = uncertainty_info.get('prediction_distribution', [])
            
            # Confidence vs Uncertainty scatter
            fig.add_trace(
                go.Scatter(x=[confidence], y=[uncertainty], mode='markers', 
                          marker=dict(size=20, color='red'), name='Current Prediction'),
                row=1, col=1
            )
            
            # Entropy gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=entropy,
                    title={'text': "Entropy"},
                    gauge={'axis': {'range': [None, 2]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                   {'range': [0.5, 1], 'color': "yellow"},
                                   {'range': [1, 2], 'color': "red"}]}
                ),
                row=1, col=2
            )
            
            # Prediction distribution
            if distribution:
                fig.add_trace(
                    go.Bar(x=[f'Class {i}' for i in range(len(distribution))], 
                          y=distribution, name='Probabilities'),
                    row=2, col=1
                )
            
            # Uncertainty metrics
            metrics = ['Confidence', 'Uncertainty', 'Entropy']
            values = [confidence, uncertainty, entropy]
            fig.add_trace(
                go.Bar(x=metrics, y=values, name='Metrics'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False, title_text="Prediction Uncertainty Analysis")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Plotly uncertainty plot: {e}")
            return None
    
    def _create_matplotlib_uncertainty_plot(self, uncertainty_info: Dict[str, Any]):
        """Create Matplotlib uncertainty visualization."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)
            
            confidence = uncertainty_info.get('confidence', 0)
            uncertainty = uncertainty_info.get('uncertainty', 0)
            entropy = uncertainty_info.get('entropy', 0)
            distribution = uncertainty_info.get('prediction_distribution', [])
            
            # Confidence vs Uncertainty
            ax1.scatter([confidence], [uncertainty], s=100, c='red', alpha=0.7)
            ax1.set_xlabel('Confidence')
            ax1.set_ylabel('Uncertainty')
            ax1.set_title('Confidence vs Uncertainty')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            
            # Entropy visualization
            ax2.bar(['Entropy'], [entropy], color='orange', alpha=0.7)
            ax2.set_ylabel('Entropy Value')
            ax2.set_title('Prediction Entropy')
            ax2.set_ylim(0, 2)
            ax2.text(0, entropy + 0.05, f'{entropy:.3f}', ha='center', va='bottom', weight='bold')
            
            # Prediction distribution
            if distribution:
                classes = [f'Class {i}' for i in range(len(distribution))]
                bars = ax3.bar(classes, distribution, alpha=0.7)
                ax3.set_ylabel('Probability')
                ax3.set_title('Prediction Distribution')
                ax3.set_ylim(0, 1)
                
                # Highlight max probability
                max_idx = np.argmax(distribution)
                bars[max_idx].set_color('red')
                
                for bar, prob in zip(bars, distribution):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{prob:.3f}', ha='center', va='bottom')
            
            # Uncertainty metrics summary
            metrics = ['Confidence', 'Uncertainty', 'Entropy']
            values = [confidence, uncertainty, entropy]
            colors = ['green', 'red', 'orange']
            bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
            ax4.set_ylabel('Value')
            ax4.set_title('Uncertainty Metrics')
            
            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', weight='bold')
            
            plt.suptitle('Prediction Uncertainty Analysis', fontsize=16, weight='bold')
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Matplotlib uncertainty plot: {e}")
            return None
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=self.dpi)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}")
            return ""
    
    def _fig_to_svg(self, fig) -> str:
        """Convert matplotlib figure to SVG string."""
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='svg', bbox_inches='tight')
            buffer.seek(0)
            svg_string = buffer.getvalue().decode('utf-8')
            buffer.close()
            return svg_string
        except Exception as e:
            logger.error(f"Error converting figure to SVG: {e}")
            return ""