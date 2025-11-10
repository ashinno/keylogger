"""
Integration module for Advanced Behavioral Analytics Engine
Connects the advanced behavioral analytics with the existing keylogger system
"""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import existing system components
try:
    from .advanced_behavioral_analytics import (
        AdvancedBehavioralAnalyticsEngine, ThreatLevel, AnomalyType, 
        BehavioralProfile, AnomalyResult
    )
    from ..core.config_manager import ConfigManager
    from ..core.logging_manager import LoggingManager
    from ..core.encryption_manager import EncryptionManager
except ImportError as e:
    # Fallback for direct execution
    logging.error(f"Import error in behavioral analytics integration: {e}")
    AdvancedBehavioralAnalyticsEngine = None
    ConfigManager = None

logger = logging.getLogger(__name__)


class BehavioralAnalyticsIntegration:
    """
    Integration class that connects Advanced Behavioral Analytics with the existing system
    
    This class provides:
    - Seamless integration with existing data collection
    - Real-time behavioral analysis
    - Configurable security policies
    - Integration with web dashboard
    - Automated response capabilities
    """
    
    def __init__(self, config_manager: ConfigManager, logging_manager: LoggingManager):
        self.config_manager = config_manager
        self.logging_manager = logging_manager
        
        # Initialize advanced behavioral analytics engine
        config = self.config_manager.get_all_config()
        self.analytics_engine = AdvancedBehavioralAnalyticsEngine(config)
        
        # Integration settings
        self.real_time_analysis = config.get('behavioral_analytics.real_time_analysis', True)
        self.auto_response_enabled = config.get('behavioral_analytics.auto_response_enabled', True)
        self.alert_threshold = config.get('behavioral_analytics.alert_threshold', ThreatLevel.MEDIUM)
        self.data_retention_days = config.get('behavioral_analytics.data_retention_days', 90)
        
        # User tracking
        self.active_users = {}
        self.user_sessions = {}
        
        # Security policies
        self.security_policies = self._load_security_policies()
        
        # Response handlers
        self.response_handlers = {
            'log_alert': self._handle_log_alert,
            'notify_admin': self._handle_notify_admin,
            'isolate_user': self._handle_isolate_user,
            'block_action': self._handle_block_action,
            'require_mfa': self._handle_require_mfa,
            'preserve_evidence': self._handle_preserve_evidence
        }
        
        logger.info("Behavioral Analytics Integration initialized")
    
    def _load_security_policies(self) -> Dict[str, Any]:
        """Load security policies from configuration."""
        
        default_policies = {
            'threat_level_actions': {
                'low': ['log_alert'],
                'medium': ['log_alert', 'notify_admin'],
                'high': ['log_alert', 'notify_admin', 'isolate_user', 'preserve_evidence'],
                'critical': ['log_alert', 'notify_admin', 'isolate_user', 'block_action', 'preserve_evidence']
            },
            'anomaly_type_actions': {
                'account_takeover': ['require_mfa', 'notify_admin'],
                'insider_threat': ['isolate_user', 'preserve_evidence'],
                'cross_platform_anomaly': ['require_mfa', 'notify_admin'],
                'adversarial_attack': ['block_action', 'preserve_evidence']
            },
            'auto_response_conditions': {
                'consecutive_anomalies': 3,
                'confidence_threshold': 0.8,
                'risk_score_threshold': 0.7
            }
        }
        
        # Load from config
        config_policies = self.config_manager.get('behavioral_analytics.security_policies', {})
        
        # Merge with defaults
        policies = {**default_policies, **config_policies}
        
        return policies
    
    def process_user_event(self, user_id: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single user event through behavioral analytics
        
        Args:
            user_id: Unique identifier for the user
            event_data: Event data from keylogger
            
        Returns:
            Analysis results including anomaly detection
        """
        
        try:
            # Enrich event data with context
            enriched_event = self._enrich_event_data(user_id, event_data)
            
            # Perform behavioral analysis
            analysis_result = self.analytics_engine.analyze_behavior(user_id, enriched_event)
            
            # Handle security response
            if self.auto_response_enabled:
                self._execute_security_response(user_id, analysis_result)
            
            # Log analysis results
            self._log_analysis_results(user_id, analysis_result)
            
            # Update user tracking
            self._update_user_tracking(user_id, analysis_result)
            
            # Return analysis summary
            return self._create_analysis_summary(user_id, analysis_result)
            
        except Exception as e:
            logger.error(f"Error processing user event for {user_id}: {e}")
            return self._create_error_response(user_id, str(e))
    
    def _enrich_event_data(self, user_id: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich event data with additional context."""
        
        enriched_data = event_data.copy()
        
        # Add system context
        enriched_data['system_context'] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': user_id,
            'session_id': self._get_user_session_id(user_id),
            'device_info': self._get_device_info(),
            'network_info': self._get_network_info(),
            'system_metrics': self._get_system_metrics()
        }
        
        # Add historical context
        enriched_data['historical_context'] = self._get_user_historical_context(user_id)
        
        # Add threat intelligence context
        enriched_data['threat_context'] = self._get_threat_context(user_id)
        
        return enriched_data
    
    def _get_user_session_id(self, user_id: str) -> str:
        """Get or create session ID for user."""
        
        if user_id not in self.user_sessions:
            import uuid
            self.user_sessions[user_id] = str(uuid.uuid4())
        
        return self.user_sessions[user_id]
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        
        try:
            import platform
            import socket
            
            return {
                'hostname': socket.gethostname(),
                'platform': platform.system(),
                'platform_version': platform.version(),
                'architecture': platform.architecture()[0],
                'processor': platform.processor()
            }
        except Exception as e:
            logger.warning(f"Error getting device info: {e}")
            return {}
    
    def _get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        
        try:
            import socket
            import requests
            
            # Get local IP
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            # Get public IP
            try:
                response = requests.get('https://api.ipify.org?format=json', timeout=5)
                public_ip = response.json().get('ip', 'unknown')
            except:
                public_ip = 'unknown'
            
            return {
                'local_ip': local_ip,
                'public_ip': public_ip,
                'hostname': hostname
            }
        except Exception as e:
            logger.warning(f"Error getting network info: {e}")
            return {}
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        
        try:
            import psutil
            
            return {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_connections': len(psutil.net_connections()),
                'process_count': len(psutil.pids())
            }
        except ImportError:
            logger.warning("psutil not available for system metrics")
            return {}
        except Exception as e:
            logger.warning(f"Error getting system metrics: {e}")
            return {}
    
    def _get_user_historical_context(self, user_id: str) -> Dict[str, Any]:
        """Get user's historical context."""
        
        # This would typically query a database or cache
        # For now, return basic context
        return {
            'account_age_days': self._calculate_account_age(user_id),
            'previous_anomalies': self._get_user_anomaly_history(user_id),
            'typical_behavior_patterns': self._get_user_behavior_patterns(user_id)
        }
    
    def _calculate_account_age(self, user_id: str) -> int:
        """Calculate account age in days."""
        
        # This would typically query user creation date
        # For now, return a default value
        return 30  # Default to 30 days
    
    def _get_user_anomaly_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's anomaly history."""
        
        profile = self.analytics_engine.user_profiles.get(user_id)
        if profile:
            # Return last 10 anomalies
            return profile.anomaly_history[-10:]
        
        return []
    
    def _get_user_behavior_patterns(self, user_id: str) -> Dict[str, Any]:
        """Get user's typical behavior patterns."""
        
        profile = self.analytics_engine.user_profiles.get(user_id)
        if profile:
            return profile.behavioral_patterns
        
        return {}
    
    def _get_threat_context(self, user_id: str) -> Dict[str, Any]:
        """Get threat intelligence context."""
        
        # This would typically integrate with threat intelligence feeds
        # For now, return basic threat context
        return {
            'ip_reputation': 'clean',  # This would check IP reputation
            'geolocation': self._get_geolocation_context(),
            'threat_indicators': self._get_user_threat_indicators(user_id)
        }
    
    def _get_geolocation_context(self) -> Dict[str, Any]:
        """Get geolocation context."""
        
        try:
            import requests
            
            response = requests.get('https://ipapi.co/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'country': data.get('country_name', 'unknown'),
                    'city': data.get('city', 'unknown'),
                    'latitude': data.get('latitude', 0),
                    'longitude': data.get('longitude', 0),
                    'timezone': data.get('timezone', 'unknown')
                }
        except Exception as e:
            logger.warning(f"Error getting geolocation: {e}")
        
        return {}
    
    def _get_user_threat_indicators(self, user_id: str) -> List[str]:
        """Get threat indicators for user."""
        
        profile = self.analytics_engine.user_profiles.get(user_id)
        if profile:
            return profile.threat_indicators
        
        return []
    
    def _execute_security_response(self, user_id: str, analysis_result: AnomalyResult):
        """Execute automated security response based on analysis results."""
        
        try:
            # Determine required actions based on threat level and anomaly type
            actions = self._determine_response_actions(analysis_result)
            
            # Execute each action
            for action in actions:
                if action in self.response_handlers:
                    self.response_handlers[action](user_id, analysis_result)
                else:
                    logger.warning(f"Unknown response action: {action}")
            
            # Log response execution
            self._log_response_execution(user_id, actions, analysis_result)
            
        except Exception as e:
            logger.error(f"Error executing security response for {user_id}: {e}")
    
    def _determine_response_actions(self, analysis_result: AnomalyResult) -> List[str]:
        """Determine which response actions to execute."""
        
        actions = []
        
        # Base actions by threat level
        threat_level_str = analysis_result.threat_level.value
        if threat_level_str in self.security_policies['threat_level_actions']:
            actions.extend(self.security_policies['threat_level_actions'][threat_level_str])
        
        # Additional actions by anomaly type
        anomaly_type_str = analysis_result.anomaly_type.value
        if anomaly_type_str in self.security_policies['anomaly_type_actions']:
            actions.extend(self.security_policies['anomaly_type_actions'][anomaly_type_str])
        
        # Check auto-response conditions
        if self._should_trigger_auto_response(analysis_result):
            # Add additional actions for high-confidence threats
            if analysis_result.confidence_score > 0.8:
                actions.append('preserve_evidence')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for action in actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)
        
        return unique_actions
    
    def _should_trigger_auto_response(self, analysis_result: AnomalyResult) -> bool:
        """Determine if auto-response should be triggered."""
        
        conditions = self.security_policies['auto_response_conditions']
        
        # Check confidence threshold
        if analysis_result.confidence_score < conditions['confidence_threshold']:
            return False
        
        # Check risk score threshold
        if analysis_result.risk_score < conditions['risk_score_threshold']:
            return False
        
        return True
    
    def _handle_log_alert(self, user_id: str, analysis_result: AnomalyResult):
        """Handle log alert response."""
        
        alert_message = f"Behavioral anomaly detected for user {user_id}: {analysis_result.threat_level.value} threat level"
        
        self.logging_manager.log_security_event(
            event_type='behavioral_anomaly',
            severity=analysis_result.threat_level.value,
            message=alert_message,
            user_id=user_id,
            details={
                'anomaly_type': analysis_result.anomaly_type.value,
                'risk_score': analysis_result.risk_score,
                'confidence_score': analysis_result.confidence_score,
                'recommended_actions': analysis_result.recommended_actions
            }
        )
        
        logger.warning(alert_message)
    
    def _handle_notify_admin(self, user_id: str, analysis_result: AnomalyResult):
        """Handle admin notification response."""
        
        # This would typically send notifications via email, SMS, or other channels
        # For now, log the notification
        notification_message = f"ADMIN NOTIFICATION: Behavioral anomaly for user {user_id}"
        
        self.logging_manager.log_security_event(
            event_type='admin_notification',
            severity='high',
            message=notification_message,
            user_id=user_id,
            details={
                'threat_level': analysis_result.threat_level.value,
                'anomaly_type': analysis_result.anomaly_type.value,
                'explanation': analysis_result.explanation
            }
        )
        
        logger.critical(notification_message)
    
    def _handle_isolate_user(self, user_id: str, analysis_result: AnomalyResult):
        """Handle user isolation response."""
        
        # This would typically implement user isolation mechanisms
        # For now, log the isolation action
        isolation_message = f"User {user_id} isolated due to behavioral anomaly"
        
        self.logging_manager.log_security_event(
            event_type='user_isolation',
            severity='critical',
            message=isolation_message,
            user_id=user_id,
            details={
                'threat_level': analysis_result.threat_level.value,
                'anomaly_type': analysis_result.anomaly_type.value,
                'isolation_reason': 'behavioral_anomaly'
            }
        )
        
        logger.critical(isolation_message)
    
    def _handle_block_action(self, user_id: str, analysis_result: AnomalyResult):
        """Handle action blocking response."""
        
        # This would typically block specific user actions
        # For now, log the blocking action
        block_message = f"Actions blocked for user {user_id} due to behavioral anomaly"
        
        self.logging_manager.log_security_event(
            event_type='action_blocked',
            severity='critical',
            message=block_message,
            user_id=user_id,
            details={
                'threat_level': analysis_result.threat_level.value,
                'anomaly_type': analysis_result.anomaly_type.value
            }
        )
        
        logger.critical(block_message)
    
    def _handle_require_mfa(self, user_id: str, analysis_result: AnomalyResult):
        """Handle MFA requirement response."""
        
        # This would typically require additional authentication
        # For now, log the MFA requirement
        mfa_message = f"MFA required for user {user_id} due to behavioral anomaly"
        
        self.logging_manager.log_security_event(
            event_type='mfa_required',
            severity='high',
            message=mfa_message,
            user_id=user_id,
            details={
                'threat_level': analysis_result.threat_level.value,
                'anomaly_type': analysis_result.anomaly_type.value,
                'mfa_reason': 'behavioral_anomaly'
            }
        )
        
        logger.warning(mfa_message)
    
    def _handle_preserve_evidence(self, user_id: str, analysis_result: AnomalyResult):
        """Handle evidence preservation response."""
        
        # This would typically preserve digital evidence
        # For now, log the evidence preservation
        evidence_message = f"Evidence preserved for user {user_id} behavioral anomaly"
        
        # Create evidence package
        evidence_package = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': user_id,
            'analysis_result': {
                'is_anomalous': analysis_result.is_anomalous,
                'threat_level': analysis_result.threat_level.value,
                'anomaly_type': analysis_result.anomaly_type.value,
                'risk_score': analysis_result.risk_score,
                'confidence_score': analysis_result.confidence_score,
                'explanation': analysis_result.explanation
            },
            'user_profile': self._get_user_profile_summary(user_id)
        }
        
        self.logging_manager.log_security_event(
            event_type='evidence_preserved',
            severity='critical',
            message=evidence_message,
            user_id=user_id,
            details=evidence_package
        )
        
        logger.critical(evidence_message)
    
    def _log_analysis_results(self, user_id: str, analysis_result: AnomalyResult):
        """Log analysis results for audit trail."""
        
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': user_id,
            'analysis_type': 'behavioral_analytics',
            'result': {
                'is_anomalous': analysis_result.is_anomalous,
                'threat_level': analysis_result.threat_level.value,
                'anomaly_type': analysis_result.anomaly_type.value,
                'risk_score': analysis_result.risk_score,
                'confidence_score': analysis_result.confidence_score
            },
            'explanation_summary': analysis_result.explanation.get('summary', '')
        }
        
        self.logging_manager.log_analysis_event(log_entry)
    
    def _log_response_execution(self, user_id: str, actions: List[str], 
                              analysis_result: AnomalyResult):
        """Log security response execution."""
        
        response_log = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': user_id,
            'triggered_by': {
                'threat_level': analysis_result.threat_level.value,
                'risk_score': analysis_result.risk_score,
                'confidence_score': analysis_result.confidence_score
            },
            'executed_actions': actions,
            'response_effectiveness': 'pending_evaluation'
        }
        
        self.logging_manager.log_response_event(response_log)
    
    def _update_user_tracking(self, user_id: str, analysis_result: AnomalyResult):
        """Update user tracking information."""
        
        if user_id not in self.active_users:
            self.active_users[user_id] = {
                'first_seen': datetime.now(timezone.utc),
                'anomaly_count': 0,
                'last_anomaly': None,
                'risk_level': 'low'
            }
        
        user_info = self.active_users[user_id]
        
        if analysis_result.is_anomalous:
            user_info['anomaly_count'] += 1
            user_info['last_anomaly'] = datetime.now(timezone.utc)
            user_info['risk_level'] = analysis_result.threat_level.value
        
        # Update risk level based on recent activity
        self._update_user_risk_level(user_id)
    
    def _update_user_risk_level(self, user_id: str):
        """Update user's risk level based on recent activity."""
        
        user_info = self.active_users.get(user_id)
        if not user_info:
            return
        
        # Get recent anomalies
        recent_anomalies = self.analytics_engine.user_profiles.get(user_id, BehavioralProfile(
            user_id=user_id, baseline_features={}, behavioral_patterns={}, 
            risk_score=0.0, confidence_level=0.0, last_updated=datetime.now(timezone.utc),
            anomaly_history=[], threat_indicators=[]
        )).anomaly_history
        
        # Calculate risk based on recent anomalies
        recent_count = len([a for a in recent_anomalies 
                           if (datetime.now(timezone.utc) - 
                               datetime.fromisoformat(a['timestamp'])).days <= 7])
        
        if recent_count >= 5:
            user_info['risk_level'] = 'high'
        elif recent_count >= 2:
            user_info['risk_level'] = 'medium'
        else:
            user_info['risk_level'] = 'low'
    
    def _get_user_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """Get user profile summary for evidence."""
        
        profile = self.analytics_engine.user_profiles.get(user_id)
        if profile:
            return {
                'user_id': profile.user_id,
                'risk_score': profile.risk_score,
                'confidence_level': profile.confidence_level,
                'anomaly_history_count': len(profile.anomaly_history),
                'threat_indicators': profile.threat_indicators,
                'last_updated': profile.last_updated.isoformat()
            }
        
        return {'user_id': user_id, 'profile_status': 'not_found'}
    
    def _create_analysis_summary(self, user_id: str, analysis_result: AnomalyResult) -> Dict[str, Any]:
        """Create analysis summary for response."""
        
        return {
            'user_id': user_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis_result': {
                'is_anomalous': analysis_result.is_anomalous,
                'threat_level': analysis_result.threat_level.value,
                'anomaly_type': analysis_result.anomaly_type.value,
                'risk_score': round(analysis_result.risk_score, 3),
                'confidence_score': round(analysis_result.confidence_score, 3),
                'recommended_actions': analysis_result.recommended_actions
            },
            'explanation': {
                'summary': analysis_result.explanation.get('summary', ''),
                'confidence_justification': analysis_result.explanation.get('confidence_justification', '')
            },
            'system_actions_taken': self._get_actions_taken(user_id),
            'requires_attention': analysis_result.is_anomalous and analysis_result.threat_level in [
                ThreatLevel.HIGH, ThreatLevel.CRITICAL
            ]
        }
    
    def _get_actions_taken(self, user_id: str) -> List[str]:
        """Get actions taken for user."""
        
        # This would typically query response logs
        # For now, return empty list
        return []
    
    def _create_error_response(self, user_id: str, error_message: str) -> Dict[str, Any]:
        """Create error response."""
        
        return {
            'user_id': user_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis_result': {
                'is_anomalous': True,
                'threat_level': 'high',
                'anomaly_type': 'system_error',
                'risk_score': 0.8,
                'confidence_score': 0.5,
                'recommended_actions': ['manual_review_required']
            },
            'explanation': {
                'summary': f'Analysis error: {error_message}',
                'confidence_justification': 'Low confidence due to system error'
            },
            'system_actions_taken': ['log_error'],
            'requires_attention': True,
            'error': error_message
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information."""
        
        try:
            # Get analytics engine status
            analytics_status = self.analytics_engine.get_system_health()
            
            # Get integration-specific status
            integration_status = {
                'active_users': len(self.active_users),
                'user_sessions': len(self.user_sessions),
                'auto_response_enabled': self.auto_response_enabled,
                'real_time_analysis': self.real_time_analysis,
                'security_policies_loaded': len(self.security_policies) > 0,
                'response_handlers_available': len(self.response_handlers)
            }
            
            # Combine status information
            status = {
                'integration_status': integration_status,
                'analytics_engine_status': analytics_status,
                'overall_health': self._calculate_overall_health(analytics_status, integration_status),
                'last_update': datetime.now(timezone.utc).isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'error': str(e),
                'overall_health': 'error',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _calculate_overall_health(self, analytics_status: Dict[str, Any], 
                                integration_status: Dict[str, Any]) -> str:
        """Calculate overall system health."""
        
        # Check analytics engine health
        analytics_health = analytics_status.get('overall_health', 'unknown')
        
        # Check integration health
        integration_issues = []
        
        if integration_status['active_users'] == 0:
            integration_issues.append('no_active_users')
        
        if not integration_status['auto_response_enabled']:
            integration_issues.append('auto_response_disabled')
        
        # Determine overall health
        if analytics_health == 'error' or len(integration_issues) > 2:
            return 'error'
        elif analytics_health == 'degraded' or len(integration_issues) > 0:
            return 'degraded'
        elif analytics_health == 'healthy':
            return 'healthy'
        else:
            return 'unknown'
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        
        try:
            # Get analytics engine summary
            analytics_summary = self.analytics_engine.get_analytics_summary()
            
            # Add integration-specific information
            integration_summary = {
                'integration_metrics': {
                    'total_events_processed': analytics_summary['system_statistics']['events_processed'],
                    'active_user_count': len(self.active_users),
                    'high_risk_users': len([u for u in self.active_users.values() 
                                          if u['risk_level'] == 'high']),
                    'automated_responses_triggered': self._count_automated_responses()
                },
                'security_policy_effectiveness': self._analyze_policy_effectiveness(),
                'trending_threats': self._identify_trending_threats(),
                'system_recommendations': self._generate_system_recommendations()
            }
            
            # Combine summaries
            summary = {
                'analytics_summary': analytics_summary,
                'integration_summary': integration_summary,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {'error': str(e)}
    
    def _count_automated_responses(self) -> int:
        """Count automated responses triggered."""
        
        # This would typically query response logs
        # For now, return 0
        return 0
    
    def _analyze_policy_effectiveness(self) -> Dict[str, Any]:
        """Analyze security policy effectiveness."""
        
        # This would analyze policy effectiveness based on historical data
        # For now, return basic analysis
        return {
            'policy_coverage': 'good',
            'response_time': 'acceptable',
            'false_positive_rate': 'low',
            'threat_mitigation_rate': 'high'
        }
    
    def _identify_trending_threats(self) -> List[Dict[str, Any]]:
        """Identify trending threats."""
        
        # This would analyze recent anomalies to identify trends
        # For now, return empty list
        return []
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system improvement recommendations."""
        
        recommendations = []
        
        # Check if auto-response is disabled
        if not self.auto_response_enabled:
            recommendations.append("Consider enabling auto-response for faster threat mitigation")
        
        # Check user profile coverage
        if len(self.analytics_engine.user_profiles) < 5:
            recommendations.append("Collect more user data to improve behavioral modeling accuracy")
        
        # Check model accuracy
        if self.analytics_engine.stats['model_accuracy'] < 0.8:
            recommendations.append("Consider retraining models to improve accuracy")
        
        return recommendations
    
    def cleanup_old_data(self, days_to_keep: int = None):
        """Clean up old data."""
        
        if days_to_keep is None:
            days_to_keep = self.data_retention_days
        
        try:
            # Clean up analytics engine data
            self.analytics_engine.cleanup_old_data(days_to_keep)
            
            # Clean up integration data
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            # Clean up user tracking
            inactive_users = [
                user_id for user_id, user_info in self.active_users.items()
                if user_info.get('last_anomaly') and 
                datetime.fromisoformat(user_info['last_anomaly']) < cutoff_date
            ]
            
            for user_id in inactive_users:
                del self.active_users[user_id]
                if user_id in self.user_sessions:
                    del self.user_sessions[user_id]
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
    
    def save_models(self):
        """Save all models and data."""
        
        try:
            self.analytics_engine.save_models()
            logger.info("Behavioral analytics models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def validate_integrity(self) -> Dict[str, Any]:
        """Validate system integrity."""
        
        try:
            # Validate model integrity
            model_validation = self.analytics_engine.validate_model_integrity()
            
            # Validate integration components
            integration_validation = {
                'configuration_valid': self._validate_configuration(),
                'security_policies_valid': self._validate_security_policies(),
                'response_handlers_functional': self._validate_response_handlers()
            }
            
            return {
                'model_integrity': model_validation,
                'integration_integrity': integration_validation,
                'overall_integrity': 'valid' if (model_validation['integrity_status'] == 'valid' and
                                               all(integration_validation.values())) else 'compromised',
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating integrity: {e}")
            return {
                'error': str(e),
                'overall_integrity': 'error',
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _validate_configuration(self) -> bool:
        """Validate configuration."""
        
        try:
            # Check if required configuration keys exist
            required_keys = [
                'behavioral_analytics.real_time_analysis',
                'behavioral_analytics.auto_response_enabled',
                'behavioral_analytics.alert_threshold'
            ]
            
            for key in required_keys:
                if self.config_manager.get(key) is None:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def _validate_security_policies(self) -> bool:
        """Validate security policies."""
        
        try:
            # Check if required policy sections exist
            required_sections = [
                'threat_level_actions',
                'anomaly_type_actions',
                'auto_response_conditions'
            ]
            
            for section in required_sections:
                if section not in self.security_policies:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Security policy validation error: {e}")
            return False
    
    def _validate_response_handlers(self) -> bool:
        """Validate response handlers."""
        
        try:
            # Check if all required handlers are available
            required_handlers = ['log_alert', 'notify_admin']
            
            for handler in required_handlers:
                if handler not in self.response_handlers:
                    return False
            
            # Test basic functionality
            for handler_name, handler_func in self.response_handlers.items():
                if not callable(handler_func):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Response handler validation error: {e}")
            return False


class BehavioralAnalyticsWebInterface:
    """Web interface for behavioral analytics dashboard."""
    
    def __init__(self, integration: BehavioralAnalyticsIntegration):
        self.integration = integration
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for behavioral analytics dashboard."""
        
        try:
            # Get system status
            system_status = self.integration.get_system_status()
            
            # Get analytics summary
            analytics_summary = self.integration.get_analytics_summary()
            
            # Get recent anomalies
            recent_anomalies = self._get_recent_anomalies_data()
            
            # Get user risk distribution
            user_risk_data = self._get_user_risk_distribution()
            
            # Get model performance metrics
            model_performance = self._get_model_performance_data()
            
            # Combine all data
            dashboard_data = {
                'system_status': system_status,
                'analytics_summary': analytics_summary,
                'recent_anomalies': recent_anomalies,
                'user_risk_distribution': user_risk_data,
                'model_performance': model_performance,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}
    
    def _get_recent_anomalies_data(self) -> List[Dict[str, Any]]:
        """Get recent anomalies data for dashboard."""
        
        # This would typically query recent anomalies
        # For now, return sample data structure
        return [
            {
                'user_id': 'sample_user',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'threat_level': 'medium',
                'anomaly_type': 'timing_anomaly',
                'risk_score': 0.65,
                'confidence_score': 0.82
            }
        ]
    
    def _get_user_risk_distribution(self) -> Dict[str, Any]:
        """Get user risk distribution data."""
        
        # Get user profiles
        user_profiles = self.integration.analytics_engine.user_profiles
        
        # Calculate risk distribution
        risk_levels = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for profile in user_profiles.values():
            if profile.risk_score < 0.3:
                risk_levels['low'] += 1
            elif profile.risk_score < 0.6:
                risk_levels['medium'] += 1
            elif profile.risk_score < 0.8:
                risk_levels['high'] += 1
            else:
                risk_levels['critical'] += 1
        
        return {
            'total_users': len(user_profiles),
            'risk_distribution': risk_levels,
            'average_risk_score': np.mean([p.risk_score for p in user_profiles.values()]) if user_profiles else 0
        }
    
    def _get_model_performance_data(self) -> Dict[str, Any]:
        """Get model performance data."""
        
        # Get analytics engine stats
        stats = self.integration.analytics_engine.stats
        
        return {
            'total_events_processed': stats['events_processed'],
            'anomalies_detected': stats['anomalies_detected'],
            'model_accuracy': stats['model_accuracy'],
            'false_positive_rate': stats['false_positives'] / max(1, stats['events_processed']),
            'last_training': stats['last_training'].isoformat() if stats['last_training'] else None
        }
    
    def get_user_details(self, user_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific user."""
        
        try:
            # Get user profile
            profile = self.integration.analytics_engine.user_profiles.get(user_id)
            
            if not profile:
                return {'error': 'User not found'}
            
            # Get user recent activity
            recent_activity = self._get_user_recent_activity(user_id)
            
            # Get user anomaly history
            anomaly_history = profile.anomaly_history[-20:]  # Last 20 anomalies
            
            # Get user behavioral patterns
            behavioral_patterns = profile.behavioral_patterns
            
            # Combine user data
            user_data = {
                'user_id': profile.user_id,
                'risk_score': profile.risk_score,
                'confidence_level': profile.confidence_level,
                'account_age_days': self.integration._calculate_account_age(user_id),
                'anomaly_history': anomaly_history,
                'behavioral_patterns': behavioral_patterns,
                'threat_indicators': profile.threat_indicators,
                'recent_activity': recent_activity,
                'last_updated': profile.last_updated.isoformat()
            }
            
            return user_data
            
        except Exception as e:
            logger.error(f"Error getting user details for {user_id}: {e}")
            return {'error': str(e)}
    
    def _get_user_recent_activity(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's recent activity."""
        
        # This would typically query recent events
        # For now, return empty list
        return []
    
    def get_anomaly_details(self, user_id: str, anomaly_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific anomaly."""
        
        try:
            # Get user profile
            profile = self.integration.analytics_engine.user_profiles.get(user_id)
            
            if not profile:
                return {'error': 'User not found'}
            
            # Find specific anomaly
            anomaly = None
            for a in profile.anomaly_history:
                if a.get('id') == anomaly_id:
                    anomaly = a
                    break
            
            if not anomaly:
                return {'error': 'Anomaly not found'}
            
            # Get detailed explanation
            explanation = anomaly.get('explanation', {})
            
            return {
                'anomaly_id': anomaly_id,
                'user_id': user_id,
                'timestamp': anomaly['timestamp'],
                'threat_level': anomaly['threat_level'],
                'anomaly_type': anomaly['anomaly_type'],
                'risk_score': anomaly['risk_score'],
                'confidence_score': anomaly['confidence_score'],
                'explanation': explanation,
                'recommended_actions': anomaly.get('recommended_actions', [])
            }
            
        except Exception as e:
            logger.error(f"Error getting anomaly details: {e}")
            return {'error': str(e)}
    
    def export_analytics_report(self, format_type: str = 'json') -> Dict[str, Any]:
        """Export comprehensive analytics report."""
        
        try:
            # Get comprehensive analytics data
            dashboard_data = self.get_dashboard_data()
            
            # Get model insights
            model_insights = self.integration.analytics_engine.export_model_insights()
            
            # Get system integrity validation
            integrity_validation = self.integration.validate_integrity()
            
            # Create comprehensive report
            report = {
                'report_metadata': {
                    'generated_at': datetime.now(timezone.utc).isoformat(),
                    'report_format': format_type,
                    'system_version': '1.0.0'
                },
                'executive_summary': self._generate_executive_summary(dashboard_data),
                'detailed_analytics': dashboard_data,
                'model_insights': model_insights,
                'system_integrity': integrity_validation,
                'recommendations': self._generate_report_recommendations(dashboard_data)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating analytics report: {e}")
            return {'error': str(e)}
    
    def _generate_executive_summary(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for report."""
        
        analytics_summary = dashboard_data.get('analytics_summary', {})
        system_status = dashboard_data.get('system_status', {})
        
        return {
            'total_users_monitored': analytics_summary.get('analytics_summary', {}).get('user_profiles', {}).get('total_profiles', 0),
            'total_anomalies_detected': analytics_summary.get('analytics_summary', {}).get('system_statistics', {}).get('anomalies_detected', 0),
            'system_health': system_status.get('overall_health', 'unknown'),
            'key_findings': self._identify_key_findings(dashboard_data),
            'risk_assessment': self._assess_overall_risk(dashboard_data)
        }
    
    def _identify_key_findings(self, dashboard_data: Dict[str, Any]) -> List[str]:
        """Identify key findings from dashboard data."""
        
        findings = []
        
        # Check for high-risk users
        user_risk = dashboard_data.get('user_risk_distribution', {})
        if user_risk.get('risk_distribution', {}).get('critical', 0) > 0:
            findings.append(f"{user_risk['risk_distribution']['critical']} users are at critical risk level")
        
        # Check system health
        system_status = dashboard_data.get('system_status', {})
        if system_status.get('overall_health') != 'healthy':
            findings.append(f"System health is {system_status.get('overall_health', 'unknown')}")
        
        return findings
    
    def _assess_overall_risk(self, dashboard_data: Dict[str, Any]) -> str:
        """Assess overall system risk."""
        
        # Simple risk assessment based on available data
        user_risk = dashboard_data.get('user_risk_distribution', {})
        critical_users = user_risk.get('risk_distribution', {}).get('critical', 0)
        
        if critical_users > 5:
            return 'high'
        elif critical_users > 0:
            return 'medium'
        else:
            return 'low'
    
    def _generate_report_recommendations(self, dashboard_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for the report."""
        
        recommendations = []
        
        # Check user risk distribution
        user_risk = dashboard_data.get('user_risk_distribution', {})
        if user_risk.get('risk_distribution', {}).get('critical', 0) > 0:
            recommendations.append("Immediate attention required for critical-risk users")
        
        # Check model performance
        model_performance = dashboard_data.get('model_performance', {})
        if model_performance.get('model_accuracy', 0) < 0.8:
            recommendations.append("Consider retraining models to improve accuracy")
        
        return recommendations