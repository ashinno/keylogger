# ML Anomaly Detection System

A comprehensive machine learning-based anomaly detection system integrated into the Enhanced Keylogger project.

## 🎯 Overview

This system provides real-time behavioral analysis and anomaly detection using advanced machine learning algorithms to identify potential security threats, insider threats, and unusual user behavior patterns.

## 🧠 Components

### 1. Behavioral Analytics Engine (`ml/behavioral_analytics.py`)
- **Supervised Learning**: Random Forest, SVM, Gradient Boosting
- **Unsupervised Learning**: Isolation Forest, One-Class SVM, DBSCAN
- **Features**: Establishes baseline user behavior patterns with configurable sensitivity
- **Adaptive Learning**: Automatically adjusts to user behavior evolution
- **Pattern Recognition**: Temporal and spatial behavior analysis

### 2. Keystroke Dynamics Analysis (`ml/keystroke_dynamics.py`)
- **Biometric Authentication**: Typing pattern recognition for user identification
- **Feature Extraction**: Keystroke timing, pressure, rhythm analysis
- **Advanced Models**: Random Forest, SVM, Neural Networks
- **User Enrollment**: Multi-user support with individual profiles
- **Adaptive Learning**: Accommodates typing pattern evolution over time

### 3. Insider Threat Detection (`ml/insider_threat.py`)
- **Multi-dimensional Analysis**: Access patterns, data usage, temporal behavior
- **Specialized Models**: Isolation Forest, One-Class SVM, DBSCAN, Random Forest
- **Correlation Analysis**: Cross-dimensional behavioral indicator analysis
- **Threat Categories**: Data exfiltration, privilege escalation, unusual access
- **Risk Indicators**: Comprehensive threat scoring and categorization

### 4. Real-time Risk Scoring (`ml/risk_scoring.py`)
- **Dynamic Scoring**: Weights different risk factors in real-time
- **Continuous Monitoring**: 24/7 user activity scoring
- **Automated Alerting**: Customizable risk thresholds and notifications
- **Ensemble Models**: Combines multiple ML algorithms for accurate scoring
- **Webhook Integration**: External system integration capabilities

### 5. Model Manager (`ml/model_manager.py`)
- **Centralized Management**: Training, persistence, and lifecycle management
- **Auto-retraining**: Automatic model updates based on new data
- **Performance Monitoring**: Model drift detection and performance tracking
- **Version Control**: Model versioning and rollback capabilities
- **Hyperparameter Tuning**: Automated optimization for better performance

### 6. Data Preprocessing (`ml/data_preprocessing.py`)
- **Feature Engineering**: Advanced feature extraction and transformation
- **Data Normalization**: Standardization and scaling for ML models
- **Pattern Detection**: Keystroke dynamics, behavioral patterns, anomalies
- **Real-time Processing**: Efficient data pipeline for live analysis

## 🔧 Configuration

All ML components are configurable through `config.json`:

```json
{
  "ml": {
    "enabled": true,
    "models_dir": "models",
    "behavioral_analytics": {
      "enabled": true,
      "sensitivity": 0.1,
      "learning_rate": 0.01,
      "baseline_window": 1000,
      "adaptation_threshold": 0.8
    },
    "keystroke_dynamics": {
      "enabled": true,
      "min_samples": 100,
      "auth_threshold": 0.8,
      "adaptation_rate": 0.1
    },
    "insider_threat": {
      "enabled": true,
      "threshold": 0.7,
      "baseline_window": 1000,
      "correlation_threshold": 0.8
    },
    "risk_scoring": {
      "enabled": true,
      "threshold": 0.8,
      "alert_threshold": 0.9,
      "decay_rate": 0.95
    }
  }
}
```

## 🌐 API Endpoints

Comprehensive REST API for integration with external systems:

### System Status
- `GET /api/ml/status` - Get ML system status
- `GET /api/ml/models/status` - Get model training status

### Behavioral Analytics
- `GET /api/ml/behavioral/baseline` - Get baseline summary
- `POST /api/ml/behavioral/reset` - Reset baseline

### Keystroke Dynamics
- `POST /api/ml/keystroke/enroll` - Enroll new user
- `POST /api/ml/keystroke/reset` - Reset session

### Insider Threat Detection
- `GET /api/ml/threat/summary` - Get threat summary
- `POST /api/ml/threat/reset` - Reset baseline

### Risk Scoring
- `GET /api/ml/risk/current` - Get current risk status
- `GET /api/ml/risk/alerts` - Get recent alerts
- `POST /api/ml/risk/callback` - Register webhook callback

### Configuration & Analysis
- `GET/POST /api/ml/config` - Get/update ML configuration
- `POST /api/ml/analytics/events` - Analyze event batch
- `POST /api/ml/export/data` - Export ML data

## 📊 Dashboard

Access the ML Dashboard at `/ml-dashboard` for:
- Real-time system status monitoring
- Risk score visualization
- Anomaly detection charts
- Alert management
- Component statistics
- Configuration management

## 🚀 Features

### Comprehensive Logging
- All ML events and anomalies are logged
- Structured logging with metadata
- Integration with existing logging system
- Audit trail for compliance

### Configurable Sensitivity
- Adjustable thresholds for each component
- Fine-tuned sensitivity parameters
- Real-time configuration updates
- Environment-specific settings

### Clear Visualization
- Interactive dashboard with real-time updates
- Risk trend charts and anomaly distribution
- Alert history and severity levels
- Component performance metrics

### API Integration
- RESTful API for external system integration
- Webhook support for real-time notifications
- Batch processing capabilities
- Data export functionality

## 🔒 Security Features

- **Encrypted Model Storage**: All models are encrypted at rest
- **Secure API Access**: Authentication required for all endpoints
- **Privacy Protection**: Sensitive data sanitization and hashing
- **Audit Logging**: Complete audit trail of all ML operations
- **Access Control**: Role-based access to ML functions

## 📈 Performance

- **Real-time Processing**: Sub-second event analysis
- **Scalable Architecture**: Handles high-volume event streams
- **Memory Efficient**: Optimized data structures and caching
- **Adaptive Learning**: Continuous improvement without manual intervention
- **Multi-threading**: Parallel processing for better performance

## 🛠 Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure ML Settings**:
   - Edit `config.json` to enable ML components
   - Adjust sensitivity thresholds as needed
   - Configure model storage directory

3. **Start the System**:
   ```bash
   python main.py
   ```

4. **Access Dashboard**:
   - Navigate to `http://localhost:5000/ml-dashboard`
   - Login with admin credentials
   - Monitor ML system status

## 🔧 Advanced Configuration

### Model Training
- Models automatically train on collected data
- Minimum sample requirements configurable
- Hyperparameter tuning available
- Cross-validation for model validation

### Alert Thresholds
- Behavioral anomaly: 0.1 (10% deviation)
- Keystroke authentication: 0.8 (80% confidence)
- Insider threat: 0.7 (70% threat score)
- Risk scoring: 0.8 (80% risk level)

### Performance Tuning
- Adjust window sizes for memory usage
- Configure cache sizes for performance
- Enable/disable components as needed
- Tune update intervals for responsiveness

## 📚 Technical Details

### Machine Learning Algorithms
- **Supervised**: Random Forest, SVM, Gradient Boosting, Neural Networks
- **Unsupervised**: Isolation Forest, One-Class SVM, DBSCAN, K-Means
- **Ensemble Methods**: Weighted voting, stacking, bagging
- **Feature Selection**: Statistical tests, recursive elimination

### Data Processing Pipeline
1. **Event Capture**: Real-time event collection
2. **Feature Extraction**: Multi-dimensional feature engineering
3. **Preprocessing**: Normalization, scaling, transformation
4. **Model Inference**: Real-time anomaly detection
5. **Post-processing**: Alert generation, logging, visualization

### Integration Points
- **Core Keylogger**: Seamless integration with existing system
- **Web Interface**: Dashboard and API endpoints
- **Configuration System**: Unified configuration management
- **Logging System**: Integrated audit and event logging
- **External Systems**: Webhook and API integration

## 🎯 Use Cases

1. **Security Monitoring**: Detect unauthorized access attempts
2. **Insider Threat Detection**: Identify malicious insider activities
3. **User Authentication**: Biometric keystroke-based authentication
4. **Behavioral Analysis**: Understand user behavior patterns
5. **Compliance**: Meet regulatory requirements for monitoring
6. **Forensics**: Detailed analysis of security incidents

## 🔮 Future Enhancements

- Deep learning models for advanced pattern recognition
- Federated learning for multi-system deployments
- Advanced visualization with 3D charts and heatmaps
- Integration with SIEM systems
- Mobile device support
- Cloud-based model training and deployment

---

**Note**: This ML anomaly detection system is designed for legitimate security monitoring and compliance purposes. Ensure proper authorization and compliance with applicable laws and regulations before deployment.