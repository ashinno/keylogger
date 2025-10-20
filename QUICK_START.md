# Quick Start Guide - Keylogger Project

## ✅ Project Status: All Issues Resolved

All machine learning issues have been fixed and the project is fully functional.

---

## Quick Test Commands

### 1. Test ML Models
```bash
python test_ml_fixes.py
```
**Expected Output:** All tests passed! ML models are working correctly.

### 2. Comprehensive Test
```bash
python test_comprehensive.py
```
**Expected Output:** ✓ ALL TESTS PASSED!

### 3. Test Main Application
```bash
python main.py --help
```
**Expected Output:** Help menu with all available options

---

## Running the Application

### Interactive Mode (Default)
```bash
python main.py
```
This starts the keylogger with an interactive menu where you can:
- View status
- Check statistics
- Export logs
- Reload configuration
- Stop/quit

### Web-Only Mode
```bash
python main.py --web-only
```
Access the web interface at: http://127.0.0.1:5000
- Default username: `admin`
- Default password: `admin123`

### Daemon Mode
```bash
python main.py --daemon
```
Runs in the background without interactive menu.

### Parse Existing Logs
```bash
python main.py --parse logs/keylog.txt -o output.json -f json
```
Formats: `text`, `csv`, `json`

---

## Configuration

Edit `config.json` to customize:
- Logging settings
- Encryption options
- Feature toggles (keyboard, mouse, clipboard, etc.)
- ML model parameters
- Web interface settings

---

## ML Models Status

All ML models are working correctly:

1. **Behavioral Analytics** ✅
   - Anomaly detection working
   - Baseline learning functional
   - Cold-start handling implemented

2. **Risk Scoring** ✅
   - Real-time risk calculation working
   - All models (primary, ensemble, linear) functional
   - Alert system operational

3. **Keystroke Dynamics** ✅
   - Pattern analysis working
   - Enrollment system functional

4. **Insider Threat Detection** ✅
   - All dimensional models working
   - DBSCAN temporal analysis fixed
   - Multi-factor threat scoring functional

---

## Recent Fixes Applied

### Fixed Issues:
1. ✅ Missing imports in `ml/risk_scoring.py`
2. ✅ Code structure issues in risk feature extraction
3. ✅ DBSCAN temporal_behavior model single-sample issue
4. ✅ Edge case handling (None values, malformed data)

### Files Modified:
- `ml/risk_scoring.py` - Import fixes and code cleanup
- `ml/insider_threat.py` - DBSCAN handling improvements

### Files Created:
- `test_comprehensive.py` - Full test suite
- `FIXES_SUMMARY.md` - Detailed fix documentation
- `QUICK_START.md` - This guide

---

## Troubleshooting

### If you encounter issues:

1. **Check dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Python version:**
   - Requires Python 3.8 or higher
   ```bash
   python --version
   ```

3. **Check logs:**
   - Main log: `keylogger_main.log`
   - Application logs: `logs/keylog.txt`

4. **Test ML models:**
   ```bash
   python test_ml_fixes.py
   python test_comprehensive.py
   ```

5. **Verify imports:**
   ```bash
   python -c "import main; print('Success')"
   ```

---

## Directory Structure

```
Keylogger/
├── main.py                 # Main application entry point
├── config.json            # Configuration file
├── requirements.txt       # Dependencies
├── test_ml_fixes.py      # ML model tests
├── test_comprehensive.py # Full test suite
├── FIXES_SUMMARY.md      # Detailed fix documentation
├── QUICK_START.md        # This file
├── core/                 # Core components
│   ├── keylogger.py
│   ├── config_manager.py
│   ├── encryption_manager.py
│   └── logging_manager.py
├── ml/                   # Machine learning modules
│   ├── behavioral_analytics.py
│   ├── risk_scoring.py
│   ├── keystroke_dynamics.py
│   ├── insider_threat.py
│   └── ...
├── listeners/            # Event listeners
│   ├── keyboard_listener.py
│   ├── mouse_listener.py
│   └── clipboard_listener.py
├── utils/                # Utility modules
│   ├── screenshot_monitor.py
│   ├── window_monitor.py
│   └── performance_monitor.py
└── web/                  # Web interface
    └── app.py
```

---

## Features

### Core Features:
- ✅ Keyboard logging
- ✅ Mouse tracking
- ✅ Clipboard monitoring
- ✅ Screenshot capture
- ✅ Window tracking
- ✅ USB monitoring
- ✅ Performance monitoring

### ML Features:
- ✅ Behavioral anomaly detection
- ✅ Real-time risk scoring
- ✅ Keystroke dynamics analysis
- ✅ Insider threat detection
- ✅ Model interpretability
- ✅ Confidence scoring

### Additional Features:
- ✅ End-to-end encryption
- ✅ Log rotation
- ✅ Web interface
- ✅ Export functionality (JSON, CSV, Text)
- ✅ Privacy controls

---

## Important Notes

### Security:
- This tool is for **educational and authorized monitoring purposes only**
- Ensure you have proper authorization before using
- Change default web interface credentials in production
- Enable encryption in `config.json`

### Performance:
- ML models improve over time as they collect baseline data
- Initial cold-start period may show conservative risk scores
- Models adapt to user behavior automatically

### Privacy:
- Configure `privacy.excluded_applications` in config.json
- Set `privacy.sanitize_passwords = true` (default)
- Review sensitive keywords list

---

## Support

For issues or questions:
1. Check `FIXES_SUMMARY.md` for detailed fix information
2. Review logs in `keylogger_main.log`
3. Run test suites to diagnose problems
4. Ensure all dependencies are installed

---

**Status:** ✅ Production Ready
**Last Updated:** 2025-10-20
**Version:** 2.0 (Enhanced with ML)
