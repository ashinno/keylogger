import re
from collections import defaultdict
from datetime import datetime

LOG_FILE = 'keylog.txt'
REPORT_FILE = 'readable_log_report.txt'

def parse_log_line(line):
    """Parses a single log line to extract timestamp, event type, details, and application."""
    # Regex to capture timestamp (optional), event description, and application context
    # Handles lines with and without timestamps
    match = re.match(r"(?:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}): )?(.*?) (?:in (.*?)(?: - .*?)?$|$)", line.strip())
    if match:
        timestamp_str, event_desc, app_name = match.groups()
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f') if timestamp_str else None
        app_name = app_name if app_name else 'Unknown Application'

        # Further classify the event based on description
        if 'Key pressed:' in event_desc:
            parts = event_desc.split('Key pressed: ')
            if len(parts) > 1:
                key = parts[1].strip()
                # Handle special keys like 'Key.space', 'Key.enter', etc.
                if key.startswith("'\\x"):
                     # Skip hex representation for now, might need better handling
                     return timestamp, 'KeyPress', None, app_name
                elif key.startswith('Key.'):
                    key_name = key.split('.')[-1]
                    if key_name == 'space': key = ' '
                    elif key_name == 'enter': key = '\n'
                    elif key_name == 'backspace': key = '[BACKSPACE]'
                    # Add more special key mappings as needed
                    else: key = f'[{key_name.upper()}]'
                else:
                     key = key.strip("'") # Keep normal characters
                return timestamp, 'KeyPress', key, app_name
            else:
                # Handle cases where 'Key pressed:' is present but no key follows
                return timestamp, 'KeyPress', None, app_name # Or log an error/warning
        elif 'Active window:' in event_desc:
            return timestamp, 'WindowChange', None, app_name # App name already captured
        elif 'Mouse pressed' in event_desc or 'Mouse released' in event_desc:
            return timestamp, 'MouseActivity', event_desc, app_name
        elif 'Clipboard changed:' in event_desc:
            parts = event_desc.split('Clipboard changed: ')
            details = parts[1] if len(parts) > 1 else ''
            return timestamp, 'ClipboardChange', details, app_name
        elif 'Screenshot captured:' in event_desc:
             parts = event_desc.split('Screenshot captured: ')
             details = parts[1] if len(parts) > 1 else ''
             return timestamp, 'Screenshot', details, app_name
        # Add more event types if needed (Key released, Shortcut used, etc.)

    return None, 'Unknown', line.strip(), 'Unknown Application' # Fallback for unparsed lines

def generate_report(log_file, report_file):
    """Reads the log file, parses it, and generates a structured report."""
    typed_text_by_app = defaultdict(str)
    app_usage = defaultdict(lambda: {'first_seen': None, 'last_seen': None, 'events': []})
    current_app = 'Unknown Application'
    last_timestamp = None

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                timestamp, event_type, details, app_name = parse_log_line(line)

                if not timestamp:
                    # Try to use the last known timestamp if a line doesn't have one
                    timestamp = last_timestamp
                else:
                    last_timestamp = timestamp

                if app_name != 'Unknown Application':
                    current_app = app_name

                # Track app usage time
                if current_app not in app_usage or not app_usage[current_app]['first_seen']:
                    app_usage[current_app]['first_seen'] = timestamp
                if timestamp:
                    app_usage[current_app]['last_seen'] = timestamp

                # Store event details per app
                app_usage[current_app]['events'].append((timestamp, event_type, details))

                # Reconstruct typed text
                if event_type == 'KeyPress' and details:
                    if details == '[BACKSPACE]':
                         if typed_text_by_app[current_app]:
                             typed_text_by_app[current_app] = typed_text_by_app[current_app][:-1]
                    elif not details.startswith('['): # Append normal characters and space/newline
                        typed_text_by_app[current_app] += details
                    # Optionally handle other special keys like [SHIFT], [CTRL] if needed

    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.")
        return
    except Exception as e:
        print(f"Error processing log file: {e}")
        return

    # Write the report
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Keylogger Activity Report\n")
            f.write("=========================\n\n")

            f.write("Application Usage Summary:\n")
            f.write("--------------------------\n")
            for app, usage in sorted(app_usage.items()):
                start = usage['first_seen'].strftime('%Y-%m-%d %H:%M:%S') if usage['first_seen'] else 'N/A'
                end = usage['last_seen'].strftime('%Y-%m-%d %H:%M:%S') if usage['last_seen'] else 'N/A'
                duration_str = 'N/A'
                if usage['first_seen'] and usage['last_seen']:
                    duration = usage['last_seen'] - usage['first_seen']
                    duration_str = str(duration)

                f.write(f"- {app}:\n")
                f.write(f"    First Seen: {start}\n")
                f.write(f"    Last Seen:  {end}\n")
                f.write(f"    Approx Duration: {duration_str}\n")
            f.write("\n")

            f.write("Typed Text Summary (per application):\n")
            f.write("-------------------------------------\n")
            for app, text in sorted(typed_text_by_app.items()):
                f.write(f"--- {app} ---\n")
                f.write(text.strip() + "\n") # Add newline after each app's text
                f.write("--------------------" + "-"*len(app) + "\n\n")

            # Optional: Add a section for other events like mouse clicks, screenshots etc.
            f.write("Other Notable Events:\n")
            f.write("---------------------\n")
            for app, usage in sorted(app_usage.items()):
                 has_other_events = any(e[1] not in ['KeyPress', 'WindowChange'] for e in usage['events'])
                 if has_other_events:
                     f.write(f"--- {app} ---\n")
                     for ts, event, detail in usage['events']:
                         if event not in ['KeyPress', 'WindowChange']:
                             ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if ts else 'Unknown Time'
                             f.write(f"    {ts_str}: {event} - {detail}\n")
                     f.write("\n")

        print(f"Report generated successfully: '{report_file}'")

    except Exception as e:
        print(f"Error writing report file: {e}")

if __name__ == "__main__":
    generate_report(LOG_FILE, REPORT_FILE)