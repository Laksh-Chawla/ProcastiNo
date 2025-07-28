# App usage tracking - keeps track of what apps you use and for how long
# Stores everything in JSON for later analysis

import json
import time
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import psutil
    import pygetwindow as gw
    from win32gui import GetWindowText, GetForegroundWindow
    import win32process
except ImportError as e:
    print(f"Import error: {e}. Please install required packages.")


class AppUsageTracker:
    # tracks how long you spend in different apps

    def __init__(self, data_file: str = "app_usage_data.json"):
        self.data_file = data_file
        self.current_app = None
        self.session_start_time = None
        self.tracking_active = False
        self.usage_data = self.load_data()
        self.lock = threading.Lock()

        # basic categories for common apps - not exhaustive but covers most stuff
        self.app_categories = {
            "chrome": "Productivity",
            "firefox": "Productivity",
            "edge": "Productivity",
            "code": "Work",
            "notepad": "Work",
            "word": "Work",
            "excel": "Work",
            "instagram": "Social",
            "facebook": "Social",
            "twitter": "Social",
            "whatsapp": "Social",
            "youtube": "Entertainment",
            "netflix": "Entertainment",
            "spotify": "Entertainment",
            "steam": "Gaming",
            "discord": "Social",
            "teams": "Work",
            "figma": "Work",
            "photoshop": "Work",
            "illustrator": "Work",
        }

        print("[AppTracker] Initialized successfully")

    def get_active_window_info(self) -> Optional[Dict[str, str]]:
        """
        Get information about the currently active window
        Returns: Dict with app_name, window_title, process_name
        """
        try:
            # Get the handle of the foreground window
            hwnd = GetForegroundWindow()
            window_title = GetWindowText(hwnd)

            if not window_title:
                return None

            # Get process ID from window handle
            _, pid = win32process.GetWindowThreadProcessId(hwnd)

            # Get process information
            try:
                process = psutil.Process(pid)
                process_name = process.name().lower().replace(".exe", "")

                return {
                    "app_name": process_name,
                    "window_title": window_title,
                    "process_name": process_name,
                    "pid": pid,
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return None

        except Exception as e:
            print(f"[AppTracker] Error getting active window: {e}")
            return None

    def get_app_category(self, app_name: str) -> str:
        """
        Categorize application based on its name
        """
        app_lower = app_name.lower()

        for key, category in self.app_categories.items():
            if key in app_lower:
                return category

        return "Other"

    def start_tracking(self) -> None:
        """
        Start tracking application usage in a separate thread
        """
        if self.tracking_active:
            print("[AppTracker] Already tracking!")
            return

        self.tracking_active = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        print("[AppTracker] Started tracking application usage")

    def stop_tracking(self) -> None:
        """
        Stop tracking and save any active session
        """
        if not self.tracking_active:
            return

        self.tracking_active = False
        self._end_current_session()
        self.save_data()
        print("[AppTracker] Stopped tracking")

    def _tracking_loop(self) -> None:
        """
        Main tracking loop - runs in separate thread
        """
        while self.tracking_active:
            try:
                window_info = self.get_active_window_info()

                if window_info:
                    current_app = window_info["app_name"]

                    # If app changed, end previous session and start new one
                    if current_app != self.current_app:
                        self._end_current_session()
                        self._start_new_session(window_info)

                # Check every 2 seconds
                time.sleep(2)

            except Exception as e:
                print(f"[AppTracker] Error in tracking loop: {e}")
                time.sleep(5)  # Wait longer on error

    def _start_new_session(self, window_info: Dict[str, str]) -> None:
        """
        Start a new app usage session
        """
        with self.lock:
            self.current_app = window_info["app_name"]
            self.session_start_time = datetime.now()

            print(f"[AppTracker] Started session: {self.current_app}")

    def _end_current_session(self) -> None:
        """
        End the current app usage session and log it
        """
        if not self.current_app or not self.session_start_time:
            return

        with self.lock:
            session_end_time = datetime.now()
            duration_seconds = (
                session_end_time - self.session_start_time
            ).total_seconds()

            # Only log sessions longer than 5 seconds
            if duration_seconds >= 5:
                session_data = {
                    "app_name": self.current_app,
                    "category": self.get_app_category(self.current_app),
                    "start_time": self.session_start_time.isoformat(),
                    "end_time": session_end_time.isoformat(),
                    "duration_seconds": round(duration_seconds, 2),
                    "date": self.session_start_time.strftime("%Y-%m-%d"),
                }

                self.usage_data.append(session_data)
                print(
                    f"[AppTracker] Logged session: {self.current_app} - {duration_seconds:.1f}s"
                )

                # Auto-save every few sessions
                if len(self.usage_data) % 10 == 0:
                    self.save_data()

            self.current_app = None
            self.session_start_time = None

    def get_today_stats(self) -> Dict:
        """
        Get statistics for today's usage
        """
        today = datetime.now().strftime("%Y-%m-%d")
        today_sessions = [s for s in self.usage_data if s.get("date") == today]

        if not today_sessions:
            return {
                "total_time_minutes": 0,
                "app_breakdown": {},
                "category_breakdown": {},
                "session_count": 0,
                "most_used_app": "None",
            }

        # Calculate totals by app and category
        app_breakdown = {}
        category_breakdown = {}

        for session in today_sessions:
            app = session["app_name"]
            category = session["category"]
            duration = session["duration_seconds"] / 60  # Convert to minutes

            app_breakdown[app] = app_breakdown.get(app, 0) + duration
            category_breakdown[category] = (
                category_breakdown.get(category, 0) + duration
            )

        total_time = sum(app_breakdown.values())
        most_used_app = (
            max(app_breakdown.items(), key=lambda x: x[1])[0]
            if app_breakdown
            else "None"
        )

        return {
            "total_time_minutes": round(total_time, 1),
            "app_breakdown": {k: round(v, 1) for k, v in app_breakdown.items()},
            "category_breakdown": {
                k: round(v, 1) for k, v in category_breakdown.items()
            },
            "session_count": len(today_sessions),
            "most_used_app": most_used_app,
        }

    def get_week_stats(self) -> Dict:
        """
        Get statistics for the past 7 days
        """
        week_ago = datetime.now() - timedelta(days=7)
        week_sessions = [
            s
            for s in self.usage_data
            if datetime.fromisoformat(s["start_time"]) >= week_ago
        ]

        daily_totals = {}
        for session in week_sessions:
            date = session["date"]
            duration = session["duration_seconds"] / 60
            daily_totals[date] = daily_totals.get(date, 0) + duration

        return {
            "daily_totals": {k: round(v, 1) for k, v in daily_totals.items()},
            "total_sessions": len(week_sessions),
            "average_daily_time": (
                round(sum(daily_totals.values()) / 7, 1) if daily_totals else 0
            ),
        }

    def get_current_session_info(self) -> Optional[Dict]:
        """
        Get information about the current active session
        """
        if not self.current_app or not self.session_start_time:
            return None

        duration = (datetime.now() - self.session_start_time).total_seconds()
        return {
            "app_name": self.current_app,
            "category": self.get_app_category(self.current_app),
            "duration_seconds": round(duration, 1),
            "start_time": self.session_start_time.isoformat(),
        }

    def save_data(self) -> None:
        """
        Save usage data to JSON file
        """
        try:
            with open(self.data_file, "w") as f:
                json.dump(self.usage_data, f, indent=2)
            print(f"[AppTracker] Data saved ({len(self.usage_data)} sessions)")
        except Exception as e:
            print(f"[AppTracker] Error saving data: {e}")

    def load_data(self) -> List[Dict]:
        """
        Load usage data from JSON file
        """
        if not os.path.exists(self.data_file):
            print(f"[AppTracker] No existing data file found")
            return []

        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
            print(f"[AppTracker] Loaded {len(data)} sessions from file")
            return data
        except Exception as e:
            print(f"[AppTracker] Error loading data: {e}")
            return []

    def clean_old_data(self, days_to_keep: int = 30) -> None:
        """
        Remove data older than specified days
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        original_count = len(self.usage_data)

        self.usage_data = [
            session
            for session in self.usage_data
            if datetime.fromisoformat(session["start_time"]) >= cutoff_date
        ]

        removed_count = original_count - len(self.usage_data)
        if removed_count > 0:
            print(f"[AppTracker] Cleaned {removed_count} old sessions")
            self.save_data()


# Example usage and testing
if __name__ == "__main__":
    import signal
    import sys

    def signal_handler(sig, frame):
        print("\n[AppTracker] Stopping tracker...")
        tracker.stop_tracking()
        sys.exit(0)

    # Create and start tracker
    tracker = AppUsageTracker()

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)

    # Start tracking
    tracker.start_tracking()

    try:
        # Print stats every 10 seconds for testing
        while True:
            time.sleep(10)

            current_session = tracker.get_current_session_info()
            if current_session:
                print(
                    f"Current: {current_session['app_name']} - {current_session['duration_seconds']:.1f}s"
                )

            today_stats = tracker.get_today_stats()
            print(
                f"Today: {today_stats['total_time_minutes']:.1f} minutes, {today_stats['session_count']} sessions"
            )
            print("---")

    except KeyboardInterrupt:
        print("\n[AppTracker] Stopping...")
        tracker.stop_tracking()
