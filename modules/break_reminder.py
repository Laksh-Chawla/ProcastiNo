# Break reminder system - reminds users to take breaks
# Based on research about productivity and preventing burnout

import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from plyer import notification


@dataclass
class WorkSession:
    # represents a work session with break info
    start_time: str
    end_time: Optional[str] = None
    duration_minutes: float = 0.0
    break_taken: bool = False
    break_duration_minutes: float = 0.0
    productivity_apps: List[str] = None

    def __post_init__(self):
        if self.productivity_apps is None:
            self.productivity_apps = []


class BreakReminder:
    # monitors how long user works and suggests breaks

    def __init__(
        self,
        data_file: str = "break_data.json",
        work_threshold_minutes: int = 45,
        break_duration_minutes: int = 15,
    ):
        self.data_file = data_file
        self.work_threshold_minutes = work_threshold_minutes  # when to suggest break
        self.break_duration_minutes = break_duration_minutes  # how long break should be

        # keep track of current work session
        self.current_session = None
        self.session_start_time: Optional[datetime] = None
        self.last_activity_time: Optional[datetime] = None
        self.on_break = False
        self.break_start_time: Optional[datetime] = None

        # Historical data
        self.work_sessions: List[WorkSession] = []

        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None

        # Callback functions
        self.on_break_reminder: Optional[Callable] = None
        self.on_break_started: Optional[Callable] = None
        self.on_break_ended: Optional[Callable] = None

        # Productivity app categories
        self.productivity_apps = {
            "work": [
                "code",
                "word",
                "excel",
                "powerpoint",
                "figma",
                "photoshop",
                "illustrator",
            ],
            "productivity": ["chrome", "firefox", "edge", "notepad", "calculator"],
            "communication": ["teams", "slack", "zoom", "outlook"],
        }

        # Load existing data
        self.load_data()

        print("[BreakReminder] Initialized successfully")

    def load_data(self) -> None:
        """
        Load break session data from JSON file
        """
        if not os.path.exists(self.data_file):
            print("[BreakReminder] No existing break data file found")
            return

        try:
            with open(self.data_file, "r") as f:
                sessions_data = json.load(f)

            self.work_sessions = []
            for session_dict in sessions_data:
                session = WorkSession(**session_dict)
                self.work_sessions.append(session)

            print(f"[BreakReminder] Loaded {len(self.work_sessions)} work sessions")

        except Exception as e:
            print(f"[BreakReminder] Error loading break data: {e}")
            self.work_sessions = []

    def save_data(self) -> None:
        """
        Save break session data to JSON file
        """
        try:
            sessions_data = [asdict(session) for session in self.work_sessions]
            with open(self.data_file, "w") as f:
                json.dump(sessions_data, f, indent=2)
            print(f"[BreakReminder] Saved {len(self.work_sessions)} work sessions")
        except Exception as e:
            print(f"[BreakReminder] Error saving break data: {e}")

    def is_productivity_app(self, app_name: str) -> bool:
        """
        Check if an app is considered productive work

        Args:
            app_name: Name of the application

        Returns:
            True if app is considered productive
        """
        app_lower = app_name.lower()

        for category, apps in self.productivity_apps.items():
            if any(prod_app in app_lower for prod_app in apps):
                return True

        return False

    def start_work_session(self) -> None:
        """
        Start a new work session
        """
        if self.current_session is not None:
            return  # Already in session

        self.current_session = WorkSession(
            start_time=datetime.now().isoformat(), productivity_apps=[]
        )
        self.session_start_time = datetime.now()
        self.last_activity_time = datetime.now()
        self.on_break = False

        print("[BreakReminder] Started new work session")

        # Start monitoring if not already active
        if not self.monitoring_active:
            self.start_monitoring()

    def end_work_session(self) -> None:
        """
        End the current work session
        """
        if self.current_session is None:
            return

        end_time = datetime.now()
        duration_minutes = (end_time - self.session_start_time).total_seconds() / 60

        self.current_session.end_time = end_time.isoformat()
        self.current_session.duration_minutes = round(duration_minutes, 1)

        # Save completed session
        self.work_sessions.append(self.current_session)
        self.save_data()

        print(
            f"[BreakReminder] Ended work session - Duration: {duration_minutes:.1f} minutes"
        )

        # Reset current session
        self.current_session = None
        self.session_start_time = None
        self.last_activity_time = None

    def update_activity(self, app_name: str) -> None:
        """
        Update activity based on current app usage

        Args:
            app_name: Name of the currently active app
        """
        current_time = datetime.now()

        # Check if user was on break and came back to work
        if self.on_break and self.is_productivity_app(app_name):
            self.end_break()

        # If no current session and using productivity app, start session
        if self.current_session is None and self.is_productivity_app(app_name):
            self.start_work_session()

        # Update activity for current session
        if self.current_session is not None:
            self.last_activity_time = current_time

            # Track productivity apps used
            if (
                self.is_productivity_app(app_name)
                and app_name not in self.current_session.productivity_apps
            ):
                self.current_session.productivity_apps.append(app_name)

        # Check if user has been away from work too long (auto-break)
        elif (
            self.current_session is not None
            and not self.is_productivity_app(app_name)
            and not self.on_break
        ):
            time_away = (current_time - self.last_activity_time).total_seconds() / 60
            if time_away >= 10:  # 10 minutes away = automatic break
                self.start_break(automatic=True)

    def check_break_needed(self) -> bool:
        """
        Check if a break reminder should be sent

        Returns:
            True if break is needed
        """
        if (
            self.current_session is None
            or self.on_break
            or self.session_start_time is None
        ):
            return False

        # Calculate continuous work time
        work_duration = (datetime.now() - self.session_start_time).total_seconds() / 60

        return work_duration >= self.work_threshold_minutes

    def send_break_reminder(self) -> None:
        """
        Send a desktop notification reminding user to take a break
        """
        try:
            work_duration = (
                datetime.now() - self.session_start_time
            ).total_seconds() / 60

            notification.notify(
                title="Break Time!",
                message=f"You've been working for {work_duration:.0f} minutes. Time for a {self.break_duration_minutes}-minute break!",
                app_name="ProcastiNo",
                timeout=15,
            )

            print(
                f"[BreakReminder] Sent break reminder after {work_duration:.0f} minutes of work"
            )

            # Call callback if set
            if self.on_break_reminder:
                self.on_break_reminder(work_duration)

        except Exception as e:
            print(f"[BreakReminder] Error sending break notification: {e}")

    def start_break(self, automatic: bool = False) -> None:
        """
        Start a break period

        Args:
            automatic: True if break was detected automatically
        """
        if self.on_break:
            return  # Already on break

        self.on_break = True
        self.break_start_time = datetime.now()

        break_type = "automatic" if automatic else "manual"
        print(f"[BreakReminder] Started {break_type} break")

        # Call callback if set
        if self.on_break_started:
            self.on_break_started(automatic)

    def end_break(self) -> None:
        """
        End the current break period
        """
        if not self.on_break or self.break_start_time is None:
            return

        break_duration = (datetime.now() - self.break_start_time).total_seconds() / 60

        # Update current session with break info
        if self.current_session:
            self.current_session.break_taken = True
            self.current_session.break_duration_minutes += break_duration

        print(f"[BreakReminder] Ended break - Duration: {break_duration:.1f} minutes")

        self.on_break = False
        self.break_start_time = None

        # Reset session start time to current time (restart work timer)
        self.session_start_time = datetime.now()
        self.last_activity_time = datetime.now()

        # Call callback if set
        if self.on_break_ended:
            self.on_break_ended(break_duration)

    def start_monitoring(self) -> None:
        """
        Start monitoring for break reminders in a separate thread
        """
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        print("[BreakReminder] Started break monitoring")

    def stop_monitoring(self) -> None:
        """
        Stop break monitoring
        """
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)

        # End current session
        if self.current_session:
            self.end_work_session()

        print("[BreakReminder] Stopped break monitoring")

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop - runs in separate thread
        """
        last_reminder_time = None

        while self.monitoring_active:
            try:
                if self.check_break_needed():
                    # Only send reminder if we haven't sent one recently
                    now = datetime.now()
                    if (
                        last_reminder_time is None
                        or (now - last_reminder_time).total_seconds() >= 600
                    ):  # 10 minutes between reminders

                        self.send_break_reminder()
                        last_reminder_time = now

                # Check every minute
                time.sleep(60)

            except Exception as e:
                print(f"[BreakReminder] Error in monitoring loop: {e}")
                time.sleep(120)  # Wait longer on error

    def get_current_session_info(self) -> Optional[Dict]:
        """
        Get information about the current work session
        """
        if self.current_session is None or self.session_start_time is None:
            return None

        duration_minutes = (
            datetime.now() - self.session_start_time
        ).total_seconds() / 60

        return {
            "start_time": self.current_session.start_time,
            "duration_minutes": round(duration_minutes, 1),
            "on_break": self.on_break,
            "break_needed": self.check_break_needed(),
            "time_until_break": max(0, self.work_threshold_minutes - duration_minutes),
            "productivity_apps": self.current_session.productivity_apps,
            "break_duration": (
                round((datetime.now() - self.break_start_time).total_seconds() / 60, 1)
                if self.on_break and self.break_start_time
                else 0
            ),
        }

    def get_today_stats(self) -> Dict:
        """
        Get break and work statistics for today
        """
        today = datetime.now().date()
        today_sessions = []

        # Get today's completed sessions
        for session in self.work_sessions:
            try:
                session_date = datetime.fromisoformat(session.start_time).date()
                if session_date == today:
                    today_sessions.append(session)
            except ValueError:
                continue

        # Include current session if active
        current_info = self.get_current_session_info()

        # Calculate statistics
        total_work_time = sum(session.duration_minutes for session in today_sessions)
        if current_info and not current_info["on_break"]:
            total_work_time += current_info["duration_minutes"]

        total_break_time = sum(
            session.break_duration_minutes for session in today_sessions
        )
        if current_info and current_info["on_break"]:
            total_break_time += current_info["break_duration"]

        breaks_taken = sum(1 for session in today_sessions if session.break_taken)
        if current_info and current_info["on_break"]:
            breaks_taken += 1

        # Calculate break compliance (breaks taken vs breaks needed)
        breaks_needed = max(1, int(total_work_time / self.work_threshold_minutes))
        break_compliance = (
            min(100, (breaks_taken / breaks_needed * 100)) if breaks_needed > 0 else 100
        )

        return {
            "total_work_time_minutes": round(total_work_time, 1),
            "total_break_time_minutes": round(total_break_time, 1),
            "breaks_taken": breaks_taken,
            "breaks_needed": breaks_needed,
            "break_compliance_percent": round(break_compliance, 1),
            "average_session_duration": (
                round(total_work_time / len(today_sessions), 1) if today_sessions else 0
            ),
            "longest_session_minutes": max(
                (session.duration_minutes for session in today_sessions), default=0
            ),
            "current_session_active": current_info is not None,
            "currently_on_break": current_info["on_break"] if current_info else False,
        }

    def get_weekly_stats(self) -> Dict:
        """
        Get break and work statistics for the past week
        """
        week_ago = datetime.now() - timedelta(days=7)
        week_sessions = []

        for session in self.work_sessions:
            try:
                session_time = datetime.fromisoformat(session.start_time)
                if session_time >= week_ago:
                    week_sessions.append(session)
            except ValueError:
                continue

        if not week_sessions:
            return {
                "total_sessions": 0,
                "total_work_hours": 0,
                "total_break_hours": 0,
                "average_daily_work": 0,
                "break_compliance": 0,
                "most_productive_day": "None",
            }

        # Calculate totals
        total_work_minutes = sum(session.duration_minutes for session in week_sessions)
        total_break_minutes = sum(
            session.break_duration_minutes for session in week_sessions
        )

        # Daily breakdown
        daily_work = {}
        for session in week_sessions:
            date = datetime.fromisoformat(session.start_time).date()
            daily_work[date] = daily_work.get(date, 0) + session.duration_minutes

        most_productive_day = (
            max(daily_work.items(), key=lambda x: x[1])[0] if daily_work else None
        )

        return {
            "total_sessions": len(week_sessions),
            "total_work_hours": round(total_work_minutes / 60, 1),
            "total_break_hours": round(total_break_minutes / 60, 1),
            "average_daily_work": round(total_work_minutes / 7 / 60, 1),
            "break_compliance": round(
                sum(1 for s in week_sessions if s.break_taken)
                / len(week_sessions)
                * 100,
                1,
            ),
            "most_productive_day": (
                most_productive_day.strftime("%A") if most_productive_day else "None"
            ),
            "daily_breakdown": {
                str(date): round(minutes / 60, 1)
                for date, minutes in daily_work.items()
            },
        }

    def force_break(self) -> None:
        """
        Manually start a break
        """
        if self.current_session is None:
            self.start_work_session()

        self.start_break(automatic=False)

    def skip_break(self) -> None:
        """
        Skip the current break reminder and reset timer
        """
        if self.current_session and self.session_start_time:
            # Reset session start time to extend work period
            self.session_start_time = datetime.now()
            print("[BreakReminder] Break skipped, work timer reset")


# Example usage and testing
if __name__ == "__main__":
    import signal
    import sys
    import random

    def signal_handler(sig, frame):
        print("\n[BreakReminder] Stopping...")
        break_reminder.stop_monitoring()
        sys.exit(0)

    # Create break reminder
    break_reminder = BreakReminder(work_threshold_minutes=2)  # 2 minutes for testing

    # Set up callbacks
    def on_break_reminder(work_duration):
        print(f"CALLBACK: Break reminder sent after {work_duration:.1f} minutes")

    def on_break_started(automatic):
        break_type = "automatic" if automatic else "manual"
        print(f"CALLBACK: {break_type.title()} break started")

    def on_break_ended(break_duration):
        print(f"CALLBACK: Break ended after {break_duration:.1f} minutes")

    break_reminder.on_break_reminder = on_break_reminder
    break_reminder.on_break_started = on_break_started
    break_reminder.on_break_ended = on_break_ended

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)

    # Start a work session
    break_reminder.start_work_session()

    try:
        print(
            f"\nBreak monitoring started - Threshold: {break_reminder.work_threshold_minutes} minutes"
        )
        print("Simulating work activity...")
        print("Press Ctrl+C to stop")

        while True:
            # Simulate app usage
            productivity_apps = ["code", "word", "figma", "chrome"]
            distraction_apps = ["instagram", "youtube", "spotify"]

            # 70% chance of using productivity app
            if random.random() < 0.7:
                app = random.choice(productivity_apps)
            else:
                app = random.choice(distraction_apps)

            break_reminder.update_activity(app)

            # Show current status
            current_info = break_reminder.get_current_session_info()
            if current_info:
                status = "On Break" if current_info["on_break"] else "Working"
                print(
                    f"Status: {status} - Duration: {current_info['duration_minutes']:.1f}m - App: {app}"
                )

                if current_info["break_needed"] and not current_info["on_break"]:
                    print("  --> BREAK NEEDED!")

            # Show daily stats occasionally
            if random.random() < 0.1:  # 10% chance
                stats = break_reminder.get_today_stats()
                print(
                    f"Today: {stats['total_work_time_minutes']:.1f}m work, {stats['breaks_taken']} breaks"
                )

            time.sleep(10)  # Check every 10 seconds

    except KeyboardInterrupt:
        print("\n[BreakReminder] Stopping...")
        break_reminder.stop_monitoring()
