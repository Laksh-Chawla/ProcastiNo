# ProcastiNO - A productivity app to help fight procrastination
# Built with PyQt5 and some ML stuff

import sys
import psutil
import time
import json
import os
import warnings
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QFrame,
    QScrollArea,
    QGridLayout,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QSplitter,
    QDialog,
    QLineEdit,
    QComboBox,
    QDialogButtonBox,
    QMessageBox,
    QTabWidget,
    QCheckBox,
    QSpinBox,
    QButtonGroup,
)
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import threading

# Suppress numpy warnings that cause crashes
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*MINGW.*")

# Try to import ML stuff - not everyone has sklearn installed
try:
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Scikit-learn not available - ML features disabled")
except Exception as e:
    ML_AVAILABLE = False
    print(f"ML libraries have issues - ML features disabled: {e}")

# Windows-specific stuff for getting active windows
try:
    import win32gui
    import win32process

    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    print("win32gui not available - some features may be limited")

# Import our custom modules
try:
    from modules.distraction_predictor import DistractionPredictor
    from modules.app_tracker import AppUsageTracker
    from modules.task_manager import TaskManager
    from modules.break_reminder import BreakReminder
    from modules.analytics import ScreenTimeAnalytics, AnalyticsWidget

    MODULES_AVAILABLE = True
    print("✓ All custom modules imported successfully!")
except ImportError as e:
    print(f"Module import error: {e}")
    MODULES_AVAILABLE = False
except Exception as e:
    print(f"Module import error (other): {e}")
    MODULES_AVAILABLE = False


# Fallback classes if modules aren't available
class FallbackTaskManager:
    def __init__(self):
        self.tasks = []

    def get_active_tasks(self):
        return []

    def create_task(self, title, description="", app_assigned="", priority="Medium"):
        return True

    def get_task_stats(self):
        return {"active": 0, "today_completed": 0, "completion_rate": 0.0}


class FallbackDistraction:
    def extract_features(self, *args):
        return [0.5] * 6

    def predict(self, features):
        return 0.3


class RealTimeAppTracker(QThread):
    # Background thread that keeps track of what apps are running
    # Also does AI prediction stuff if available

    apps_updated = pyqtSignal(list)
    distraction_predicted = pyqtSignal(float, str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.current_apps = {}
        if MODULES_AVAILABLE:
            self.predictor = DistractionPredictor()
        else:
            self.predictor = FallbackDistraction()
        self.app_switches = []
        self.current_app_name = None
        self.current_app_start = datetime.now()

    def run(self):
        self.running = True
        prediction_counter = 0

        while self.running:
            try:
                apps = self.get_running_apps()
                self.apps_updated.emit(apps)

                # check which app is currently active for AI prediction
                active_apps = [app for app in apps if app.get("is_active", False)]
                if active_apps:
                    current_app = active_apps[0]["name"]
                    if current_app != self.current_app_name:
                        # user switched apps
                        if self.current_app_name:
                            self.app_switches.append(
                                {
                                    "time": datetime.now().isoformat(),
                                    "from": self.current_app_name,
                                    "to": current_app,
                                }
                            )

                        self.current_app_name = current_app
                        self.current_app_start = datetime.now()

                    # Make AI prediction every 10 cycles (~10 seconds)
                    prediction_counter += 1
                    if prediction_counter >= 10:
                        prediction_counter = 0
                        try:
                            probability, reason = self.make_distraction_prediction(apps)
                            self.distraction_predicted.emit(probability, reason)
                        except Exception as e:
                            # Fallback if AI prediction fails
                            self.distraction_predicted.emit(
                                0.3, "AI prediction unavailable"
                            )

                QThread.msleep(1000)  # Update every second

            except Exception as e:
                print(f"Tracker error: {e}")
                QThread.msleep(2000)

    def make_distraction_prediction(self, apps):
        """Make AI distraction prediction"""
        if not self.current_app_name:
            return 0.3, "No active app detected"

        if not self.predictor:
            return 0.3, "No predictor available"

        try:
            # figure out current app usage patterns
            time_on_app = (datetime.now() - self.current_app_start).total_seconds() / 60
            total_apps = len(apps)
            total_memory = sum(app.get("memory_mb", 0) for app in apps)
            current_hour = datetime.now().hour
            switches_today = len(
                [
                    s
                    for s in self.app_switches
                    if datetime.fromisoformat(s["time"]).date() == datetime.now().date()
                ]
            )

            features = self.predictor.extract_features(
                self.current_app_name,
                time_on_app,
                total_apps,
                total_memory,
                current_hour,
                switches_today,
            )

            probability = self.predictor.predict(features)

            # come up with a reason for this prediction
            reason = self.generate_prediction_reason(
                self.current_app_name, time_on_app, switches_today, current_hour
            )

            return probability, reason

        except Exception as e:
            return 0.3, f"Prediction error: {str(e)[:50]}"

    def generate_prediction_reason(self, app_name, time_on_app, switches, hour):
        # try to explain why we think user might be distracted
        reasons = []

        # check what kind of app they're using
        if any(
            keyword in app_name.lower() for keyword in ["chrome", "firefox", "edge"]
        ):
            if time_on_app > 30:
                reasons.append("Long browser session")
        elif any(
            keyword in app_name.lower() for keyword in ["youtube", "netflix", "spotify"]
        ):
            reasons.append("Entertainment app active")
        elif any(
            keyword in app_name.lower() for keyword in ["discord", "slack", "teams"]
        ):
            reasons.append("Communication app")
        elif any(
            keyword in app_name.lower() for keyword in ["code", "studio", "pycharm"]
        ):
            reasons.append("Development environment")

        # time of day matters too
        if hour < 9 or hour > 17:
            reasons.append("Outside work hours")

        # Switch frequency
        if switches > 20:
            reasons.append("High app switching")
        elif switches < 5:
            reasons.append("Stable focus pattern")

        # Session length
        if time_on_app > 60:
            reasons.append("Extended session")
        elif time_on_app < 2:
            reasons.append("Brief usage")

        return "; ".join(reasons) if reasons else "Normal usage pattern"

    def get_app_switches_today(self):
        """Get app switches from today"""
        today = datetime.now().date()
        return [
            s
            for s in self.app_switches
            if datetime.fromisoformat(s["time"]).date() == today
        ]

    def stop(self):
        """Stop the tracking thread"""
        self.running = False

    def get_running_apps(self):
        """Get list of running applications that user is interacting with"""
        apps = []
        current_time = datetime.now()

        try:
            # Get all running processes
            for proc in psutil.process_iter(
                ["pid", "name", "memory_info", "cpu_percent"]
            ):
                try:
                    proc_info = proc.info
                    pid = proc_info["pid"]
                    name = proc_info["name"]

                    # Filter out system processes and focus on user applications
                    if self.is_user_app(name, pid):
                        memory_mb = proc_info["memory_info"].rss / 1024 / 1024
                        cpu = proc_info["cpu_percent"] or 0

                        # Check if this is the active window
                        is_active = (
                            self.is_active_window(pid) if WIN32_AVAILABLE else False
                        )

                        app_data = {
                            "name": name,
                            "pid": pid,
                            "memory_mb": memory_mb,
                            "cpu_percent": cpu,
                            "is_active": is_active,
                            "last_seen": current_time,
                        }
                        apps.append(app_data)

                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    continue

        except Exception as e:
            print(f"Error getting running apps: {e}")

        # Sort by activity (active window first, then by CPU usage)
        apps.sort(key=lambda x: (not x["is_active"], -x["cpu_percent"]))
        return apps[:20]  # Return top 20 apps

    def is_user_app(self, name, pid):
        """Check if this is likely a user application"""
        system_processes = {
            "svchost.exe",
            "dwm.exe",
            "winlogon.exe",
            "csrss.exe",
            "smss.exe",
            "explorer.exe",
            "lsass.exe",
            "services.exe",
            "spoolsv.exe",
            "audiodg.exe",
            "conhost.exe",
            "fontdrvhost.exe",
            "taskhostw.exe",
        }

        # Skip system processes
        if name.lower() in system_processes:
            return False

        # Skip very low PID processes (usually system)
        if pid < 100:
            return False

        return True

    def is_active_window(self, pid):
        """Check if process has the active window"""
        if not WIN32_AVAILABLE:
            return False

        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                _, window_pid = win32process.GetWindowThreadProcessId(hwnd)
                return window_pid == pid
        except:
            pass
        return False


class AddTaskDialog(QDialog):
    # dialog box for adding new tasks
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Task")
        self.setFixedSize(650, 550)  # made it a bit bigger
        self.setup_ui()

    def setup_ui(self):
        # get screen size to make fonts look good on different monitors
        screen = QApplication.primaryScreen().geometry()
        base_font_size = max(12, screen.width() // 120)

        # dark theme that matches the main app
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: #3c3f5a;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QLabel {{
                color: #ffffff;
                font-size: {base_font_size + 2}px;
                font-weight: 500;
            }}
            QLineEdit {{
                background-color: #2a2d42;
                border: 1px solid #4a4d62;
                border-radius: 8px;
                padding: 12px;
                color: #ffffff;
                font-size: {base_font_size}px;
                min-height: 20px;
            }}
            QLineEdit::placeholder {{
                color: #8b8f9f;
            }}
            QPushButton#priority_btn {{
                background-color: #2a2d42;
                border: 2px solid #4a4d62;
                border-radius: 12px;
                padding: 20px;
                font-size: {base_font_size + 4}px;
                font-weight: bold;
                min-height: 40px;
                min-width: 120px;
            }}
            QPushButton#priority_btn:checked {{
                border: 2px solid #ffffff;
            }}
            QPushButton#priority_btn:hover {{
                background-color: #353851;
            }}
            QPushButton#create_btn {{
                background-color: #8b5cf6;
                border: none;
                border-radius: 12px;
                padding: 20px;
                color: #ffffff;
                font-size: {base_font_size + 6}px;
                font-weight: bold;
                min-height: 40px;
            }}
            QPushButton#create_btn:hover {{
                background-color: #7c3aed;
            }}
        """
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(25)
        layout.setContentsMargins(40, 40, 40, 40)

        # Title - "Add Task" at the top
        title = QLabel("Add Task")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"""
            font-size: {base_font_size + 12}px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 20px;
        """
        )
        layout.addWidget(title)

        # Task Name section
        task_name_label = QLabel("Task Name")
        layout.addWidget(task_name_label)

        self.task_name_input = QLineEdit()
        self.task_name_input.setPlaceholderText("Type")
        layout.addWidget(self.task_name_input)

        # Priority section
        priority_label = QLabel("Priority")
        priority_label.setAlignment(Qt.AlignCenter)
        priority_label.setStyleSheet(
            f"""
            font-size: {base_font_size + 8}px;
            font-weight: bold;
            margin: 20px 0px 15px 0px;
        """
        )
        layout.addWidget(priority_label)

        # Priority buttons - exactly like screenshot
        priority_layout = QHBoxLayout()
        priority_layout.setSpacing(20)

        self.priority_group = QButtonGroup()

        # Easy button (Green)
        self.easy_btn = QPushButton("Easy")
        self.easy_btn.setObjectName("priority_btn")
        self.easy_btn.setCheckable(True)
        self.easy_btn.setStyleSheet(
            f"""
            QPushButton#priority_btn {{
                background-color: #2a2d42;
                border: 2px solid #4a4d62;
                border-radius: 12px;
                padding: 20px;
                font-size: {base_font_size + 4}px;
                font-weight: bold;
                min-height: 40px;
                min-width: 120px;
                color: #22c55e;
            }}
            QPushButton#priority_btn:checked {{
                border: 2px solid #22c55e;
                background-color: #16a34a20;
            }}
            QPushButton#priority_btn:hover {{
                background-color: #353851;
            }}
        """
        )
        self.priority_group.addButton(self.easy_btn)
        priority_layout.addWidget(self.easy_btn)

        # Medium button (Yellow)
        self.medium_btn = QPushButton("Medium")
        self.medium_btn.setObjectName("priority_btn")
        self.medium_btn.setCheckable(True)
        self.medium_btn.setStyleSheet(
            f"""
            QPushButton#priority_btn {{
                background-color: #2a2d42;
                border: 2px solid #4a4d62;
                border-radius: 12px;
                padding: 20px;
                font-size: {base_font_size + 4}px;
                font-weight: bold;
                min-height: 40px;
                min-width: 120px;
                color: #eab308;
            }}
            QPushButton#priority_btn:checked {{
                border: 2px solid #eab308;
                background-color: #ca8a0420;
            }}
            QPushButton#priority_btn:hover {{
                background-color: #353851;
            }}
        """
        )
        self.priority_group.addButton(self.medium_btn)
        priority_layout.addWidget(self.medium_btn)

        # High button (Red)
        self.high_btn = QPushButton("High")
        self.high_btn.setObjectName("priority_btn")
        self.high_btn.setCheckable(True)
        self.high_btn.setStyleSheet(
            f"""
            QPushButton#priority_btn {{
                background-color: #2a2d42;
                border: 2px solid #4a4d62;
                border-radius: 12px;
                padding: 20px;
                font-size: {base_font_size + 4}px;
                font-weight: bold;
                min-height: 40px;
                min-width: 120px;
                color: #ef4444;
            }}
            QPushButton#priority_btn:checked {{
                border: 2px solid #ef4444;
                background-color: #dc262620;
            }}
            QPushButton#priority_btn:hover {{
                background-color: #353851;
            }}
        """
        )
        self.priority_group.addButton(self.high_btn)
        priority_layout.addWidget(self.high_btn)

        layout.addLayout(priority_layout)

        # Add spacer
        layout.addStretch()

        # CREATE button - exactly like screenshot
        self.create_btn = QPushButton("CREATE")
        self.create_btn.setObjectName("create_btn")
        self.create_btn.clicked.connect(self.accept)
        layout.addWidget(self.create_btn)

        # Set default selection
        self.medium_btn.setChecked(True)

    def get_task_data(self):
        """Get the task data from the dialog"""
        priority = "Medium"  # default
        if self.easy_btn.isChecked():
            priority = "Low"
        elif self.medium_btn.isChecked():
            priority = "Medium"
        elif self.high_btn.isChecked():
            priority = "High"

        return {
            "title": self.task_name_input.text().strip(),
            "description": "",
            "priority": priority,
            "app_assigned": None,
        }


class TaskListWidget(QFrame):
    """Widget to display and manage tasks"""

    def __init__(self, task_manager):
        super().__init__()
        self.task_manager = task_manager
        self.setup_ui()

    def setup_ui(self):
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px;
            }
        """
        )

        layout = QVBoxLayout(self)

        # Header with add button
        header_layout = QHBoxLayout()
        header = QLabel("📋 Tasks & Goals")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")

        add_btn = QPushButton("+ Add Task")
        add_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """
        )
        add_btn.clicked.connect(self.add_task)

        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(add_btn)
        layout.addLayout(header_layout)

        # Task list
        self.task_list = QListWidget()
        self.task_list.setStyleSheet(
            """
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: #f8f9fa;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
        """
        )
        # task stats display
        self.stats_label = QLabel("Loading tasks...")
        self.stats_label.setFont(QFont("Arial", 10))
        self.stats_label.setStyleSheet("color: #7f8c8d; margin-top: 5px;")
        layout.addWidget(self.stats_label)

        self.refresh_tasks()

    def add_task(self):
        # show the add task dialog when user clicks add
        dialog = AddTaskDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            task_data = dialog.get_task_data()
            if task_data["title"]:
                self.task_manager.create_task(
                    task_data["title"],
                    task_data["description"],
                    task_data.get("app_assigned", ""),
                    task_data.get("priority", "Medium"),
                )
                self.refresh_tasks()

    def refresh_tasks(self):
        # update the task list display
        self.task_list.clear()

        active_tasks = self.task_manager.get_active_tasks()

        for task in active_tasks:
            # add some emojis to make it look nicer
            priority_icon = (
                "🔥"
                if task["priority"] == "High"
                else "⚡" if task["priority"] == "Medium" else "📝"
            )
            app_text = f" → {task['app_assigned']}" if task["app_assigned"] else ""

            item_text = f"{priority_icon} {task['title']}{app_text}"

            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, task["id"])

            # different colors for different priorities
            if task["priority"] == "High":
                item.setBackground(QColor("#fee"))
            elif task["priority"] == "Medium":
                item.setBackground(QColor("#fff3cd"))

            self.task_list.addItem(item)

        # Update stats
        stats = self.task_manager.get_task_stats()
        self.stats_label.setText(
            f"Active: {stats['active']} | Completed Today: {stats['today_completed']} | "
            f"Total Progress: {stats['completion_rate']:.1f}%"
        )


class DistractionWidget(QFrame):
    """Widget to display AI distraction predictions"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border: 2px solid #bb6bd9;
                border-radius: 15px;
                padding: 15px;
                box-shadow: 0 4px 15px rgba(155, 89, 182, 0.2);
            }
        """
        )

        layout = QVBoxLayout(self)

        # Header
        header = QLabel("🤖 AI Distraction Prediction")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setStyleSheet("color: #9b59b6; margin-bottom: 10px;")
        layout.addWidget(header)

        # Prediction display
        self.prediction_label = QLabel("Analyzing your focus patterns...")
        self.prediction_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.prediction_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prediction_label)

        # Progress bar for distraction probability
        self.distraction_bar = QProgressBar()
        self.distraction_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #e74c3c;
                border-radius: 6px;
            }
        """
        )
        layout.addWidget(self.distraction_bar)

        # Reason
        self.reason_label = QLabel("Gathering data...")
        self.reason_label.setFont(QFont("Arial", 10))
        self.reason_label.setStyleSheet("color: #7f8c8d; margin: 5px;")
        self.reason_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.reason_label)

        # ML Status
        ml_status = (
            "✅ Machine Learning Active"
            if ML_AVAILABLE
            else "⚠️ ML Disabled (install scikit-learn)"
        )
        ml_label = QLabel(ml_status)
        ml_label.setFont(QFont("Arial", 9))
        ml_label.setStyleSheet("color: #95a5a6; margin: 5px;")
        ml_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(ml_label)

    def update_prediction(self, probability, reason):
        """Update distraction prediction display"""
        percentage = int(probability * 100)

        if percentage < 30:
            status = "🟢 Focused"
            color = "#27ae60"
        elif percentage < 60:
            status = "🟡 Moderate Risk"
            color = "#f39c12"
        else:
            status = "🔴 High Distraction Risk"
            color = "#e74c3c"

        self.prediction_label.setText(f"{status} ({percentage}%)")
        self.prediction_label.setStyleSheet(f"color: {color}; margin: 10px;")

        self.distraction_bar.setValue(percentage)
        self.distraction_bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                height: 25px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 6px;
            }}
        """
        )

        self.reason_label.setText(f"Primary factor: {reason}")


class AllAppsWidget(QFrame):
    # widget that shows all the apps currently running on the system

    def __init__(self):
        super().__init__()
        self.apps = []
        self.setup_ui()

    def setup_ui(self):
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px;
            }
        """
        )

        layout = QVBoxLayout(self)

        # header with some basic controls
        header_layout = QHBoxLayout()
        header = QLabel("🖥️ ALL Running Applications")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setStyleSheet("color: #e74c3c; margin-bottom: 10px;")

        # Category filter
        self.category_filter = QComboBox()
        self.category_filter.addItems(
            [
                "All",
                "Productivity",
                "Browser",
                "Communication",
                "Entertainment",
                "Gaming",
                "System",
                "Other",
            ]
        )
        self.category_filter.currentTextChanged.connect(self.filter_apps)

        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Filter:"))
        header_layout.addWidget(self.category_filter)
        layout.addLayout(header_layout)

        # Apps list with better readability
        self.apps_list = QListWidget()
        self.apps_list.setStyleSheet(
            """
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: #f8f9fa;
                font-size: 14px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-weight: 500;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #ecf0f1;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
                border-radius: 4px;
            }
            QListWidget::item:hover {
                background-color: #e8f4f8;
                border-radius: 4px;
            }
        """
        )
        layout.addWidget(self.apps_list)

        # Stats
        self.stats_label = QLabel("Scanning for applications...")
        self.stats_label.setFont(QFont("Arial", 10))
        self.stats_label.setStyleSheet("color: #7f8c8d; margin-top: 5px;")
        layout.addWidget(self.stats_label)

    def update_apps(self, apps):
        """Update the list of ALL running applications"""
        self.apps = apps if apps else self.generate_fake_apps()
        self.filter_apps()

    def generate_fake_apps(self):
        """Generate fake app data for demo purposes"""
        fake_apps = [
            {
                "name": "Google Chrome",
                "memory_mb": 245.6,
                "cpu_percent": 8.3,
                "is_active": True,
                "category": "Browser",
                "runtime_minutes": 45,
            },
            {
                "name": "Visual Studio Code",
                "memory_mb": 189.2,
                "cpu_percent": 12.1,
                "is_active": False,
                "category": "Productivity",
                "runtime_minutes": 120,
            },
            {
                "name": "Discord",
                "memory_mb": 156.8,
                "cpu_percent": 3.2,
                "is_active": False,
                "category": "Communication",
                "runtime_minutes": 30,
            },
            {
                "name": "Spotify",
                "memory_mb": 98.4,
                "cpu_percent": 1.8,
                "is_active": False,
                "category": "Entertainment",
                "runtime_minutes": 85,
            },
            {
                "name": "Microsoft Word",
                "memory_mb": 87.3,
                "cpu_percent": 2.1,
                "is_active": False,
                "category": "Productivity",
                "runtime_minutes": 25,
            },
            {
                "name": "Steam",
                "memory_mb": 67.9,
                "cpu_percent": 0.5,
                "is_active": False,
                "category": "Gaming",
                "runtime_minutes": 10,
            },
            {
                "name": "WhatsApp",
                "memory_mb": 52.1,
                "cpu_percent": 1.2,
                "is_active": False,
                "category": "Communication",
                "runtime_minutes": 15,
            },
            {
                "name": "YouTube Music",
                "memory_mb": 48.7,
                "cpu_percent": 2.8,
                "is_active": False,
                "category": "Entertainment",
                "runtime_minutes": 60,
            },
            {
                "name": "Figma",
                "memory_mb": 156.3,
                "cpu_percent": 5.4,
                "is_active": False,
                "category": "Productivity",
                "runtime_minutes": 90,
            },
            {
                "name": "Adobe Photoshop",
                "memory_mb": 312.8,
                "cpu_percent": 15.2,
                "is_active": False,
                "category": "Productivity",
                "runtime_minutes": 35,
            },
            {
                "name": "Netflix",
                "memory_mb": 78.9,
                "cpu_percent": 3.1,
                "is_active": False,
                "category": "Entertainment",
                "runtime_minutes": 40,
            },
            {
                "name": "Microsoft Teams",
                "memory_mb": 134.5,
                "cpu_percent": 4.7,
                "is_active": False,
                "category": "Communication",
                "runtime_minutes": 55,
            },
            {
                "name": "PyCharm",
                "memory_mb": 198.7,
                "cpu_percent": 6.8,
                "is_active": False,
                "category": "Productivity",
                "runtime_minutes": 75,
            },
            {
                "name": "Firefox",
                "memory_mb": 167.4,
                "cpu_percent": 5.9,
                "is_active": False,
                "category": "Browser",
                "runtime_minutes": 20,
            },
            {
                "name": "Blender",
                "memory_mb": 456.2,
                "cpu_percent": 18.5,
                "is_active": False,
                "category": "Productivity",
                "runtime_minutes": 65,
            },
        ]
        return fake_apps

    def filter_apps(self):
        """Filter apps by category"""
        self.apps_list.clear()

        if not self.apps:
            return

        filter_category = self.category_filter.currentText()

        filtered_apps = self.apps
        if filter_category != "All":
            filtered_apps = [
                app
                for app in self.apps
                if app.get("category", "Other") == filter_category
            ]

        active_count = 0
        total_memory = 0
        total_cpu = 0

        for i, app in enumerate(filtered_apps):
            name = app["name"].replace(".exe", "")
            memory = app["memory_mb"]
            cpu = app["cpu_percent"]
            category = app.get("category", "Other")
            runtime = app.get("runtime_minutes", 0)

            # Format the display text with more info
            status = "🟢 ACTIVE" if app["is_active"] else "⚫"

            # Format runtime
            if runtime < 60:
                runtime_str = f"{runtime:.0f}m"
            else:
                hours = int(runtime // 60)
                mins = int(runtime % 60)
                runtime_str = f"{hours}h{mins}m"

            item_text = f"{status} {name:<20} | {category:<12} | CPU:{cpu:5.1f}% | RAM:{memory:6.1f}MB | Up:{runtime_str}"

            item = QListWidgetItem(item_text)

            # Color coding
            if app["is_active"]:
                item.setBackground(QColor("#e8f5e8"))
                active_count += 1
            elif cpu > 10:
                item.setBackground(QColor("#fff3cd"))
            elif memory > 100:
                item.setBackground(QColor("#f8d7da"))

            self.apps_list.addItem(item)
            total_memory += memory
            total_cpu += cpu

        # Update stats
        avg_cpu = total_cpu / len(filtered_apps) if filtered_apps else 0
        self.stats_label.setText(
            f"Showing: {len(filtered_apps)} apps | Active: {active_count} | "
            f"Total RAM: {total_memory:.0f}MB | Avg CPU: {avg_cpu:.1f}%"
        )


class StatsCard(QFrame):
    """Simple stats card widget"""

    def __init__(self, title, value, subtitle="", color="#3498db"):
        super().__init__()
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.color = color
        self.setup_ui()

    def setup_ui(self):
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: #ffffff;
                border: 2px solid #e0e6ed;
                border-radius: 15px;
                padding: 18px;
                margin: 8px;
            }}
            QFrame:hover {{
                border: 2px solid {self.color};
                background-color: {self.color}10;
            }}
        """
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(5)

        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title_label.setStyleSheet("color: #34495e; margin-bottom: 5px;")

        # Value
        self.value_label = QLabel(str(self.value))
        self.value_label.setFont(QFont("Segoe UI", 26, QFont.Bold))
        self.value_label.setStyleSheet(
            f"""
            color: {self.color}; 
            margin: 8px 0px;
        """
        )

        # Subtitle
        if self.subtitle:
            subtitle_label = QLabel(self.subtitle)
            subtitle_label.setFont(QFont("Segoe UI", 11))
            subtitle_label.setStyleSheet("color: #7f8c8d; margin-top: 2px;")
            layout.addWidget(subtitle_label)

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def update_value(self, new_value, new_subtitle=""):
        """Update the displayed value"""
        self.value_label.setText(str(new_value))


class RunningAppsWidget(QFrame):
    """Widget to display running applications"""

    def __init__(self):
        super().__init__()
        self.apps = []
        self.setup_ui()

    def setup_ui(self):
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px;
            }
        """
        )

        layout = QVBoxLayout(self)

        # Header
        header = QLabel("🔴 LIVE: Running Applications")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setStyleSheet("color: #e74c3c; margin-bottom: 10px;")
        layout.addWidget(header)

        # Apps list with better readability
        self.apps_list = QListWidget()
        self.apps_list.setStyleSheet(
            """
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: #f8f9fa;
                font-size: 14px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-weight: 500;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #ecf0f1;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
                border-radius: 4px;
            }
            QListWidget::item:hover {
                background-color: #e8f4f8;
                border-radius: 4px;
            }
        """
        )
        layout.addWidget(self.apps_list)

        # Stats
        self.stats_label = QLabel("Scanning for applications...")
        self.stats_label.setFont(QFont("Arial", 10))
        self.stats_label.setStyleSheet("color: #7f8c8d; margin-top: 5px;")
        layout.addWidget(self.stats_label)

    def update_apps(self, apps):
        """Update the list of running applications"""
        self.apps = apps if apps else self.generate_fake_apps()
        self.apps_list.clear()

        active_count = 0
        total_memory = 0

        for app in self.apps:
            # Create list item
            name = app["name"].replace(".exe", "")
            memory = app["memory_mb"]
            cpu = app["cpu_percent"]

            # Format the display text
            status = "🟢 ACTIVE" if app["is_active"] else "⚫"
            item_text = f"{status} {name} | CPU: {cpu:.1f}% | RAM: {memory:.1f}MB"

            item = QListWidgetItem(item_text)

            # Color coding
            if app["is_active"]:
                item.setBackground(QColor("#e8f5e8"))
                active_count += 1
            elif cpu > 10:
                item.setBackground(QColor("#fff3cd"))

            self.apps_list.addItem(item)
            total_memory += memory

        # Update stats
        self.stats_label.setText(
            f"Total: {len(self.apps)} apps | Active: {active_count} | "
            f"Total RAM: {total_memory:.1f}MB"
        )

    def generate_fake_apps(self):
        """Generate fake app data for demo purposes"""
        fake_apps = [
            {
                "name": "Google Chrome",
                "memory_mb": 245.6,
                "cpu_percent": 8.3,
                "is_active": True,
            },
            {
                "name": "Visual Studio Code",
                "memory_mb": 189.2,
                "cpu_percent": 12.1,
                "is_active": False,
            },
            {
                "name": "Discord",
                "memory_mb": 156.8,
                "cpu_percent": 3.2,
                "is_active": False,
            },
            {
                "name": "Spotify",
                "memory_mb": 98.4,
                "cpu_percent": 1.8,
                "is_active": False,
            },
            {
                "name": "Microsoft Word",
                "memory_mb": 87.3,
                "cpu_percent": 2.1,
                "is_active": False,
            },
            {
                "name": "Steam",
                "memory_mb": 67.9,
                "cpu_percent": 0.5,
                "is_active": False,
            },
            {
                "name": "Microsoft Teams",
                "memory_mb": 134.5,
                "cpu_percent": 4.7,
                "is_active": False,
            },
        ]
        return fake_apps


class ActivityFeedWidget(QFrame):
    """Widget to show recent activity"""

    def __init__(self):
        super().__init__()
        self.activity_log = []
        self.setup_ui()

    def setup_ui(self):
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border: 2px solid #4caf50;
                border-radius: 15px;
                padding: 15px;
            }
        """
        )

        layout = QVBoxLayout(self)

        # Header
        header = QLabel("📝 Activity Feed")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(header)

        # Activity log
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(200)
        self.log_display.setStyleSheet(
            """
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: #f8f9fa;
                font-family: 'Courier New';
                font-size: 10px;
            }
        """
        )
        layout.addWidget(self.log_display)

    def add_activity(self, message):
        """Add new activity to the feed"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        self.activity_log.append(formatted_message)

        # Keep only last 100 entries
        if len(self.activity_log) > 100:
            self.activity_log = self.activity_log[-100:]

        # Update display
        self.log_display.setPlainText(
            "\n".join(self.activity_log[-20:])
        )  # Show last 20

        # Scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class ProcastiNOMainWindow(QMainWindow):
    # main window for the ProcastiNO app - this is where everything happens

    def __init__(self):
        super().__init__()
        self.app_tracker = None
        self.task_manager = (
            TaskManager() if MODULES_AVAILABLE else FallbackTaskManager()
        )
        self.last_active_app = None
        self.session_start = datetime.now()
        self.current_apps = []  # keep track of what apps are running

        # timer for updating screen time every second
        self.screen_time_timer = QTimer()
        self.screen_time_timer.timeout.connect(self.update_screen_time)
        self.screen_time_timer.start(1000)

        # try to load data from previous sessions
        self.load_session_data()

        self.setup_ui()
        self.start_tracking()

        # save info about this session
        self.save_session_data()

    def load_session_data(self):
        # load data from the last time the app was used
        try:
            if os.path.exists("session_data.json"):
                with open("session_data.json", "r") as f:
                    data = json.load(f)
                    self.last_session_time = data.get("last_run", "Never")
                    self.total_sessions = data.get("total_sessions", 0)
                    print(f"[ProcastiNO] Last session: {self.last_session_time}")
            else:
                self.last_session_time = "Never"
                self.total_sessions = 0
        except Exception as e:
            print(f"[ProcastiNO] Error loading session data: {e}")
            self.last_session_time = "Never"
            self.total_sessions = 0

    def save_session_data(self):
        # save current session info to file
        try:
            data = {
                "last_run": self.session_start.strftime("%Y-%m-%d %H:%M:%S"),
                "total_sessions": self.total_sessions + 1,
            }
            with open("session_data.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ProcastiNO] Error saving session data: {e}")

    def setup_ui(self):
        self.setWindowTitle("ProcastiNO - Complete Productivity Tracker")
        self.setGeometry(50, 50, 1400, 900)

        # Get screen size for relative font sizing
        screen = QApplication.primaryScreen().geometry()
        self.base_font_size = max(
            12, screen.width() // 120
        )  # Store for use throughout app

        # Modern dark theme styling with relative fonts
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: #2c2f36;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QWidget {{
                background-color: #2c2f36;
                color: #ffffff;
            }}
            QFrame#sidebar {{
                background-color: #23262d;
                border-right: 1px solid #3a3d44;
            }}
            QFrame#content {{
                background-color: #2c2f36;
            }}
            QPushButton#nav_button {{
                background-color: transparent;
                border: none;
                color: #b0b3b8;
                text-align: left;
                padding: 15px 20px;
                font-size: {self.base_font_size + 2}px;
                font-weight: 500;
                border-radius: 0px;
            }}
            QPushButton#nav_button:hover {{
                background-color: #3a3d44;
                color: #ffffff;
            }}
            QPushButton#nav_button:pressed, QPushButton#nav_button:checked {{
                background-color: #4a5568;
                color: #ffffff;
                border-left: 3px solid #e53e3e;
            }}
            QLabel#title {{
                color: #ffffff;
                font-size: {self.base_font_size + 16}px;
                font-weight: bold;
                text-align: center;
                padding: 20px;
                background-color: #23262d;
                border-bottom: 1px solid #3a3d44;
            }}
            QLabel#page_title {{
                color: #ffffff;
                font-size: {self.base_font_size + 12}px;
                font-weight: bold;
                margin: 20px 0px;
            }}
            QLabel#subtitle {{
                color: #b0b3b8;
                font-size: {self.base_font_size + 4}px;
                margin-bottom: 30px;
            }}
            QListWidget {{
                background-color: #3a3d44;
                border: 1px solid #4a5568;
                border-radius: 8px;
                color: #ffffff;
                font-size: {self.base_font_size + 2}px;
                padding: 15px;
            }}
            QListWidget::item {{
                padding: 18px 15px;
                border-bottom: 1px solid #4a5568;
                border-radius: 4px;
                margin: 4px 0px;
                background-color: rgba(74, 85, 104, 0.3);
            }}
            QListWidget::item:selected {{
                background-color: #4a5568;
                color: #ffffff;
            }}
            QListWidget::item:hover {{
                background-color: #555a66;
            }}
        """
        )

        # Create main container
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create sidebar
        self.create_sidebar(main_layout)

        # Create main content area
        self.create_main_content(main_layout)

        # Set initial page
        self.current_page = "Dashboard"
        self.show_dashboard()

    def create_sidebar(self, main_layout):
        """Create the sidebar navigation"""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(300)
        sidebar.setFrameStyle(QFrame.NoFrame)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Title/Logo
        title_label = QLabel()
        title_label.setText(
            '<span style="color: #ffffff;">Procasti</span><span style="color: #e53e3e;">NO</span>'
        )
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        # Add the red "NO" styling with relative font size
        title_label.setStyleSheet(
            f"""
            QLabel#title {{
                font-size: {self.base_font_size + 12}px;
                font-weight: bold;
                text-align: center;
                padding: 20px;
                background-color: #23262d;
                border-bottom: 1px solid #3a3d44;
                min-width: 258px;
            }}
        """
        )
        sidebar_layout.addWidget(title_label)

        # Navigation buttons
        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 20, 0, 0)
        nav_layout.setSpacing(5)

        # Navigation items
        nav_items = [
            ("Dashboard", "📊"),
            ("ALL APPS", "💻"),
            ("Tasks", "📋"),
            ("Analytics", "📈"),
            ("Settings", "⚙️"),
        ]

        self.nav_buttons = {}
        for name, icon in nav_items:
            btn = QPushButton(f"  {name}")
            btn.setObjectName("nav_button")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, page=name: self.switch_page(page))
            btn.setStyleSheet(
                f"""
                QPushButton#nav_button {{
                    background-color: transparent;
                    border: none;
                    color: #b0b3b8;
                    text-align: left;
                    padding: 15px 20px;
                    font-size: {self.base_font_size + 2}px;
                    font-weight: 500;
                    border-radius: 0px;
                }}
                QPushButton#nav_button:hover {{
                    background-color: #3a3d44;
                    color: #ffffff;
                }}
                QPushButton#nav_button:pressed, QPushButton#nav_button:checked {{
                    background-color: #4a5568;
                    color: #ffffff;
                    border-left: 3px solid #e53e3e;
                }}
            """
            )
            nav_layout.addWidget(btn)
            self.nav_buttons[name] = btn

        nav_layout.addStretch()
        sidebar_layout.addWidget(nav_container)

        main_layout.addWidget(sidebar)

    def create_main_content(self, main_layout):
        """Create the main content area"""
        self.content_frame = QFrame()
        self.content_frame.setObjectName("content")

        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setContentsMargins(40, 30, 40, 30)
        self.content_layout.setSpacing(20)

        main_layout.addWidget(self.content_frame, 1)

    def switch_page(self, page_name):
        """Switch to a different page"""
        # Update button states
        for name, btn in self.nav_buttons.items():
            btn.setChecked(name == page_name)

        # Clear current content more thoroughly
        self.clear_content_layout()

        self.current_page = page_name

        # Show the requested page
        if page_name == "Dashboard":
            self.show_dashboard()
        elif page_name == "ALL APPS":
            self.show_all_apps()
        elif page_name == "Tasks":
            self.show_tasks()
        elif page_name == "Analytics":
            self.show_analytics()
        elif page_name == "Settings":
            self.show_settings()

    def clear_content_layout(self):
        """Thoroughly clear all content from the layout"""
        # First, clear any existing dashboard stat cards references
        if hasattr(self, "dashboard_stat_cards"):
            for card in self.dashboard_stat_cards:
                if card:
                    card.setParent(None)
                    card.deleteLater()
            self.dashboard_stat_cards = []

        # Clear dashboard widget references to prevent access errors
        if hasattr(self, "dashboard_tasks_list"):
            self.dashboard_tasks_list = None
        if hasattr(self, "dashboard_apps_list"):
            self.dashboard_apps_list = None

        # Clear all widgets and layouts recursively
        def clear_layout(layout):
            if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    if item is not None:
                        widget = item.widget()
                        if widget is not None:
                            widget.setParent(None)
                            widget.deleteLater()
                        else:
                            # If it's a sub-layout, clear it recursively
                            child_layout = item.layout()
                            if child_layout is not None:
                                clear_layout(child_layout)

        clear_layout(self.content_layout)

        # Force update the UI to ensure everything is cleared
        QApplication.processEvents()

    def show_dashboard(self):
        """Show the dashboard page exactly like the screenshot - 2x2 grid layout"""
        # Clear any existing widget references
        if hasattr(self, "dashboard_stat_cards"):
            self.dashboard_stat_cards = []
        if hasattr(self, "activity_list"):
            self.activity_list = None

        # Create the main grid layout for 2x2 dashboard
        main_grid = QGridLayout()
        main_grid.setSpacing(15)  # Consistent spacing between sections
        main_grid.setContentsMargins(20, 20, 20, 20)

        # Row 1: Applications Running (0,0) and Tasks (0,1)
        apps_column = self.create_dashboard_apps_column()
        tasks_column = self.create_dashboard_tasks_column()

        main_grid.addWidget(apps_column, 0, 0)
        main_grid.addWidget(tasks_column, 0, 1)

        # Row 2: Distraction Predictor (1,0) and Screen Time (1,1)
        distraction_column = self.create_dashboard_distraction_column()
        screentime_column = self.create_dashboard_screentime_column()

        main_grid.addWidget(distraction_column, 1, 0)
        main_grid.addWidget(screentime_column, 1, 1)

        # Set responsive column stretch ratios (inspired by responsive UI)
        main_grid.setColumnStretch(0, 2)  # Apps section gets more space
        main_grid.setColumnStretch(1, 2)  # Tasks section gets equal space
        main_grid.setRowStretch(0, 1)
        main_grid.setRowStretch(1, 1)

        # Add the grid layout to content and make it expand to fill all space
        self.content_layout.addLayout(main_grid)
        self.content_layout.setStretchFactor(main_grid, 1)

    def create_dashboard_apps_column(self):
        """Create the APPLICATIONS RUNNING column with modern responsive design"""
        column = QFrame()
        column.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(42, 42, 61, 0.8), 
                    stop:1 rgba(26, 26, 42, 0.9));
                border: 1px solid rgba(124, 77, 255, 0.2);
                border-radius: 16px;
                margin: 8px;
            }
        """
        )

        layout = QVBoxLayout(column)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        # Title with modern styling
        title = QLabel("APPLICATIONS\nRUNNING")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"""
            QLabel {{
                color: #FFFFFF;
                font-size: {self.base_font_size + 8}px;
                font-weight: bold;
                background: transparent;
                border: none;
                margin-bottom: 10px;
            }}
        """
        )
        layout.addWidget(title)

        # Running count with modern styling
        self.dashboard_app_count_label = QLabel(f"RUNNING: 0")
        self.dashboard_app_count_label.setStyleSheet(
            f"""
            QLabel {{
                color: #A0A0A0;
                font-size: {self.base_font_size}px;
                background: transparent;
                border: none;
                margin-bottom: 10px;
            }}
        """
        )
        layout.addWidget(self.dashboard_app_count_label)

        # Content area with modern styling
        content_area = QFrame()
        content_area.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 30, 46, 0.9), 
                    stop:1 rgba(26, 26, 42, 0.9));
                border: 1px solid rgba(124, 77, 255, 0.15);
                border-radius: 12px;
                padding: 15px;
            }
        """
        )
        content_layout = QVBoxLayout(content_area)

        # Apps list with modern styling
        self.dashboard_apps_list = QListWidget()
        self.dashboard_apps_list.setStyleSheet(
            f"""
            QListWidget {{
                background: transparent;
                border: none;
                color: #ffffff;
                font-size: {self.base_font_size}px;
                padding: 8px;
            }}
            QListWidget::item {{
                padding: 15px 12px;
                border-bottom: 1px solid rgba(74, 74, 93, 0.5);
                border-radius: 4px;
                margin: 4px 0px;
                background: transparent;
            }}
            QListWidget::item:hover {{
                background: rgba(124, 77, 255, 0.2);
            }}
        """
        )

        # Initial population
        self.update_dashboard_apps_display()
        content_layout.addWidget(self.dashboard_apps_list)
        layout.addWidget(content_area)
        return column

    def update_dashboard_apps_display(self):
        """Update the dashboard apps list with current data"""
        if (
            not hasattr(self, "dashboard_apps_list")
            or self.dashboard_apps_list is None
            or not hasattr(self, "dashboard_app_count_label")
            or self.dashboard_app_count_label is None
        ):
            return

        try:
            # Update count
            app_count = len(self.current_apps) if hasattr(self, "current_apps") else 0
            self.dashboard_app_count_label.setText(f"RUNNING: {app_count}")

            # Update list
            self.dashboard_apps_list.clear()

            if hasattr(self, "current_apps") and self.current_apps:
                for app in self.current_apps[:5]:  # Show top 5
                    name = app.get("name", "Unknown").replace(".exe", "")
                    status = "🟢" if app.get("is_active", False) else "⚫"
                    self.dashboard_apps_list.addItem(f"{status} {name}")
            else:
                none_item = QListWidgetItem("None")
                none_item.setTextAlignment(Qt.AlignCenter)
                self.dashboard_apps_list.addItem(none_item)
        except (RuntimeError, AttributeError):
            # Widget has been deleted or is None, ignore the update
            return

    def create_dashboard_tasks_column(self):
        """Create the TASKS column with modern responsive design"""
        column = QFrame()
        column.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(42, 42, 61, 0.8), 
                    stop:1 rgba(26, 26, 42, 0.9));
                border: 1px solid rgba(124, 77, 255, 0.2);
                border-radius: 16px;
                margin: 8px;
            }
        """
        )

        layout = QVBoxLayout(column)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        # Title with modern styling
        title = QLabel("TASKS")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"""
            QLabel {{
                color: #FFFFFF;
                font-size: {self.base_font_size + 8}px;
                font-weight: bold;
                background: transparent;
                border: none;
                margin-bottom: 10px;
            }}
        """
        )
        layout.addWidget(title)

        # Task stats with modern styling
        task_stats = (
            self.task_manager.get_task_statistics()
            if self.task_manager
            else {
                "active_tasks": 0,
                "completed_tasks": 0,
                "total_tasks": 0,
                "today_completed": 0,
                "completion_rate": 0,
            }
        )
        stats_label = QLabel(f"Total Tasks Today: {task_stats.get('active_tasks', 0)}")
        stats_label.setAlignment(Qt.AlignCenter)
        stats_label.setStyleSheet(
            f"""
            QLabel {{
                color: #A0A0A0;
                font-size: {self.base_font_size}px;
                background: transparent;
                border: none;
                margin-bottom: 15px;
            }}
        """
        )
        layout.addWidget(stats_label)

        # Modern New Task button (inspired by responsive UI)
        new_task_btn = QPushButton("New Task")
        new_task_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #7C4DFF, stop:1 #5C34CC);
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: bold;
                font-size: {self.base_font_size}px;
                padding: 15px 25px;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #8A5AFF, stop:1 #6A42DA);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6A42DA, stop:1 #4A28AA);
            }}
        """
        )
        new_task_btn.clicked.connect(self.add_task)
        layout.addWidget(new_task_btn)

        # Content area with modern styling
        content_area = QFrame()
        content_area.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 30, 46, 0.9), 
                    stop:1 rgba(26, 26, 42, 0.9));
                border: 1px solid rgba(124, 77, 255, 0.15);
                border-radius: 12px;
                padding: 15px;
            }
        """
        )
        content_layout = QVBoxLayout(content_area)

        # Tasks list with modern styling
        self.dashboard_tasks_list = QListWidget()
        self.dashboard_tasks_list.setStyleSheet(
            f"""
            QListWidget {{
                background: transparent;
                border: none;
                color: #ffffff;
                font-size: {self.base_font_size}px;
                padding: 8px;
            }}
            QListWidget::item {{
                padding: 15px 12px;
                border-bottom: 1px solid rgba(74, 74, 93, 0.5);
                border-radius: 4px;
                margin: 4px 0px;
                background: transparent;
            }}
            QListWidget::item:hover {{
                background: rgba(124, 77, 255, 0.2);
            }}
        """
        )

        # Populate tasks
        self.populate_dashboard_tasks()
        content_layout.addWidget(self.dashboard_tasks_list)
        layout.addWidget(content_area)

        return column

    def create_dashboard_distraction_column(self):
        # create the AI distraction prediction section
        column = QFrame()
        column.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(42, 42, 61, 0.8), 
                    stop:1 rgba(26, 26, 42, 0.9));
                border: 1px solid rgba(124, 77, 255, 0.2);
                border-radius: 16px;
                margin: 8px;
            }
        """
        )

        layout = QVBoxLayout(column)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        # title for this section
        title = QLabel("DISTRACTION\nPREDICTOR")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"""
            QLabel {{
                color: #FFFFFF;
                font-size: {self.base_font_size + 8}px;
                font-weight: bold;
                background: transparent;
                border: none;
                margin-bottom: 20px;
            }}
        """
        )
        layout.addWidget(title)

        # area where prediction results show up
        prediction_area = QFrame()
        prediction_area.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 30, 46, 0.9), 
                    stop:1 rgba(26, 26, 42, 0.9));
                border: 1px solid rgba(124, 77, 255, 0.15);
                border-radius: 12px;
                padding: 20px;
            }
        """
        )

        pred_layout = QVBoxLayout(prediction_area)
        pred_layout.setAlignment(Qt.AlignCenter)

        # Prediction text with modern styling
        prediction_label = QLabel("Needs more data")
        prediction_label.setAlignment(Qt.AlignCenter)
        prediction_label.setStyleSheet(
            f"""
            QLabel {{
                color: #A0A0A0;
                font-size: {self.base_font_size + 2}px;
                font-weight: 500;
                background: transparent;
                border: none;
            }}
        """
        )
        pred_layout.addWidget(prediction_label)

        layout.addWidget(prediction_area)
        layout.addStretch()

        return column

    def create_dashboard_screentime_column(self):
        """Create the SCREEN TIME column with modern responsive design"""
        column = QFrame()
        column.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(42, 42, 61, 0.8), 
                    stop:1 rgba(26, 26, 42, 0.9));
                border: 1px solid rgba(124, 77, 255, 0.2);
                border-radius: 16px;
                margin: 8px;
            }
        """
        )

        layout = QVBoxLayout(column)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        # screen time section title
        title = QLabel("SCREEN TIME")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"""
            QLabel {{
                color: #FFFFFF;
                font-size: {self.base_font_size + 8}px;
                font-weight: bold;
                background: transparent;
                border: none;
                margin-bottom: 30px;
            }}
        """
        )
        layout.addWidget(title)

        # main content area
        content_area = QFrame()
        content_area.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 30, 46, 0.9), 
                    stop:1 rgba(26, 26, 42, 0.9));
                border: 1px solid rgba(124, 77, 255, 0.15);
                border-radius: 12px;
                padding: 20px;
            }
        """
        )
        content_layout = QVBoxLayout(content_area)
        content_layout.setAlignment(Qt.AlignCenter)

        # the actual time display - shows in minutes:seconds
        self.screen_time_display = QLabel("00:00")
        self.screen_time_display.setAlignment(Qt.AlignCenter)
        self.screen_time_display.setStyleSheet(
            f"""
            QLabel {{
                color: #FFFFFF;
                font-size: {self.base_font_size + 20}px;
                font-weight: bold;
                background: transparent;
                border: none;
                margin: 20px 0px;
            }}
        """
        )
        content_layout.addWidget(self.screen_time_display)

        # Label to show it's minutes:seconds
        time_label = QLabel("minutes : seconds")
        time_label.setAlignment(Qt.AlignCenter)
        time_label.setStyleSheet(
            f"""
            QLabel {{
                color: #A0A0A0;
                font-size: {self.base_font_size}px;
                background: transparent;
                border: none;
                margin-top: 10px;
            }}
        """
        )
        content_layout.addWidget(time_label)

        layout.addWidget(content_area)
        layout.addStretch()

        return column

    def update_screen_time(self):
        """Update the screen time display in minutes:seconds format"""
        try:
            if hasattr(self, "screen_time_display") and self.screen_time_display:
                current_time = datetime.now()
                elapsed = current_time - self.session_start

                # Format as MM:SS (total minutes and seconds)
                total_seconds = int(elapsed.total_seconds())
                minutes = total_seconds // 60
                seconds = total_seconds % 60

                time_str = f"{minutes:02d}:{seconds:02d}"
                self.screen_time_display.setText(time_str)
        except RuntimeError:
            # Widget has been deleted, ignore the update
            pass
        except Exception as e:
            # Any other error, just ignore silently
            pass

    def populate_dashboard_tasks(self):
        # update the task list on the dashboard
        # make sure the widget exists first
        if (
            not hasattr(self, "dashboard_tasks_list")
            or self.dashboard_tasks_list is None
        ):
            return

        try:
            self.dashboard_tasks_list.clear()
        except RuntimeError:
            # widget got deleted somehow, just ignore
            return

        active_tasks = self.task_manager.get_active_tasks()

        if active_tasks:
            for task in active_tasks[:3]:  # only show first 3 tasks
                # create custom widget for each task
                task_widget = QWidget()
                task_widget.setStyleSheet(
                    """
                    QWidget {
                        background: rgba(74, 85, 104, 0.5);
                        border: 1px solid #4a5568;
                        border-radius: 8px;
                        color: #ffffff;
                        min-height: 100px;
                    }
                    QWidget:hover {
                        background: rgba(124, 77, 255, 0.3);
                        border-color: #7C4DFF;
                    }
                """
                )
                task_layout = QHBoxLayout(task_widget)
                task_layout.setContentsMargins(15, 12, 15, 12)
                task_layout.setSpacing(15)

                # different icons for priority levels
                if task.priority == "High":
                    priority_icon = "🔥"
                elif task.priority == "Medium":
                    priority_icon = "⚡"
                else:
                    priority_icon = "📝"

                # task title with icon
                task_text = QLabel(f"{priority_icon} {task.title}")
                task_text.setStyleSheet(
                    f"""
                    color: #ffffff !important;
                    font-size: {self.base_font_size}px;
                    background: transparent;
                    border: none;
                """
                )
                task_layout.addWidget(task_text)

                # Complete button
                complete_btn = QPushButton("✓")
                complete_btn.setFixedSize(25, 25)
                complete_btn.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #38a169;
                        color: white;
                        border: none;
                        border-radius: 12px;
                        font-weight: bold;
                        font-size: 14px;
                    }
                    QPushButton:hover {
                        background-color: #2f855a;
                    }
                """
                )
                complete_btn.clicked.connect(
                    lambda checked, task_id=task.id: self.complete_task_from_dashboard(
                        task_id
                    )
                )
                task_layout.addWidget(complete_btn)

                # Add to list
                item = QListWidgetItem()
                item.setSizeHint(task_widget.sizeHint())
                self.dashboard_tasks_list.addItem(item)
                self.dashboard_tasks_list.setItemWidget(item, task_widget)
        else:
            none_item = QListWidgetItem("None")
            none_item.setTextAlignment(Qt.AlignCenter)
            self.dashboard_tasks_list.addItem(none_item)

    def complete_task_from_dashboard(self, task_id):
        """Complete a task from the dashboard"""
        success = self.task_manager.complete_task(task_id)
        if success:
            # Only refresh if dashboard tasks list exists and is valid
            if (
                hasattr(self, "dashboard_tasks_list")
                and self.dashboard_tasks_list is not None
            ):
                try:
                    self.populate_dashboard_tasks()  # Refresh the task list
                except RuntimeError:
                    # Widget has been deleted, ignore the update
                    pass
            print(f"Task {task_id} completed!")

    def complete_task_from_tasks_page(self, task_id):
        # complete task when user clicks the button on tasks page
        success = self.task_manager.complete_task(task_id)
        if success:
            self.populate_tasks_list()  # refresh the list
            print(f"Task {task_id} completed from Tasks page!")

    def show_all_apps(self):
        # show the page with all running applications
        # clear dashboard stuff first
        if hasattr(self, "dashboard_stat_cards"):
            self.dashboard_stat_cards = []

        # page header
        title = QLabel("APPLICATIONS\nRUNNING")
        title.setObjectName("page_title")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"""
            color: #ffffff;
            font-size: {self.base_font_size + 20}px;
            font-weight: bold;
            margin: 40px 0px;
            text-align: center;
        """
        )
        self.content_layout.addWidget(title)

        # show how many apps are running
        app_count = len(self.current_apps) if hasattr(self, "current_apps") else 0
        count_label = QLabel(f"RUNNING: {app_count}")
        count_label.setAlignment(Qt.AlignLeft)
        count_label.setStyleSheet(
            f"""
            color: #b0b3b8;
            font-size: {self.base_font_size + 4}px;
            font-weight: 500;
            margin-bottom: 20px;
        """
        )
        self.content_layout.addWidget(count_label)

        # list of all apps
        self.all_apps_list = QListWidget()
        self.all_apps_list.setStyleSheet(
            f"""
            QListWidget {{
                background-color: #3a3d44;
                border: 1px solid #4a5568;
                border-radius: 8px;
                color: #ffffff;
                font-size: {self.base_font_size + 2}px;
                padding: 15px;
                min-height: 400px;
            }}
            QListWidget::item {{
                padding: 15px;
                border-bottom: 1px solid #4a5568;
                border-radius: 4px;
                margin: 3px 0px;
            }}
            QListWidget::item:selected {{
                background-color: #4a5568;
                color: #ffffff;
            }}
            QListWidget::item:hover {{
                background-color: #555a66;
            }}
        """
        )

        self.populate_apps_list()
        self.content_layout.addWidget(self.all_apps_list)

    def populate_apps_list(self):
        """Populate the apps list with real or demo data"""
        self.all_apps_list.clear()

        if hasattr(self, "current_apps") and self.current_apps:
            # Use real app data
            for app in self.current_apps[:10]:  # Show top 10
                name = app.get("name", "Unknown").replace(".exe", "")
                memory = app.get("memory_mb", 0)
                cpu = app.get("cpu_percent", 0)
                status = "🟢 ACTIVE" if app.get("is_active", False) else "⚫ Running"

                item_text = f"{status}  {name} - CPU: {cpu:.1f}% | RAM: {memory:.0f}MB"
                self.all_apps_list.addItem(item_text)
        else:
            # Show "None" message like in your screenshot
            none_item = QListWidgetItem("None")
            none_item.setTextAlignment(Qt.AlignCenter)
            none_item.setFlags(Qt.NoItemFlags)  # Make it non-selectable
            self.all_apps_list.addItem(none_item)

    def create_stat_card(self, title, value, color):
        # creates those little stat cards for the dashboard
        card = QFrame()
        card.setStyleSheet(
            f"""
            QFrame {{
                background-color: #3a3d44;
                border: 1px solid #4a5568;
                border-radius: 8px;
                padding: 20px;
                min-height: 120px;
            }}
            QFrame:hover {{
                border-color: {color};
                background-color: #4a5568;
            }}
        """
        )

        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignCenter)

        # the main value/number
        value_label = QLabel(value)
        value_label.setStyleSheet(
            f"""
            color: {color};
            font-size: {self.base_font_size + 16}px;
            font-weight: bold;
            margin-bottom: 5px;
        """
        )
        value_label.setAlignment(Qt.AlignCenter)

        # card title
        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"""
            color: #b0b3b8;
            font-size: {self.base_font_size + 2}px;
            font-weight: 500;
        """
        )
        title_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(value_label)
        layout.addWidget(title_label)

        return card

    def populate_activity_list(self):
        """Populate recent activity"""
        self.activity_list.clear()

        # Add some sample activities (you can replace with real data)
        activities = [
            "🎯 Task 'Complete project proposal' added",
            "💻 Switched to Visual Studio Code",
            "⏰ Break reminder triggered",
            "✅ Task 'Review documentation' completed",
            "🔴 High distraction risk detected",
        ]

        for activity in activities:
            self.activity_list.addItem(activity)

    def show_tasks(self):
        """Show the tasks page"""
        # Clear any dashboard references
        if hasattr(self, "dashboard_stat_cards"):
            self.dashboard_stat_cards = []

        # Page title
        title = QLabel("📋 Tasks & Goals")
        title.setObjectName("page_title")
        self.content_layout.addWidget(title)

        subtitle = QLabel("Manage your tasks and productivity goals")
        subtitle.setObjectName("subtitle")
        self.content_layout.addWidget(subtitle)

        # Add task button
        add_btn = QPushButton("+ Add New Task")
        add_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #38a169;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: {self.base_font_size + 2}px;
            }}
            QPushButton:hover {{
                background-color: #2f855a;
            }}
        """
        )
        add_btn.clicked.connect(self.add_task)
        self.content_layout.addWidget(add_btn)

        # Tasks list
        self.tasks_list = QListWidget()
        self.populate_tasks_list()
        self.content_layout.addWidget(self.tasks_list)

        self.content_layout.addStretch()

    def show_analytics(self):
        """Show the analytics page"""
        # Clear any dashboard references
        if hasattr(self, "dashboard_stat_cards"):
            self.dashboard_stat_cards = []

        title = QLabel("📈 Analytics")
        title.setObjectName("page_title")
        self.content_layout.addWidget(title)

        subtitle = QLabel("Detailed insights into your productivity patterns")
        subtitle.setObjectName("subtitle")
        self.content_layout.addWidget(subtitle)

        # Analytics content
        analytics_label = QLabel("Analytics dashboard coming soon...")
        analytics_label.setStyleSheet(
            f"color: #b0b3b8; font-size: {self.base_font_size + 4}px; text-align: center; margin: 50px;"
        )
        analytics_label.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(analytics_label)

        self.content_layout.addStretch()

    def show_settings(self):
        """Show the settings page"""
        # Clear any dashboard references
        if hasattr(self, "dashboard_stat_cards"):
            self.dashboard_stat_cards = []

        title = QLabel("⚙️ Settings")
        title.setObjectName("page_title")
        self.content_layout.addWidget(title)

        subtitle = QLabel("Configure your ProcastiNO preferences")
        subtitle.setObjectName("subtitle")
        self.content_layout.addWidget(subtitle)

        # Settings content
        settings_label = QLabel("Settings panel coming soon...")
        settings_label.setStyleSheet(
            f"color: #b0b3b8; font-size: {self.base_font_size + 4}px; text-align: center; margin: 50px;"
        )
        settings_label.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(settings_label)

        self.content_layout.addStretch()

    def populate_tasks_list(self):
        """Populate the tasks list with complete buttons"""
        self.tasks_list.clear()

        active_tasks = self.task_manager.get_active_tasks()

        if active_tasks:
            for task in active_tasks:
                # Create a custom widget for each task
                task_widget = QWidget()
                task_widget.setStyleSheet(
                    """
                    QWidget {
                        background: rgba(74, 85, 104, 0.5);
                        border: 1px solid #4a5568;
                        border-radius: 8px;
                        color: #ffffff;
                        min-height: 50px;
                    }
                    QWidget:hover {
                        background: rgba(124, 77, 255, 0.3);
                        border-color: #7C4DFF;
                    }
                """
                )
                task_layout = QHBoxLayout(task_widget)
                task_layout.setContentsMargins(20, 12, 20, 12)
                task_layout.setSpacing(20)

                # Priority icon
                priority_icon = (
                    "🔥"
                    if task.priority == "High"
                    else "⚡" if task.priority == "Medium" else "📝"
                )

                # Task text
                item_text = f"{priority_icon} {task.title}"
                if getattr(task, "assigned_app", None):
                    item_text += f" → {task.assigned_app}"

                task_text = QLabel(item_text)
                task_text.setStyleSheet(
                    f"""
                    QLabel {{
                        color: #ffffff !important;
                        font-size: {self.base_font_size + 2}px;
                        background: transparent;
                        border: none;
                    }}
                """
                )
                task_layout.addWidget(task_text)
                task_layout.addStretch()  # Push complete button to the right

                # Complete button
                complete_btn = QPushButton("✓ Complete")
                complete_btn.setStyleSheet(
                    f"""
                    QPushButton {{
                        background-color: #38a169;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-weight: bold;
                        font-size: {self.base_font_size}px;
                        padding: 8px 16px;
                        min-width: 80px;
                    }}
                    QPushButton:hover {{
                        background-color: #2f855a;
                    }}
                    QPushButton:pressed {{
                        background-color: #276749;
                    }}
                """
                )
                complete_btn.clicked.connect(
                    lambda checked, task_id=task.id: self.complete_task_from_tasks_page(
                        task_id
                    )
                )
                task_layout.addWidget(complete_btn)

                # Add to list
                item = QListWidgetItem()
                item.setSizeHint(task_widget.sizeHint())
                self.tasks_list.addItem(item)
                self.tasks_list.setItemWidget(item, task_widget)
        else:
            none_item = QListWidgetItem("No active tasks")
            none_item.setTextAlignment(Qt.AlignCenter)
            none_item.setFlags(Qt.NoItemFlags)
            self.tasks_list.addItem(none_item)

    def add_task(self):
        """Show add task dialog"""
        dialog = AddTaskDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            task_data = dialog.get_task_data()
            if task_data["title"]:
                self.task_manager.create_task(
                    task_data["title"],
                    task_data["description"],
                    task_data.get("app_assigned", ""),
                    task_data.get("priority", "Medium"),
                )
                # Refresh the appropriate list based on current page
                if hasattr(self, "populate_tasks_list"):
                    self.populate_tasks_list()  # Refresh tasks page list
                if (
                    hasattr(self, "dashboard_tasks_list")
                    and self.dashboard_tasks_list is not None
                ):
                    try:
                        self.populate_dashboard_tasks()  # Refresh dashboard list
                    except RuntimeError:
                        # Dashboard widget doesn't exist, skip refresh
                        pass

    def create_dashboard_tab(self):
        """Main dashboard with overview"""
        dashboard = QWidget()
        layout = QVBoxLayout(dashboard)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        self.setup_header(layout)

        # Stats cards
        self.setup_stats_cards(layout)

        # Main content area
        main_content = QHBoxLayout()

        # Left column
        left_layout = QVBoxLayout()

        # AI Distraction Prediction
        self.distraction_widget = DistractionWidget()
        left_layout.addWidget(self.distraction_widget)

        # Recent activity
        self.activity_widget = ActivityFeedWidget()
        left_layout.addWidget(self.activity_widget)

        main_content.addLayout(left_layout, 1)

        # Right column - Quick task view
        self.task_widget = TaskListWidget(self.task_manager)
        main_content.addWidget(self.task_widget, 1)

        layout.addLayout(main_content)

        self.tab_widget.addTab(dashboard, "📊 Dashboard")

    def create_apps_tab(self):
        """Tab showing ALL running applications"""
        apps_tab = QWidget()
        layout = QVBoxLayout(apps_tab)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("🖥️ Complete Application Monitor")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 15px;")
        layout.addWidget(title)

        # All apps widget
        self.all_apps_widget = AllAppsWidget()
        layout.addWidget(self.all_apps_widget)

        self.tab_widget.addTab(apps_tab, "🖥️ All Apps")

    def create_tasks_tab(self):
        """Tab for detailed task management"""
        tasks_tab = QWidget()
        layout = QVBoxLayout(tasks_tab)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("📋 Task & Goal Management")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 15px;")
        layout.addWidget(title)

        # Task management area
        task_layout = QHBoxLayout()

        # Active tasks
        active_frame = QFrame()
        active_frame.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 2px solid #27ae60;
                border-radius: 10px;
                padding: 15px;
            }
        """
        )
        active_layout = QVBoxLayout(active_frame)

        active_title = QLabel("🎯 Active Tasks")
        active_title.setFont(QFont("Arial", 14, QFont.Bold))
        active_title.setStyleSheet("color: #27ae60;")
        active_layout.addWidget(active_title)

        self.active_tasks_list = QListWidget()
        active_layout.addWidget(self.active_tasks_list)

        task_layout.addWidget(active_frame)

        # Completed tasks
        completed_frame = QFrame()
        completed_frame.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 2px solid #95a5a6;
                border-radius: 10px;
                padding: 15px;
            }
        """
        )
        completed_layout = QVBoxLayout(completed_frame)

        completed_title = QLabel("✅ Completed Tasks")
        completed_title.setFont(QFont("Arial", 14, QFont.Bold))
        completed_title.setStyleSheet("color: #95a5a6;")
        completed_layout.addWidget(completed_title)

        self.completed_tasks_list = QListWidget()
        completed_layout.addWidget(self.completed_tasks_list)

        task_layout.addWidget(completed_frame)

        layout.addLayout(task_layout)

        # Task controls
        controls_layout = QHBoxLayout()

        add_task_btn = QPushButton("➕ Add New Task")
        add_task_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """
        )
        add_task_btn.clicked.connect(self.add_new_task)

        refresh_btn = QPushButton("🔄 Refresh Tasks")
        refresh_btn.clicked.connect(self.refresh_all_tasks)

        controls_layout.addWidget(add_task_btn)
        controls_layout.addWidget(refresh_btn)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        self.tab_widget.addTab(tasks_tab, "📋 Tasks")

    def create_analytics_tab(self):
        """Tab for analytics and insights"""
        analytics_tab = QWidget()
        layout = QVBoxLayout(analytics_tab)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("📈 Analytics & AI Insights")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 15px;")
        layout.addWidget(title)

        # Analytics content
        analytics_content = QHBoxLayout()

        # App usage breakdown
        usage_frame = QFrame()
        usage_frame.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 10px;
                padding: 15px;
            }
        """
        )
        usage_layout = QVBoxLayout(usage_frame)

        usage_title = QLabel("📊 App Usage Today")
        usage_title.setFont(QFont("Arial", 14, QFont.Bold))
        usage_title.setStyleSheet("color: #3498db;")
        usage_layout.addWidget(usage_title)

        self.usage_display = QTextEdit()
        self.usage_display.setReadOnly(True)
        self.usage_display.setMaximumHeight(200)
        usage_layout.addWidget(self.usage_display)

        analytics_content.addWidget(usage_frame)

        # AI training data
        ai_frame = QFrame()
        ai_frame.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 2px solid #9b59b6;
                border-radius: 10px;
                padding: 15px;
            }
        """
        )
        ai_layout = QVBoxLayout(ai_frame)

        ai_title = QLabel("🤖 AI Training & Predictions")
        ai_title.setFont(QFont("Arial", 14, QFont.Bold))
        ai_title.setStyleSheet("color: #9b59b6;")
        ai_layout.addWidget(ai_title)

        # Training feedback buttons
        feedback_layout = QHBoxLayout()

        self.distracted_btn = QPushButton("😵 I'm Distracted")
        self.distracted_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
        """
        )
        self.distracted_btn.clicked.connect(lambda: self.add_training_data(True))

        self.focused_btn = QPushButton("🎯 I'm Focused")
        self.focused_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
        """
        )
        self.focused_btn.clicked.connect(lambda: self.add_training_data(False))

        feedback_layout.addWidget(QLabel("Train the AI:"))
        feedback_layout.addWidget(self.distracted_btn)
        feedback_layout.addWidget(self.focused_btn)
        feedback_layout.addStretch()

        ai_layout.addLayout(feedback_layout)

        self.ai_info_display = QTextEdit()
        self.ai_info_display.setReadOnly(True)
        self.ai_info_display.setMaximumHeight(150)
        ai_layout.addWidget(self.ai_info_display)

        analytics_content.addWidget(ai_frame)

        layout.addLayout(analytics_content)

        self.tab_widget.addTab(analytics_tab, "📈 Analytics")

    def setup_header(self, layout):
        """Setup header with title and session info"""
        header_layout = QHBoxLayout()

        # Title
        title = QLabel("🚫 ProcastiNO - Complete Productivity Suite")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #e74c3c; margin-bottom: 10px;")

        # Session info layout
        session_layout = QVBoxLayout()
        session_layout.setAlignment(Qt.AlignRight)

        # Current session
        current_session = QLabel(f"Current: {self.session_start.strftime('%H:%M:%S')}")
        current_session.setFont(QFont("Arial", 12))
        current_session.setStyleSheet("color: #27ae60; font-weight: bold;")

        # Last session
        if hasattr(self, "last_session_time") and self.last_session_time != "Never":
            last_session = QLabel(f"Last: {self.last_session_time}")
            last_session.setFont(QFont("Arial", 10))
            last_session.setStyleSheet("color: #7f8c8d;")
        else:
            last_session = QLabel("Last: First time running!")
            last_session.setFont(QFont("Arial", 10))
            last_session.setStyleSheet("color: #f39c12;")

        # Total sessions
        if hasattr(self, "total_sessions"):
            total_sessions = QLabel(f"Sessions: {self.total_sessions + 1}")
            total_sessions.setFont(QFont("Arial", 9))
            total_sessions.setStyleSheet("color: #95a5a6;")
            session_layout.addWidget(total_sessions)

        session_layout.addWidget(current_session)
        session_layout.addWidget(last_session)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addLayout(session_layout)

        layout.addLayout(header_layout)

    def setup_stats_cards(self, layout):
        """Setup enhanced statistics cards"""
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)

        # Create enhanced stats cards
        self.total_apps_card = StatsCard(
            "Apps Running", "0", "Currently active", "#3498db"
        )
        self.active_app_card = StatsCard(
            "Current Focus", "None", "Active application", "#27ae60"
        )
        self.session_time_card = StatsCard(
            "Session Time", "0m", "Time tracking", "#9b59b6"
        )
        self.distraction_card = StatsCard(
            "Focus Level", "Loading...", "AI prediction", "#e74c3c"
        )
        self.tasks_card = StatsCard("Tasks Today", "0/0", "Completed/Total", "#f39c12")

        stats_layout.addWidget(self.total_apps_card)
        stats_layout.addWidget(self.active_app_card)
        stats_layout.addWidget(self.session_time_card)
        stats_layout.addWidget(self.distraction_card)
        stats_layout.addWidget(self.tasks_card)

        layout.addLayout(stats_layout)

    def start_tracking(self):
        """Start all tracking systems"""
        # Initialize enhanced tracker thread
        self.app_tracker = RealTimeAppTracker()
        self.app_tracker.apps_updated.connect(self.update_apps_display)
        self.app_tracker.distraction_predicted.connect(
            self.update_distraction_prediction
        )
        self.app_tracker.start()

        # Timer for UI updates
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui_stats)
        self.ui_timer.start(1000)  # Update every second

        # Timer for analytics updates
        self.analytics_timer = QTimer()
        self.analytics_timer.timeout.connect(self.update_analytics)
        self.analytics_timer.start(10000)  # Update every 10 seconds

        # Activity log (new UI doesn't have persistent activity widget)
        print("[ProcastiNO] 🚀 Complete Suite started")
        print("[ProcastiNO] 🤖 AI distraction prediction active")
        print("[ProcastiNO] 📋 Task management ready")

        # Initial task refresh
        self.refresh_all_tasks()

    def update_apps_display(self, apps):
        """Update all app displays with real-time data"""
        # Store current apps for use in UI
        self.current_apps = apps

        # Update ALL APPS page if it's currently visible
        if self.current_page == "ALL APPS" and hasattr(self, "all_apps_list"):
            self.populate_apps_list()

        # Update dashboard if on dashboard
        if (
            self.current_page == "Dashboard"
            and hasattr(self, "dashboard_apps_list")
            and self.dashboard_apps_list is not None
        ):
            # Update the dashboard apps column
            self.update_dashboard_apps_display()

        # Update all apps widget (legacy)
        if hasattr(self, "all_apps_widget"):
            self.all_apps_widget.update_apps(apps)

        # Update dashboard stats (legacy)
        if apps:
            # Find active app
            active_apps = [app for app in apps if app["is_active"]]
            if active_apps:
                active_app = active_apps[0]["name"].replace(".exe", "")

                # Log app switches
                if active_app != self.last_active_app:
                    if self.last_active_app:
                        print(f"[ProcastiNO] 🔄 {self.last_active_app} → {active_app}")
                    else:
                        print(f"[ProcastiNO] 🎯 Focused on: {active_app}")
                    self.last_active_app = active_app

                if hasattr(self, "active_app_card"):
                    self.active_app_card.update_value(active_app)
            else:
                if hasattr(self, "active_app_card"):
                    self.active_app_card.update_value("None")

            # Update total apps
            if hasattr(self, "total_apps_card"):
                self.total_apps_card.update_value(len(apps))

    def update_distraction_prediction(self, probability, reason):
        """Update distraction prediction display"""
        # Update dashboard card if it exists (legacy support)
        if hasattr(self, "distraction_card"):
            percentage = int(probability * 100)
            if percentage < 30:
                status = "Focused"
            elif percentage < 60:
                status = "Moderate"
            else:
                status = "Distracted"

            self.distraction_card.update_value(f"{percentage}%", status)

    def update_ui_stats(self):
        """Update UI statistics every second"""
        # Update session time
        elapsed = datetime.now() - self.session_start
        total_seconds = int(elapsed.total_seconds())

        if total_seconds < 60:
            time_str = f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            time_str = f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            time_str = f"{hours}h {minutes}m"

        if hasattr(self, "session_time_card"):
            self.session_time_card.update_value(time_str)

        # Update task stats
        task_stats = (
            self.task_manager.get_task_statistics()
            if self.task_manager
            else {
                "active_tasks": 0,
                "completed_tasks": 0,
                "total_tasks": 0,
                "today_completed": 0,
                "completion_rate": 0,
            }
        )
        if hasattr(self, "tasks_card"):
            self.tasks_card.update_value(
                f"{task_stats.get('active_tasks', 0)} Active • {task_stats.get('completed_tasks', 0)} Done"
            )

    def update_analytics(self):
        """Update analytics displays"""
        if hasattr(self, "usage_display"):
            # App usage analytics
            usage_text = "📊 App Usage Analytics:\n\n"
            if self.app_tracker:
                switches = self.app_tracker.get_app_switches_today()
                usage_text += f"App switches today: {len(switches)}\n"

                if switches:
                    recent_switches = switches[-10:]  # Last 10
                    usage_text += "\nRecent app switches:\n"
                    for switch in recent_switches:
                        time_str = datetime.fromisoformat(switch["time"]).strftime(
                            "%H:%M"
                        )
                        usage_text += f"{time_str}: {switch['from']} → {switch['to']}\n"

            self.usage_display.setPlainText(usage_text)

        if hasattr(self, "ai_info_display"):
            # AI training info
            if self.app_tracker and hasattr(self.app_tracker, "predictor"):
                predictor = self.app_tracker.predictor
                ai_text = f"🤖 AI Model Status:\n\n"
                ai_text += f"Training samples: {len(predictor.training_data)}\n"
                ai_text += f"Model trained: {'Yes' if predictor.model else 'No'}\n"
                ai_text += f"ML Available: {'Yes' if ML_AVAILABLE else 'No'}\n"

                if predictor.last_prediction:
                    ai_text += f"\nLast prediction: {predictor.last_prediction:.2f}\n"
                    ai_text += f"Confidence: {predictor.prediction_confidence:.2f}\n"

                ai_text += "\n💡 Tip: Use the feedback buttons to train the AI!"

                self.ai_info_display.setPlainText(ai_text)

    def add_training_data(self, was_distracted):
        """Add training data for AI"""
        if self.app_tracker and hasattr(self.app_tracker, "predictor"):
            # Get current app info for training
            if (
                hasattr(self.app_tracker, "current_app_name")
                and self.app_tracker.current_app_name
            ):
                features = self.app_tracker.predictor.extract_features(
                    self.app_tracker.current_app_name,
                    (
                        datetime.now() - self.app_tracker.current_app_start
                    ).total_seconds()
                    / 60,
                    50,  # Dummy values for now
                    100,
                    datetime.now().hour,
                    len(self.app_tracker.app_switches),
                )

                self.app_tracker.predictor.add_training_sample(features, was_distracted)

                status = "distracted" if was_distracted else "focused"
                print(f"[ProcastiNO] 🤖 Trained AI: marked as {status}")

    def add_new_task(self):
        """Add a new task"""
        dialog = AddTaskDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            task_data = dialog.get_task_data()
            if task_data["title"]:
                task = self.task_manager.create_task(
                    task_data["title"],
                    task_data["description"],
                    task_data.get("app_assigned", ""),
                    task_data.get("priority", "Medium"),
                )
                print(f"[ProcastiNO] 📋 Added task: {task['title']}")
                self.refresh_all_tasks()

    def refresh_all_tasks(self):
        """Refresh all task displays"""
        if hasattr(self, "task_widget"):
            self.task_widget.refresh_tasks()

        if hasattr(self, "active_tasks_list"):
            self.active_tasks_list.clear()
            for task in self.task_manager.get_active_tasks():
                priority_icon = (
                    "🔥"
                    if task["priority"] == "High"
                    else "⚡" if task["priority"] == "Medium" else "📝"
                )
                item_text = f"{priority_icon} {task['title']}"
                if task["app_assigned"]:
                    item_text += f" → {task['app_assigned']}"
                self.active_tasks_list.addItem(item_text)

        if hasattr(self, "completed_tasks_list"):
            self.completed_tasks_list.clear()
            for task in self.task_manager.get_completed_tasks():
                completed_time = ""
                if task["completed_at"]:
                    completed_time = datetime.fromisoformat(
                        task["completed_at"]
                    ).strftime("%H:%M")
                item_text = f"✅ {task['title']} ({completed_time})"
                self.completed_tasks_list.addItem(item_text)

    def closeEvent(self, event):
        """Clean shutdown"""
        if self.app_tracker:
            self.app_tracker.stop()
            self.app_tracker.wait()

        print("[ProcastiNO] Shutting down...")
        event.accept()


def main():
    # main function to start the app
    app = QApplication(sys.argv)

    # basic app info
    app.setApplicationName("ProcastiNO")
    app.setApplicationVersion("2.0")

    # create main window and show it
    window = ProcastiNOMainWindow()
    window.show()

    # startup messages
    print("🚀 ProcastiNO started!")
    print("📱 Monitoring your apps")
    print("🔄 Switch between apps to see live updates")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
