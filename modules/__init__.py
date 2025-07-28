"""
ProcastiNo Modules Package

This package contains all the core tracking and analysis modules for the ProcastiNo application.

Modules:
- app_tracker: Real-time application usage tracking
- distraction_predictor: Machine learning based distraction prediction
- task_manager: Task management with inactivity detection
- break_reminder: Break monitoring and wellness reminders
- analytics: Data visualization and analytics engine
"""

__version__ = "1.0.0"
__author__ = "ProcastiNo Team"


# Module availability check
def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import psutil
        import pygetwindow
        import sklearn
        import matplotlib
        import pandas
        import numpy
        import plyer

        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False


# Export main classes for easy importing
try:
    from .app_tracker import AppUsageTracker
    from .distraction_predictor import DistractionPredictor
    from .task_manager import TaskManager
    from .break_reminder import BreakReminder
    from .analytics import ScreenTimeAnalytics, AnalyticsWidget

    __all__ = [
        "AppUsageTracker",
        "DistractionPredictor",
        "TaskManager",
        "BreakReminder",
        "ScreenTimeAnalytics",
        "AnalyticsWidget",
        "check_dependencies",
    ]

except ImportError:
    # If modules can't be imported, at least export the checker
    __all__ = ["check_dependencies"]
