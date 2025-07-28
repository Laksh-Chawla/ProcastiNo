# Analytics and visualization for screen time data
# Uses matplotlib to create charts and graphs

import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from PyQt5.QtWidgets import QWidget, QVBoxLayout


class ScreenTimeAnalytics:
    # analyzes app usage and creates nice charts

    def __init__(
        self,
        app_data_file: str = "app_usage_data.json",
        break_data_file: str = "break_data.json",
    ):
        self.app_data_file = app_data_file
        self.break_data_file = break_data_file

        # load data from files
        self.app_usage_data = self.load_app_data()
        self.break_data = self.load_break_data()

        # set up matplotlib to look nice
        plt.style.use("dark_background")
        self.setup_plot_style()

        print("[ScreenTimeAnalytics] Ready!")

    def setup_plot_style(self):
        # configure how the charts look
        plt.rcParams.update(
            {
                "figure.facecolor": "#ffffff",
                "axes.facecolor": "#f8f8f8",
                "axes.edgecolor": "#e0e0e0",
                "axes.labelcolor": "#333333",
                "text.color": "#333333",
                "xtick.color": "#666666",
                "ytick.color": "#666666",
                "grid.color": "#e0e0e0",
                "grid.alpha": 0.7,
                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "figure.titlesize": 14,
            }
        )

    def load_app_data(self) -> List[Dict]:
        """
        Load app usage data from JSON file
        """
        if not os.path.exists(self.app_data_file):
            print(f"[ScreenTimeAnalytics] No app usage data found")
            return []

        try:
            with open(self.app_data_file, "r") as f:
                data = json.load(f)
            print(f"[ScreenTimeAnalytics] Loaded {len(data)} app usage records")
            return data
        except Exception as e:
            print(f"[ScreenTimeAnalytics] Error loading app data: {e}")
            return []

    def load_break_data(self) -> List[Dict]:
        """
        Load break data from JSON file
        """
        if not os.path.exists(self.break_data_file):
            print(f"[ScreenTimeAnalytics] No break data found")
            return []

        try:
            with open(self.break_data_file, "r") as f:
                data = json.load(f)
            print(f"[ScreenTimeAnalytics] Loaded {len(data)} break records")
            return data
        except Exception as e:
            print(f"[ScreenTimeAnalytics] Error loading break data: {e}")
            return []

    def get_daily_screen_time(self, days: int = 7) -> Dict:
        """
        Get daily screen time for the last N days

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with dates as keys and total minutes as values
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days - 1)

        daily_totals = {}

        # Initialize all dates with 0
        current_date = start_date
        while current_date <= end_date:
            daily_totals[current_date.strftime("%Y-%m-%d")] = 0
            current_date += timedelta(days=1)

        # Sum up usage for each day
        for session in self.app_usage_data:
            try:
                session_date = datetime.fromisoformat(session["start_time"]).date()
                date_str = session_date.strftime("%Y-%m-%d")

                if date_str in daily_totals:
                    daily_totals[date_str] += session["duration_seconds"] / 60
            except (ValueError, KeyError):
                continue

        return daily_totals

    def get_app_usage_breakdown(self, days: int = 7) -> Dict:
        """
        Get app usage breakdown for the last N days

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with app names as keys and total minutes as values
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        app_totals = {}

        for session in self.app_usage_data:
            try:
                session_time = datetime.fromisoformat(session["start_time"])
                if session_time >= cutoff_date:
                    app_name = session["app_name"]
                    duration_minutes = session["duration_seconds"] / 60
                    app_totals[app_name] = (
                        app_totals.get(app_name, 0) + duration_minutes
                    )
            except (ValueError, KeyError):
                continue

        # Sort by usage time
        return dict(sorted(app_totals.items(), key=lambda x: x[1], reverse=True))

    def get_category_breakdown(self, days: int = 7) -> Dict:
        """
        Get category usage breakdown for the last N days
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        category_totals = {}

        for session in self.app_usage_data:
            try:
                session_time = datetime.fromisoformat(session["start_time"])
                if session_time >= cutoff_date:
                    category = session.get("category", "Other")
                    duration_minutes = session["duration_seconds"] / 60
                    category_totals[category] = (
                        category_totals.get(category, 0) + duration_minutes
                    )
            except (ValueError, KeyError):
                continue

        return category_totals

    def get_hourly_usage_pattern(self, days: int = 7) -> Dict:
        """
        Get hourly usage patterns for the last N days
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        hourly_totals = {hour: 0 for hour in range(24)}

        for session in self.app_usage_data:
            try:
                session_time = datetime.fromisoformat(session["start_time"])
                if session_time >= cutoff_date:
                    hour = session_time.hour
                    duration_minutes = session["duration_seconds"] / 60
                    hourly_totals[hour] += duration_minutes
            except (ValueError, KeyError):
                continue

        return hourly_totals

    def create_daily_screen_time_chart(self, days: int = 7) -> Figure:
        """
        Create a bar chart showing daily screen time

        Args:
            days: Number of days to show

        Returns:
            matplotlib Figure object
        """
        daily_data = self.get_daily_screen_time(days)

        fig, ax = plt.subplots(figsize=(12, 6))

        dates = list(daily_data.keys())
        times = [daily_data[date] / 60 for date in dates]  # Convert to hours

        # Create bar chart
        bars = ax.bar(
            dates, times, color="#4a9eff", alpha=0.8, edgecolor="#357abd", linewidth=1
        )

        # Customize chart
        ax.set_title("Daily Screen Time", fontweight="bold", pad=20)
        ax.set_ylabel("Hours")
        ax.set_xlabel("Date")

        # Format x-axis
        ax.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{time:.1f}h",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Adjust layout
        plt.tight_layout()

        return fig

    def create_app_usage_pie_chart(self, days: int = 7, top_n: int = 8) -> Figure:
        """
        Create a pie chart showing app usage breakdown

        Args:
            days: Number of days to analyze
            top_n: Number of top apps to show individually

        Returns:
            matplotlib Figure object
        """
        app_data = self.get_app_usage_breakdown(days)

        if not app_data:
            # Create empty chart
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(
                0.5,
                0.5,
                "No usage data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title("App Usage Breakdown", fontweight="bold")
            return fig

        # Get top N apps and group the rest
        apps = list(app_data.keys())[:top_n]
        values = [app_data[app] / 60 for app in apps]  # Convert to hours

        if len(app_data) > top_n:
            other_total = (
                sum(app_data[app] for app in list(app_data.keys())[top_n:]) / 60
            )
            apps.append("Others")
            values.append(other_total)

        # Color palette
        colors = [
            "#4a9eff",
            "#2ed573",
            "#ff4757",
            "#ffa502",
            "#9c88ff",
            "#3742fa",
            "#2f3542",
            "#ff6348",
            "#7bed9f",
            "#70a1ff",
        ]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=apps,
            autopct="%1.1f%%",
            colors=colors[: len(apps)],
            startangle=90,
            textprops={"fontsize": 10},
        )

        # Enhance appearance
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        ax.set_title("App Usage Breakdown (Last 7 Days)", fontweight="bold", pad=20)

        return fig

    def create_category_breakdown_chart(self, days: int = 7) -> Figure:
        """
        Create a horizontal bar chart for category breakdown
        """
        category_data = self.get_category_breakdown(days)

        if not category_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No category data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title("Category Breakdown", fontweight="bold")
            return fig

        categories = list(category_data.keys())
        times = [category_data[cat] / 60 for cat in categories]  # Convert to hours

        # Color mapping for categories
        category_colors = {
            "Work": "#4a9eff",
            "Productivity": "#2ed573",
            "Social": "#ff4757",
            "Entertainment": "#ffa502",
            "Gaming": "#9c88ff",
            "Other": "#7f8c8d",
        }

        colors = [category_colors.get(cat, "#7f8c8d") for cat in categories]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create horizontal bar chart
        bars = ax.barh(
            categories, times, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5
        )

        # Customize chart
        ax.set_title("Usage by Category", fontweight="bold", pad=20)
        ax.set_xlabel("Hours")

        # Add value labels
        for bar, time in zip(bars, times):
            width = bar.get_width()
            ax.text(
                width + 0.1,
                bar.get_y() + bar.get_height() / 2.0,
                f"{time:.1f}h",
                ha="left",
                va="center",
                fontsize=9,
            )

        # Add grid
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_axisbelow(True)

        plt.tight_layout()

        return fig

    def create_hourly_pattern_chart(self, days: int = 7) -> Figure:
        """
        Create a line chart showing hourly usage patterns
        """
        hourly_data = self.get_hourly_usage_pattern(days)

        hours = list(range(24))
        usage = [hourly_data[hour] / 60 for hour in hours]  # Convert to hours

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create line chart
        ax.plot(
            hours,
            usage,
            color="#4a9eff",
            linewidth=3,
            marker="o",
            markersize=6,
            markerfacecolor="#357abd",
            markeredgecolor="white",
            markeredgewidth=1,
        )

        # Fill area under the curve
        ax.fill_between(hours, usage, alpha=0.3, color="#4a9eff")

        # Customize chart
        ax.set_title("Hourly Usage Pattern", fontweight="bold", pad=20)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Hours per Day")

        # Set x-axis ticks
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Highlight peak hours
        if usage:
            peak_hour = hours[usage.index(max(usage))]
            ax.axvline(
                x=peak_hour, color="#ff4757", linestyle="--", alpha=0.7, linewidth=2
            )
            ax.text(
                peak_hour + 0.5,
                max(usage),
                f"Peak: {peak_hour:02d}:00",
                color="#ff4757",
                fontweight="bold",
            )

        plt.tight_layout()

        return fig

    def create_productivity_trends_chart(self, days: int = 30) -> Figure:
        """
        Create a chart showing productivity trends over time
        """
        # Calculate daily productivity scores
        daily_productivity = {}
        cutoff_date = datetime.now() - timedelta(days=days)

        for i in range(days):
            date = (cutoff_date + timedelta(days=i)).date()
            date_str = date.strftime("%Y-%m-%d")
            daily_productivity[date_str] = self.calculate_productivity_score(date)

        dates = list(daily_productivity.keys())
        scores = list(daily_productivity.values())

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create line chart
        ax.plot(
            dates,
            scores,
            color="#2ed573",
            linewidth=3,
            marker="o",
            markersize=4,
            markerfacecolor="#26d169",
            markeredgecolor="white",
        )

        # Add trend line
        if len(scores) > 1:
            z = np.polyfit(range(len(scores)), scores, 1)
            p = np.poly1d(z)
            ax.plot(
                dates,
                p(range(len(scores))),
                "--",
                color="#ff4757",
                alpha=0.8,
                linewidth=2,
            )

        # Customize chart
        ax.set_title("Productivity Trends", fontweight="bold", pad=20)
        ax.set_ylabel("Productivity Score (%)")
        ax.set_xlabel("Date")
        ax.set_ylim(0, 100)

        # Format x-axis
        ax.tick_params(axis="x", rotation=45)

        # Add horizontal reference lines
        ax.axhline(y=50, color="#ffa502", linestyle=":", alpha=0.7, label="Average")
        ax.axhline(y=75, color="#2ed573", linestyle=":", alpha=0.7, label="Good")

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()

        return fig

    def calculate_productivity_score(self, date: datetime.date) -> float:
        """
        Calculate productivity score for a given date

        Args:
            date: Date to calculate score for

        Returns:
            Productivity score between 0 and 100
        """
        date_str = date.strftime("%Y-%m-%d")

        # Get sessions for this date
        day_sessions = [s for s in self.app_usage_data if s.get("date") == date_str]

        if not day_sessions:
            return 0

        # Calculate productive vs non-productive time
        productive_categories = ["Work", "Productivity"]

        total_time = sum(s["duration_seconds"] for s in day_sessions)
        productive_time = sum(
            s["duration_seconds"]
            for s in day_sessions
            if s.get("category") in productive_categories
        )

        if total_time == 0:
            return 0

        # Base score from productive time ratio
        productivity_score = (productive_time / total_time) * 100

        # Adjust based on session patterns
        # Penalty for too many short sessions (context switching)
        short_sessions = sum(
            1 for s in day_sessions if s["duration_seconds"] < 300
        )  # < 5 minutes
        if len(day_sessions) > 0:
            short_session_ratio = short_sessions / len(day_sessions)
            productivity_score *= 1 - short_session_ratio * 0.3  # Up to 30% penalty

        return min(100, max(0, productivity_score))


class AnalyticsWidget(QWidget):
    """
    PyQt5 widget that embeds matplotlib charts
    """

    def __init__(self, analytics: ScreenTimeAnalytics):
        super().__init__()
        self.analytics = analytics
        self.setup_ui()

    def setup_ui(self):
        """
        Set up the widget UI
        """
        layout = QVBoxLayout(self)

        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8), facecolor="#1a1a1a")
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(self.canvas)

        # Initial chart
        self.update_chart("daily_screen_time")

    def update_chart(self, chart_type: str, **kwargs):
        """
        Update the displayed chart

        Args:
            chart_type: Type of chart to display
            **kwargs: Additional arguments for chart creation
        """
        self.figure.clear()

        try:
            if chart_type == "daily_screen_time":
                fig = self.analytics.create_daily_screen_time_chart(
                    kwargs.get("days", 7)
                )
            elif chart_type == "app_breakdown":
                fig = self.analytics.create_app_usage_pie_chart(kwargs.get("days", 7))
            elif chart_type == "category_breakdown":
                fig = self.analytics.create_category_breakdown_chart(
                    kwargs.get("days", 7)
                )
            elif chart_type == "hourly_pattern":
                fig = self.analytics.create_hourly_pattern_chart(kwargs.get("days", 7))
            elif chart_type == "productivity_trends":
                fig = self.analytics.create_productivity_trends_chart(
                    kwargs.get("days", 30)
                )
            else:
                return

            # Copy the figure to our canvas
            for ax in fig.get_axes():
                new_ax = self.figure.add_subplot(111)
                # Copy all the properties and data from the original axis
                new_ax.clear()

                # Simple approach: recreate the chart on our figure
                if chart_type == "daily_screen_time":
                    daily_data = self.analytics.get_daily_screen_time(
                        kwargs.get("days", 7)
                    )
                    dates = list(daily_data.keys())
                    times = [daily_data[date] / 60 for date in dates]

                    bars = new_ax.bar(dates, times, color="#4a9eff", alpha=0.8)
                    new_ax.set_title("Daily Screen Time", fontweight="bold", pad=20)
                    new_ax.set_ylabel("Hours")
                    new_ax.set_xlabel("Date")
                    new_ax.tick_params(axis="x", rotation=45)

                    for bar, time in zip(bars, times):
                        height = bar.get_height()
                        new_ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.1,
                            f"{time:.1f}h",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                        )

                    new_ax.grid(True, alpha=0.3)

            plt.close(fig)  # Close the temporary figure

        except Exception as e:
            print(f"[AnalyticsWidget] Error updating chart: {e}")

        self.canvas.draw()


# Example usage and testing
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QVBoxLayout,
        QWidget,
        QPushButton,
        QHBoxLayout,
    )

    # Create some sample data for testing
    def create_sample_data():
        """Create sample app usage data for testing"""
        sample_data = []

        # Generate data for the last 7 days
        for day_offset in range(7):
            date = datetime.now() - timedelta(days=day_offset)

            # Generate random sessions throughout the day
            for _ in range(np.random.randint(5, 15)):
                start_time = date.replace(
                    hour=np.random.randint(8, 22),
                    minute=np.random.randint(0, 60),
                    second=0,
                    microsecond=0,
                )

                duration = np.random.randint(300, 3600)  # 5 minutes to 1 hour

                apps = ["code", "chrome", "instagram", "youtube", "word", "figma"]
                categories = [
                    "Work",
                    "Productivity",
                    "Social",
                    "Entertainment",
                    "Work",
                    "Work",
                ]

                app_idx = np.random.randint(len(apps))

                session = {
                    "app_name": apps[app_idx],
                    "category": categories[app_idx],
                    "start_time": start_time.isoformat(),
                    "end_time": (start_time + timedelta(seconds=duration)).isoformat(),
                    "duration_seconds": duration,
                    "date": start_time.strftime("%Y-%m-%d"),
                }

                sample_data.append(session)

        # Save sample data
        with open("app_usage_data.json", "w") as f:
            json.dump(sample_data, f, indent=2)

        print(f"Created {len(sample_data)} sample app usage records")

    class AnalyticsTestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Screen Time Analytics Test")
            self.setGeometry(100, 100, 1200, 800)

            # Create analytics
            self.analytics = ScreenTimeAnalytics()

            # Set up UI
            central_widget = QWidget()
            self.setCentralWidget(central_widget)

            layout = QVBoxLayout(central_widget)

            # Buttons for different charts
            button_layout = QHBoxLayout()

            charts = [
                ("Daily Screen Time", "daily_screen_time"),
                ("App Breakdown", "app_breakdown"),
                ("Category Breakdown", "category_breakdown"),
                ("Hourly Pattern", "hourly_pattern"),
                ("Productivity Trends", "productivity_trends"),
            ]

            for label, chart_type in charts:
                btn = QPushButton(label)
                btn.clicked.connect(lambda checked, ct=chart_type: self.show_chart(ct))
                button_layout.addWidget(btn)

            layout.addLayout(button_layout)

            # Analytics widget
            self.analytics_widget = AnalyticsWidget(self.analytics)
            layout.addWidget(self.analytics_widget)

        def show_chart(self, chart_type):
            self.analytics_widget.update_chart(chart_type)

    # Create sample data if none exists
    if not os.path.exists("app_usage_data.json"):
        create_sample_data()

    # Run the test application
    app = QApplication(sys.argv)

    # Set dark theme
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.Window, plt.rcParams["figure.facecolor"])
    app.setPalette(palette)

    window = AnalyticsTestWindow()
    window.show()

    sys.exit(app.exec_())
