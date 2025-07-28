# Task management system for ProcastiNo
# Handles task creation, tracking, and notifications

import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from plyer import notification


@dataclass
class Task:
    # basic task structure - pretty straightforward
    id: int
    title: str
    description: str
    assigned_app: str  # which app this task relates to
    priority: str  # High, Medium, Low priorities
    created_at: str
    deadline: Optional[str] = None
    completed: bool = False
    completed_at: Optional[str] = None
    total_time_spent: float = 0.0  # in minutes
    last_activity_time: Optional[str] = None


class TaskManager:
    # handles all the task stuff - creating, updating, tracking

    def __init__(
        self, data_file: str = "tasks_data.json", inactivity_threshold_minutes: int = 5
    ):
        self.data_file = data_file
        self.inactivity_threshold_minutes = inactivity_threshold_minutes
        self.tasks = []  # list of all tasks
        self.active_task = None
        self.task_start_time = None
        self.monitoring_active = False
        self.monitoring_thread = None
        self.task_id_counter = 1

        # Callback functions for UI updates
        self.on_inactivity_detected: Optional[Callable] = None
        self.on_task_completed = None
        self.on_focus_reminder_sent = None

        # load any existing tasks from file
        self.load_tasks()

        print("[TaskManager] Ready to go!")

    def load_tasks(self):
        # try to load tasks from the json file
        if not os.path.exists(self.data_file):
            print("[TaskManager] No tasks file found, starting fresh")
            return

        try:
            with open(self.data_file, "r") as f:
                tasks_data = json.load(f)

            self.tasks = []
            for task_dict in tasks_data:
                task = Task(**task_dict)
                self.tasks.append(task)
                # make sure we don't reuse IDs
                if task.id >= self.task_id_counter:
                    self.task_id_counter = task.id + 1

            print(f"[TaskManager] Loaded {len(self.tasks)} existing tasks")

        except Exception as e:
            print(f"[TaskManager] Couldn't load tasks: {e}")
            self.tasks = []

    def save_tasks(self):
        # save all tasks to json file
        try:
            tasks_data = [asdict(task) for task in self.tasks]
            with open(self.data_file, "w") as f:
                json.dump(tasks_data, f, indent=2)
            print(f"[TaskManager] Saved {len(self.tasks)} tasks to file")
        except Exception as e:
            print(f"[TaskManager] Error saving tasks: {e}")

    def create_task(
        self,
        title: str,
        description: str,
        assigned_app: str,
        priority: str = "Medium",
        deadline: Optional[str] = None,
    ) -> Task:
        # creates a new task and adds it to our list
        task = Task(
            id=self.task_id_counter,
            title=title,
            description=description,
            assigned_app=assigned_app.lower(),
            priority=priority,
            created_at=datetime.now().isoformat(),
            deadline=deadline,
        )

        self.tasks.append(task)
        self.task_id_counter += 1
        self.save_tasks()

        print(f"[TaskManager] Created task: {title} -> {assigned_app}")
        return task

    def get_task_by_id(self, task_id: int):
        # find a task by its ID
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def start_task(self, task_id: int):
        # start working on a specific task
        task = self.get_task_by_id(task_id)
        if not task:
            print(f"[TaskManager] Task {task_id} not found")
            return False

        if task.completed:
            print(f"[TaskManager] Task {task_id} is already done")
            return False

        # stop previous task if we have one running
        if self.active_task:
            self.end_current_task()

        self.active_task = task
        self.task_start_time = datetime.now()
        task.last_activity_time = datetime.now().isoformat()

        print(f"[TaskManager] Started working on: {task.title}")

        # start watching for inactivity
        if not self.monitoring_active:
            self.start_inactivity_monitoring()

        return True

    def end_current_task(self):
        # stop the current task and save time spent
        if not self.active_task or not self.task_start_time:
            return

        # figure out how long we worked on it
        time_spent_minutes = (
            datetime.now() - self.task_start_time
        ).total_seconds() / 60
        self.active_task.total_time_spent += time_spent_minutes

        print(
            f"[TaskManager] Ended task: {self.active_task.title} - Time spent: {time_spent_minutes:.1f} minutes"
        )

        self.active_task = None
        self.task_start_time = None
        self.save_tasks()

    def complete_task(self, task_id: int) -> bool:
        """
        Mark a task as completed

        Args:
            task_id: ID of the task to complete

        Returns:
            True if task completed successfully
        """
        task = self.get_task_by_id(task_id)
        if not task:
            print(f"[TaskManager] Task {task_id} not found")
            return False

        if task.completed:
            print(f"[TaskManager] Task {task_id} is already completed")
            return False

        # End task if it's currently active
        if self.active_task and self.active_task.id == task_id:
            self.end_current_task()

        task.completed = True
        task.completed_at = datetime.now().isoformat()
        self.save_tasks()

        print(f"[TaskManager] Completed task: {task.title}")

        # Call callback if set
        if self.on_task_completed:
            self.on_task_completed(task)

        return True

    def delete_task(self, task_id: int) -> bool:
        """
        Delete a task

        Args:
            task_id: ID of the task to delete

        Returns:
            True if task deleted successfully
        """
        task = self.get_task_by_id(task_id)
        if not task:
            print(f"[TaskManager] Task {task_id} not found")
            return False

        # End task if it's currently active
        if self.active_task and self.active_task.id == task_id:
            self.end_current_task()

        self.tasks = [t for t in self.tasks if t.id != task_id]
        self.save_tasks()

        print(f"[TaskManager] Deleted task: {task.title}")
        return True

    def update_task_activity(self, current_app: str) -> None:
        """
        Update task activity based on current app usage

        Args:
            current_app: Name of the currently active app
        """
        if not self.active_task:
            return

        # Check if current app matches the task's assigned app
        if current_app.lower() == self.active_task.assigned_app.lower():
            self.active_task.last_activity_time = datetime.now().isoformat()
            print(f"[TaskManager] Updated activity for task: {self.active_task.title}")

    def check_inactivity(self) -> bool:
        """
        Check if the user has been inactive on the current task

        Returns:
            True if user is inactive (should be reminded)
        """
        if not self.active_task or not self.active_task.last_activity_time:
            return False

        last_activity = datetime.fromisoformat(self.active_task.last_activity_time)
        time_since_activity = (datetime.now() - last_activity).total_seconds() / 60

        return time_since_activity >= self.inactivity_threshold_minutes

    def send_focus_reminder(self) -> None:
        """
        Send a desktop notification to remind user to focus
        """
        if not self.active_task:
            return

        try:
            notification.notify(
                title="Focus Reminder",
                message=f"You've been away from '{self.active_task.title}' for {self.inactivity_threshold_minutes} minutes. Time to refocus!",
                app_name="ProcastiNo",
                timeout=10,
            )

            print(
                f"[TaskManager] Sent focus reminder for task: {self.active_task.title}"
            )

            # Call callback if set
            if self.on_focus_reminder_sent:
                self.on_focus_reminder_sent(self.active_task)

        except Exception as e:
            print(f"[TaskManager] Error sending notification: {e}")

    def start_inactivity_monitoring(self) -> None:
        """
        Start monitoring for task inactivity in a separate thread
        """
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        print("[TaskManager] Started inactivity monitoring")

    def stop_inactivity_monitoring(self) -> None:
        """
        Stop inactivity monitoring
        """
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)

        print("[TaskManager] Stopped inactivity monitoring")

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop - runs in separate thread
        """
        last_reminder_time = None

        while self.monitoring_active:
            try:
                if self.active_task and self.check_inactivity():
                    # Only send reminder if we haven't sent one recently
                    now = datetime.now()
                    if (
                        last_reminder_time is None
                        or (now - last_reminder_time).total_seconds() >= 300
                    ):  # 5 minutes between reminders

                        self.send_focus_reminder()
                        last_reminder_time = now

                        # Call callback if set
                        if self.on_inactivity_detected:
                            self.on_inactivity_detected(self.active_task)

                # Check every 30 seconds
                time.sleep(30)

            except Exception as e:
                print(f"[TaskManager] Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def get_active_tasks(self) -> List[Task]:
        """
        Get all active (non-completed) tasks
        """
        return [task for task in self.tasks if not task.completed]

    def get_completed_tasks(self) -> List[Task]:
        """
        Get all completed tasks
        """
        return [task for task in self.tasks if task.completed]

    def get_tasks_by_priority(self, priority: str) -> List[Task]:
        """
        Get tasks by priority level
        """
        return [
            task
            for task in self.tasks
            if task.priority == priority and not task.completed
        ]

    def get_overdue_tasks(self) -> List[Task]:
        """
        Get tasks that are past their deadline
        """
        today = datetime.now().date()
        overdue_tasks = []

        for task in self.tasks:
            if not task.completed and task.deadline:
                try:
                    deadline_date = datetime.fromisoformat(task.deadline).date()
                    if deadline_date < today:
                        overdue_tasks.append(task)
                except ValueError:
                    continue  # Skip tasks with invalid deadline format

        return overdue_tasks

    def get_task_statistics(self) -> Dict:
        """
        Get statistics about tasks
        """
        total_tasks = len(self.tasks)
        completed_tasks = len(self.get_completed_tasks())
        active_tasks = len(self.get_active_tasks())
        overdue_tasks = len(self.get_overdue_tasks())

        # Calculate total time spent
        total_time_spent = sum(task.total_time_spent for task in self.tasks)

        # Calculate completion rate
        completion_rate = (
            (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        )

        # Priority breakdown
        priority_breakdown = {}
        for priority in ["High", "Medium", "Low"]:
            priority_breakdown[priority] = len(self.get_tasks_by_priority(priority))

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "active_tasks": active_tasks,
            "overdue_tasks": overdue_tasks,
            "completion_rate": round(completion_rate, 1),
            "total_time_spent_hours": round(total_time_spent / 60, 1),
            "priority_breakdown": priority_breakdown,
            "active_task_id": self.active_task.id if self.active_task else None,
            "active_task_title": self.active_task.title if self.active_task else None,
        }

    def get_today_task_time(self) -> float:
        """
        Get total time spent on tasks today
        """
        today = datetime.now().date()
        total_time = 0.0

        for task in self.tasks:
            # For completed tasks, check completion date
            if task.completed and task.completed_at:
                try:
                    completed_date = datetime.fromisoformat(task.completed_at).date()
                    if completed_date == today:
                        total_time += task.total_time_spent
                except ValueError:
                    continue

            # For active task, add current session time
            elif (
                task == self.active_task
                and self.task_start_time
                and self.task_start_time.date() == today
            ):
                session_time = (
                    datetime.now() - self.task_start_time
                ).total_seconds() / 60
                total_time += session_time

        return total_time


# Example usage and testing
if __name__ == "__main__":
    import signal
    import sys

    def signal_handler(sig, frame):
        print("\n[TaskManager] Stopping...")
        task_manager.stop_inactivity_monitoring()
        sys.exit(0)

    # Create task manager
    task_manager = TaskManager()

    # Set up callbacks
    def on_inactivity(task):
        print(f"CALLBACK: Inactivity detected for task: {task.title}")

    def on_task_complete(task):
        print(f"CALLBACK: Task completed: {task.title}")

    def on_reminder_sent(task):
        print(f"CALLBACK: Focus reminder sent for: {task.title}")

    task_manager.on_inactivity_detected = on_inactivity
    task_manager.on_task_completed = on_task_complete
    task_manager.on_focus_reminder_sent = on_reminder_sent

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)

    # Create some example tasks
    if len(task_manager.get_active_tasks()) == 0:
        task1 = task_manager.create_task(
            "Complete Python Project",
            "Finish the procrastination tracker application",
            "code",
            "High",
            "2025-08-01",
        )

        task2 = task_manager.create_task(
            "Write Documentation",
            "Document the app usage tracking features",
            "word",
            "Medium",
        )

        task3 = task_manager.create_task(
            "Design UI Mockups",
            "Create wireframes for the analytics dashboard",
            "figma",
            "Low",
        )

    # Start a task for testing
    active_tasks = task_manager.get_active_tasks()
    if active_tasks:
        task_manager.start_task(active_tasks[0].id)
        print(f"Started task: {active_tasks[0].title}")

    # Show statistics
    stats = task_manager.get_task_statistics()
    print(f"\nTask Statistics:")
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"Active tasks: {stats['active_tasks']}")
    print(f"Completed tasks: {stats['completed_tasks']}")
    print(f"Completion rate: {stats['completion_rate']}%")
    print(f"Active task: {stats['active_task_title']}")

    try:
        # Test monitoring (simulate app usage)
        print(f"\nSimulating inactivity monitoring...")
        print(
            f"Inactivity threshold: {task_manager.inactivity_threshold_minutes} minutes"
        )
        print("Press Ctrl+C to stop")

        while True:
            time.sleep(10)

            # Simulate occasional app activity
            import random

            if random.random() < 0.3:  # 30% chance of activity
                task_manager.update_task_activity("code")

            stats = task_manager.get_task_statistics()
            if stats["active_task_title"]:
                print(f"Active task: {stats['active_task_title']}")

    except KeyboardInterrupt:
        print("\n[TaskManager] Stopping...")
        task_manager.stop_inactivity_monitoring()
