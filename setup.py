"""
ProcastiNo Setup and Installation Script
Automatically sets up the environment and installs dependencies
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path(".venv")

    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True

    print("📦 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False


def get_python_executable():
    """Get the path to the Python executable in the virtual environment"""
    if os.name == "nt":  # Windows
        return Path(".venv") / "Scripts" / "python.exe"
    else:  # Linux/Mac
        return Path(".venv") / "bin" / "python"


def install_dependencies():
    """Install required dependencies"""
    python_exe = get_python_executable()

    if not python_exe.exists():
        print("❌ Virtual environment Python not found")
        return False

    print("📋 Installing dependencies...")
    try:
        # Install dependencies from requirements.txt
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_config_file():
    """Create a default configuration file"""
    config = {
        "app_tracker": {"tracking_interval_seconds": 2, "data_retention_days": 30},
        "distraction_predictor": {
            "retrain_interval_hours": 24,
            "minimum_training_samples": 50,
        },
        "task_manager": {
            "inactivity_threshold_minutes": 5,
            "reminder_interval_minutes": 5,
        },
        "break_reminder": {"work_threshold_minutes": 45, "break_duration_minutes": 15},
        "ui": {"theme": "dark", "update_interval_seconds": 5},
    }

    config_path = Path("config.json")
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("✅ Created default configuration file")
    else:
        print("✅ Configuration file already exists")

    return True


def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)"""
    if os.name != "nt":
        return True

    try:
        import winshell
        from win32com.client import Dispatch

        desktop = winshell.desktop()
        path = os.path.join(desktop, "ProcastiNo.lnk")
        target = str(Path.cwd() / "run.bat")
        wDir = str(Path.cwd())
        icon = str(Path.cwd() / "icon.jpeg")

        shell = Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = target
        shortcut.WorkingDirectory = wDir
        shortcut.IconLocation = icon
        shortcut.save()

        print("✅ Desktop shortcut created")
        return True
    except ImportError:
        print("⚠️  Desktop shortcut creation skipped (optional dependency missing)")
        return True
    except Exception as e:
        print(f"⚠️  Could not create desktop shortcut: {e}")
        return True


def run_tests():
    """Run the test suite to verify installation"""
    python_exe = get_python_executable()

    print("🧪 Running installation tests...")
    try:
        result = subprocess.run(
            [str(python_exe), "quick_start_test.py"], capture_output=True, text=True
        )

        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 ProcastiNo Setup and Installation")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        return False

    # Create virtual environment
    if not create_virtual_environment():
        return False

    # Install dependencies
    if not install_dependencies():
        return False

    # Create configuration
    if not create_config_file():
        return False

    # Create desktop shortcut (optional)
    create_desktop_shortcut()

    # Run tests
    if not run_tests():
        print("\n⚠️  Installation completed but tests failed")
        print("   You may still try running the application")
    else:
        print("\n🎉 Installation completed successfully!")

    print("\n" + "=" * 50)
    print("📋 Next Steps:")
    print("1. Run the application:")
    print("   • Double-click run.bat (Windows)")
    print("   • Or run: python main.py")
    print("2. Create your first task in the Tasks tab")
    print("3. Let the app track your usage for insights")
    print("4. Check the Analytics tab for visualizations")

    print("\n📚 Documentation:")
    print("• README.md - Full documentation")
    print("• demo.py - Feature demonstrations")
    print("• config.json - Application settings")

    print("\n🔗 Useful Commands:")
    print("• python demo.py - Run feature demo")
    print("• python quick_start_test.py - Test installation")
    print("• python modules/[module].py - Test individual modules")

    return True


if __name__ == "__main__":
    success = main()

    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")

        # Ask if user wants to run the app now
        try:
            choice = (
                input("\nWould you like to run the demo now? (y/n): ").lower().strip()
            )
            if choice in ["y", "yes"]:
                print("\nStarting demo...")
                python_exe = get_python_executable()
                subprocess.run([str(python_exe), "demo.py"])
        except KeyboardInterrupt:
            print("\nSetup complete. Run 'python main.py' to start the application.")
