# ProcastiNo 🚀

**ProcastiNo** is an intelligent productivity and focus management application designed to help users overcome procrastination and maintain focus on their tasks. Built with Python and PyQt5, it combines task management, application tracking, distraction prediction, and smart break reminders to create a comprehensive productivity solution.

## 🌟 Features

### 📋 Task Management

- **Smart Task Creation**: Create and organize tasks with priorities, deadlines, and app assignments
- **Progress Tracking**: Monitor task completion and productivity metrics
- **Deadline Management**: Get timely reminders for upcoming deadlines
- **App Integration**: Assign specific applications to tasks for focused work

### 🤖 Intelligent Monitoring

- **Application Tracking**: Real-time monitoring of active applications and window titles
- **Distraction Detection**: Machine learning-powered prediction of potential distractions
- **Focus Analytics**: Detailed insights into productivity patterns and app usage
- **Inactivity Detection**: Smart notifications when you've been away from productive work

### 🧠 Machine Learning Features

- **Distraction Prediction**: AI model that learns your patterns to predict when you might get distracted
- **Productivity Analytics**: Advanced analytics with charts and insights
- **Behavioral Learning**: Adapts to your work patterns over time

### ⏰ Break Management

- **Smart Break Reminders**: Intelligent break suggestions based on work patterns
- **Customizable Intervals**: Set personalized break schedules
- **Health Focused**: Promotes healthy work-life balance

### 🎨 Modern UI

- **Responsive Design**: Clean, modern interface that adapts to your workflow
- **Dark/Light Themes**: Customizable appearance
- **Real-time Updates**: Live data visualization and progress tracking
- **Intuitive Navigation**: Easy-to-use tabbed interface

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- Windows OS (for full functionality)

### Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Laksh-Chawla/ProcastiNo.git
   cd ProcastiNo
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

## 📦 Dependencies

### Core Dependencies

- **PyQt5** (5.15.10) - GUI framework
- **psutil** (5.9.8) - System and process monitoring
- **pygetwindow** (0.0.9) - Window management
- **pywin32** (306) - Windows API integration
- **plyer** (2.1.0) - Cross-platform notifications

### Machine Learning

- **scikit-learn** (1.4.2) - ML algorithms for distraction prediction
- **numpy** (1.26.4) - Numerical computations
- **pandas** (2.2.2) - Data manipulation and analysis
- **matplotlib** (3.8.4) - Data visualization
- **joblib** (1.4.2) - Model persistence

## 🏗️ Project Structure

```
ProcastiNo/
├── main.py                 # Main application entry point
├── requirements.txt        # Project dependencies
├── setup.py               # Package setup configuration
├── .gitignore             # Git ignore rules
├── modules/               # Core functionality modules
│   ├── __init__.py
│   ├── analytics.py       # Analytics and data visualization
│   ├── app_tracker.py     # Application monitoring
│   ├── break_reminder.py  # Break management system
│   ├── distraction_predictor.py  # ML-based distraction prediction
│   └── task_manager.py    # Task management core
└── ui/                    # User interface components
    └── responsive_procastino_ui.py  # Main UI implementation
```

## 🚀 Usage

### Getting Started

1. Launch the application using `python main.py`
2. Create your first task using the "Add Task" button
3. Assign an application to your task for focused tracking
4. Start working and let ProcastiNo monitor your productivity

### Key Workflows

#### Task Management

- **Create Tasks**: Add new tasks with titles, descriptions, and priorities
- **Set Deadlines**: Assign deadlines to get timely reminders
- **Track Progress**: Monitor completion status and productivity metrics
- **App Assignment**: Link tasks to specific applications for better tracking

#### Productivity Monitoring

- **Real-time Tracking**: View live application usage and focus metrics
- **Analytics Dashboard**: Access detailed productivity insights and charts
- **Distraction Alerts**: Receive intelligent notifications about potential distractions
- **Break Reminders**: Get personalized break suggestions based on your work patterns

#### Customization

- **Break Intervals**: Adjust break reminder frequencies
- **Notification Settings**: Customize alert preferences
- **UI Themes**: Switch between different visual themes
- **Priority Levels**: Organize tasks by importance

## 🧠 Machine Learning Features

ProcastiNo includes sophisticated ML capabilities:

- **Distraction Prediction**: Uses Random Forest algorithms to predict when you might get distracted
- **Pattern Recognition**: Learns from your work habits to provide personalized insights
- **Adaptive Reminders**: Adjusts break and focus reminders based on your productivity patterns
- **Behavioral Analytics**: Provides deep insights into your work behavior

## 🔧 Configuration

The application stores configuration and data in JSON files:

- `session_data.json` - Current session data and settings
- Model files are automatically saved for ML features

## 🤝 Contributing

We welcome contributions to ProcastiNo! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

1. Follow the installation steps above
2. Install development dependencies (if any)
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📊 Features in Development

- [ ] Cross-platform support (macOS, Linux)
- [ ] Cloud synchronization
- [ ] Team collaboration features
- [ ] Mobile companion app
- [ ] Advanced AI insights
- [ ] Integrations with popular productivity tools

## 🐛 Bug Reports

Found a bug? Please open an issue with:

- Detailed description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyQt5 for the beautiful user interface
- Machine learning powered by scikit-learn
- System monitoring capabilities provided by psutil
- Cross-platform notifications via plyer

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Laksh-Chawla/ProcastiNo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Laksh-Chawla/ProcastiNo/discussions)

---

**ProcastiNo** - Because procrastination is not an option! 💪
