# AI distraction prediction - tries to predict when user gets distracted
# Uses machine learning with sklearn

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


class DistractionPredictor:
    # ML model to predict when user might get distracted

    def __init__(
        self,
        data_file: str = "distraction_data.json",
        model_file: str = "distraction_model.pkl",
    ):
        self.data_file = data_file
        self.model_file = model_file
        self.model = None
        self.training_data = self.load_training_data()
        self.feature_columns = [
            "hour_of_day",
            "day_of_week",
            "session_duration",
            "app_category_encoded",
            "previous_session_duration",
        ]

        # try to load existing model or create new one
        self.load_model()

        print("[DistractionPredictor] Ready!")

    def load_training_data(self):
        # load training data from json file
        if not os.path.exists(self.data_file):
            print(f"[DistractionPredictor] No training data found, starting fresh")
            return []

        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
            print(f"[DistractionPredictor] Got {len(data)} training samples")
            return data
        except Exception as e:
            print(f"[DistractionPredictor] Couldn't load training data: {e}")
            return []

    def save_training_data(self):
        # save all the training data to file
        try:
            with open(self.data_file, "w") as f:
                json.dump(self.training_data, f, indent=2)
            print(f"[DistractionPredictor] Saved {len(self.training_data)} samples")
        except Exception as e:
            print(f"[DistractionPredictor] Error saving data: {e}")

    def encode_app_category(self, category: str) -> int:
        # convert app category to number for ML algorithm
        category_mapping = {
            "Work": 0,
            "Productivity": 1,
            "Social": 2,
            "Entertainment": 3,
            "Gaming": 4,
            "Other": 5,
        }
        return category_mapping.get(category, 5)  # default to "Other"

    def add_training_sample(
        self,
        session_data: Dict,
        was_distracted: bool,
        distraction_time_minutes: float = None,
    ) -> None:
        """
        Add a new training sample based on user behavior

        Args:
            session_data: Dict containing session information
            was_distracted: Boolean indicating if user got distracted
            distraction_time_minutes: How long before distraction occurred
        """
        try:
            start_time = datetime.fromisoformat(session_data["start_time"])

            # Extract features
            sample = {
                "timestamp": session_data["start_time"],
                "hour_of_day": start_time.hour,
                "day_of_week": start_time.weekday(),
                "app_name": session_data["app_name"],
                "app_category": session_data["category"],
                "app_category_encoded": self.encode_app_category(
                    session_data["category"]
                ),
                "session_duration": session_data["duration_seconds"]
                / 60,  # Convert to minutes
                "was_distracted": was_distracted,
                "distraction_time_minutes": (
                    distraction_time_minutes if was_distracted else None
                ),
            }

            # Add previous session duration if available
            if len(self.training_data) > 0:
                previous_duration = self.training_data[-1].get("session_duration", 0)
                sample["previous_session_duration"] = previous_duration
            else:
                sample["previous_session_duration"] = 0

            self.training_data.append(sample)

            # Auto-save every 10 samples
            if len(self.training_data) % 10 == 0:
                self.save_training_data()

            print(
                f"[DistractionPredictor] Added training sample: {session_data['app_name']} - Distracted: {was_distracted}"
            )

        except Exception as e:
            print(f"[DistractionPredictor] Error adding training sample: {e}")

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training the ML model

        Returns:
            X: Feature matrix
            y: Target variable (distraction time in minutes)
        """
        if len(self.training_data) < 10:
            print(
                "[DistractionPredictor] Not enough training data (minimum 10 samples)"
            )
            return None, None

        # Create DataFrame
        df = pd.DataFrame(self.training_data)

        # Filter out samples without distraction time
        df_distracted = df[df["was_distracted"] == True].copy()

        if len(df_distracted) < 5:
            print("[DistractionPredictor] Not enough distraction samples for training")
            return None, None

        # Prepare features
        X = df_distracted[self.feature_columns].values
        y = df_distracted["distraction_time_minutes"].values

        # Handle any missing values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=30.0)  # Default 30 minutes if missing

        return X, y

    def train_model(self) -> bool:
        """
        Train the distraction prediction model

        Returns:
            True if training successful, False otherwise
        """
        X, y = self.prepare_training_data()

        if X is None or y is None:
            print("[DistractionPredictor] Cannot train model - insufficient data")
            return False

        try:
            # Split data for training and testing
            if len(X) > 20:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y

            # Try Random Forest first (generally better for this type of problem)
            self.model = RandomForestRegressor(
                n_estimators=50, random_state=42, max_depth=10
            )
            self.model.fit(X_train, y_train)

            # Evaluate model if we have test data
            if len(X_test) > 0:
                y_pred = self.model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                print(f"[DistractionPredictor] Model trained successfully")
                print(f"[DistractionPredictor] MAE: {mae:.2f} minutes, R²: {r2:.3f}")

            # Save the trained model
            self.save_model()
            return True

        except Exception as e:
            print(f"[DistractionPredictor] Error training model: {e}")
            return False

    def predict_distraction_time(self, session_data: Dict) -> Optional[float]:
        """
        Predict how long until user gets distracted

        Args:
            session_data: Current session information

        Returns:
            Predicted time until distraction in minutes, or None if prediction not possible
        """
        if self.model is None:
            print("[DistractionPredictor] No trained model available")
            return None

        try:
            start_time = datetime.fromisoformat(session_data["start_time"])

            # Prepare features for prediction
            features = [
                start_time.hour,  # hour_of_day
                start_time.weekday(),  # day_of_week
                session_data["duration_seconds"] / 60,  # session_duration
                self.encode_app_category(
                    session_data["category"]
                ),  # app_category_encoded
                self.get_previous_session_duration(),  # previous_session_duration
            ]

            # Make prediction
            features_array = np.array([features])
            prediction = self.model.predict(features_array)[0]

            # Ensure reasonable bounds (between 1 and 120 minutes)
            prediction = max(1.0, min(120.0, prediction))

            print(
                f"[DistractionPredictor] Predicted distraction time: {prediction:.1f} minutes for {session_data['app_name']}"
            )
            return prediction

        except Exception as e:
            print(f"[DistractionPredictor] Error making prediction: {e}")
            return None

    def extract_features(
        self,
        app_name,
        time_on_app,
        total_apps,
        total_memory,
        current_hour,
        switches_today,
    ):
        """
        Extract features for prediction in the format expected by main.py

        Args:
            app_name: Name of current application
            time_on_app: Time spent on current app in minutes
            total_apps: Total number of running apps
            total_memory: Total memory usage
            current_hour: Current hour of day
            switches_today: Number of app switches today

        Returns:
            List of features for prediction
        """
        try:
            # Determine app category
            app_category = self._categorize_app(app_name)
            app_category_encoded = self.encode_app_category(app_category)

            # Get previous session duration
            previous_duration = self.get_previous_session_duration()

            # Create feature vector matching the training data format
            features = [
                current_hour,  # hour_of_day
                datetime.now().weekday(),  # day_of_week
                time_on_app,  # session_duration
                app_category_encoded,  # app_category_encoded
                previous_duration,  # previous_session_duration
            ]

            return features

        except Exception as e:
            print(f"[DistractionPredictor] Error extracting features: {e}")
            return [0, 0, 0, 0, 0]  # Default fallback features

    def predict(self, features):
        """
        Make prediction using the provided features

        Args:
            features: List of feature values

        Returns:
            Probability of distraction (0.0 to 1.0)
        """
        try:
            if self.model is None:
                # Fallback heuristic prediction if no model
                time_on_app = features[2] if len(features) > 2 else 0
                hour = features[0] if len(features) > 0 else 12

                # Simple heuristic
                if time_on_app > 60:  # More than 1 hour
                    return 0.8
                elif time_on_app > 30:  # More than 30 minutes
                    return 0.6
                elif hour < 9 or hour > 17:  # Outside work hours
                    return 0.7
                else:
                    return 0.3

            # Use trained model for prediction
            features_array = np.array([features])

            # The model predicts distraction time, convert to probability
            predicted_time = self.model.predict(features_array)[0]

            # Convert time to probability (shorter time = higher probability)
            # Normalize: 1-120 minutes -> 0.9-0.1 probability
            probability = max(0.1, min(0.9, 1.0 - (predicted_time - 1) / 119))

            return probability

        except Exception as e:
            print(f"[DistractionPredictor] Error in prediction: {e}")
            return 0.3  # Default moderate probability

    def _categorize_app(self, app_name):
        """
        Categorize an application by its name

        Args:
            app_name: Name of the application

        Returns:
            Category string
        """
        app_name_lower = app_name.lower()

        if any(
            browser in app_name_lower
            for browser in ["chrome", "firefox", "edge", "safari", "opera"]
        ):
            return "Browser"
        elif any(
            social in app_name_lower
            for social in ["discord", "slack", "teams", "whatsapp", "telegram"]
        ):
            return "Communication"
        elif any(
            media in app_name_lower
            for media in ["youtube", "netflix", "spotify", "vlc", "media"]
        ):
            return "Entertainment"
        elif any(
            code in app_name_lower
            for code in ["code", "studio", "pycharm", "intellij", "eclipse"]
        ):
            return "Productivity"
        elif any(
            game in app_name_lower for game in ["game", "steam", "epic", "origin"]
        ):
            return "Gaming"
        else:
            return "Other"

    def get_previous_session_duration(self) -> float:
        """
        Get duration of the previous session for feature engineering
        """
        if len(self.training_data) > 0:
            return self.training_data[-1].get("session_duration", 0)
        return 0

    def get_distraction_probability(self, session_data: Dict) -> float:
        """
        Get probability of distraction based on historical patterns

        Returns:
            Probability between 0 and 1
        """
        if len(self.training_data) < 5:
            return 0.5  # Default neutral probability

        try:
            # Analyze similar sessions in training data
            current_hour = datetime.fromisoformat(session_data["start_time"]).hour
            current_category = session_data["category"]
            current_duration = session_data["duration_seconds"] / 60

            similar_sessions = []
            for sample in self.training_data:
                # Find sessions with similar characteristics
                if (
                    abs(sample["hour_of_day"] - current_hour) <= 2
                    and sample["app_category"] == current_category
                ):
                    similar_sessions.append(sample)

            if not similar_sessions:
                return 0.5

            # Calculate distraction rate for similar sessions
            distracted_count = sum(1 for s in similar_sessions if s["was_distracted"])
            probability = distracted_count / len(similar_sessions)

            # Adjust based on current session duration
            if (
                current_duration > 30
            ):  # Longer sessions = higher distraction probability
                probability = min(1.0, probability * 1.2)

            return probability

        except Exception as e:
            print(
                f"[DistractionPredictor] Error calculating distraction probability: {e}"
            )
            return 0.5

    def save_model(self) -> None:
        """
        Save the trained model to disk
        """
        if self.model is None:
            return

        try:
            joblib.dump(self.model, self.model_file)
            print(f"[DistractionPredictor] Model saved to {self.model_file}")
        except Exception as e:
            print(f"[DistractionPredictor] Error saving model: {e}")

    def load_model(self) -> None:
        """
        Load the trained model from disk
        """
        if not os.path.exists(self.model_file):
            print("[DistractionPredictor] No saved model found, will train new model")
            return

        try:
            self.model = joblib.load(self.model_file)
            print(f"[DistractionPredictor] Model loaded from {self.model_file}")
        except Exception as e:
            print(f"[DistractionPredictor] Error loading model: {e}")
            self.model = None

    def get_insights(self) -> Dict:
        """
        Get insights about distraction patterns
        """
        if len(self.training_data) < 5:
            return {
                "total_samples": len(self.training_data),
                "distraction_rate": 0,
                "avg_distraction_time": 0,
                "most_distracting_hour": "Unknown",
                "most_distracting_category": "Unknown",
            }

        df = pd.DataFrame(self.training_data)

        # Calculate basic statistics
        total_samples = len(df)
        distraction_rate = df["was_distracted"].mean()

        distracted_df = df[df["was_distracted"] == True]
        avg_distraction_time = (
            distracted_df["distraction_time_minutes"].mean()
            if len(distracted_df) > 0
            else 0
        )

        # Find most distracting patterns
        most_distracting_hour = (
            df[df["was_distracted"] == True]["hour_of_day"].mode().iloc[0]
            if len(distracted_df) > 0
            else "Unknown"
        )
        most_distracting_category = (
            df[df["was_distracted"] == True]["app_category"].mode().iloc[0]
            if len(distracted_df) > 0
            else "Unknown"
        )

        return {
            "total_samples": total_samples,
            "distraction_rate": round(distraction_rate * 100, 1),
            "avg_distraction_time": round(avg_distraction_time, 1),
            "most_distracting_hour": most_distracting_hour,
            "most_distracting_category": most_distracting_category,
        }

    def simulate_training_data(self, num_samples: int = 50) -> None:
        """
        Generate some realistic training data for testing purposes
        """
        import random

        categories = ["Work", "Social", "Entertainment", "Productivity"]
        apps = {
            "Work": ["code", "word", "excel", "figma"],
            "Social": ["instagram", "facebook", "twitter", "whatsapp"],
            "Entertainment": ["youtube", "netflix", "spotify"],
            "Productivity": ["chrome", "firefox", "notepad"],
        }

        print(
            f"[DistractionPredictor] Generating {num_samples} sample training data points..."
        )

        for i in range(num_samples):
            # Random time and category
            hour = random.randint(8, 22)
            day = random.randint(0, 6)
            category = random.choice(categories)
            app = random.choice(apps[category])

            # Simulate distraction patterns
            # Social and Entertainment apps more likely to cause distraction
            if category in ["Social", "Entertainment"]:
                was_distracted = random.random() < 0.7
                base_distraction_time = random.uniform(5, 30)
            else:
                was_distracted = random.random() < 0.3
                base_distraction_time = random.uniform(15, 60)

            # Evening hours more distracting
            if hour >= 18:
                was_distracted = random.random() < 0.6

            distraction_time = base_distraction_time if was_distracted else None
            session_duration = random.uniform(2, 45)

            # Create sample data
            timestamp = datetime.now() - timedelta(
                days=random.randint(0, 30), hours=random.randint(0, 23)
            )

            sample = {
                "timestamp": timestamp.isoformat(),
                "hour_of_day": hour,
                "day_of_week": day,
                "app_name": app,
                "app_category": category,
                "app_category_encoded": self.encode_app_category(category),
                "session_duration": session_duration,
                "previous_session_duration": random.uniform(1, 30),
                "was_distracted": was_distracted,
                "distraction_time_minutes": distraction_time,
            }

            self.training_data.append(sample)

        self.save_training_data()
        print(f"[DistractionPredictor] Generated {num_samples} training samples")


# Example usage and testing
if __name__ == "__main__":
    # Create predictor
    predictor = DistractionPredictor()

    # Generate some sample data if no training data exists
    if len(predictor.training_data) < 10:
        predictor.simulate_training_data(100)

    # Train the model
    success = predictor.train_model()

    if success:
        # Test prediction with sample data
        test_session = {
            "app_name": "instagram",
            "category": "Social",
            "start_time": datetime.now().isoformat(),
            "duration_seconds": 300,  # 5 minutes so far
        }

        prediction = predictor.predict_distraction_time(test_session)
        probability = predictor.get_distraction_probability(test_session)

        print(f"\nTest Prediction:")
        print(f"App: {test_session['app_name']}")
        print(f"Predicted distraction time: {prediction:.1f} minutes")
        print(f"Distraction probability: {probability:.1%}")

        # Show insights
        insights = predictor.get_insights()
        print(f"\nInsights:")
        print(f"Total training samples: {insights['total_samples']}")
        print(f"Distraction rate: {insights['distraction_rate']}%")
        print(f"Average distraction time: {insights['avg_distraction_time']} minutes")
        print(f"Most distracting hour: {insights['most_distracting_hour']}:00")
        print(f"Most distracting category: {insights['most_distracting_category']}")
