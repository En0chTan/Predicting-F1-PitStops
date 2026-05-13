# 🏎️ F1 Pit Stop Predictor

A comprehensive machine learning and analytics dashboard for predicting and analyzing Formula 1 pit stop decisions using real-world telemetry data.

---

## Overview

This project combines **exploratory data analysis (EDA)** with **machine learning models** to understand and predict pit stop decisions in Formula 1 racing. The interactive Streamlit dashboard provides insights into pit stop patterns, tyre performance, driver behavior, and race dynamics, while dual ML models (Logistic Regression & Neural Networks) enable accurate pit stop prediction.

**Key Insight:** Tyre life is the strongest predictor of pit stops, but compound type, stint number, race progress, and driver behavior all play significant roles in pit stop strategy.

---

## Features

### **Exploratory Data Analysis Dashboard**

- **Overview Tab**
  - Key performance metrics (pit stop rate, average tyre life, lap times)
  - Pit stop distribution analysis
  - Pit stop timing by lap number and stint
  - Comparison of pit stop vs pit next lap predictions

- **Tyre Analysis Tab**
  - Tyre compound usage distribution
  - Tyre life statistics by compound
  - Impact of tyre life on pit stop decisions
  - Tyre compound evolution across seasons

- **Driver & Race Tab**
  - Top drivers by pit stop frequency
  - Longest average tyre life performance
  - Driver vs compound preference heatmap
  - Pit stop rates by race
  - Driver pit stop trends across years

- **Position & Stint Tab**
  - Race position impact on pit stop likelihood
  - Stint distribution analysis
  - Tyre life vs race position scatter plot
  - Position change dynamics
  - Feature correlation heatmap

- **Lap Time & Degradation Tab**
  - Lap time distribution by pit stop
  - Average lap times by compound
  - Lap time delta distribution
  - Cumulative tyre degradation analysis
  - Race progress vs degradation scatter plot

- **Season Trends Tab**
  - Year-over-year pit stop rate trends
  - Laps recorded per season
  - Average lap time evolution
  - Long-term strategy changes

### **Model Evaluation Page**

- **Dual Model Support**
  - Logistic Regression (fast, interpretable)
  - Neural Network/ANN (complex, high accuracy)

- **Comprehensive Metrics**
  - Accuracy, Precision, Recall, F1 Score, ROC AUC
  - Confusion matrix visualization
  - ROC curve with AUC scoring
  - Per-class classification reports

- **Feature Analysis**
  - Feature importance rankings
  - Direct coefficient interpretation (LR)
  - Model performance comparison

### **Interactive Controls**

- Real-time data filtering by:
  - Driver selection
  - Race selection
  - Season/Year range
  - Tyre compound
  - Stint range

- Model selection and training
- Real-time performance metrics update

---

## Project Structure

```
Predicting-F1-PitStops/
├── README.md                          # Project documentation
├── app.py                             # Streamlit application
├── Predicting_F1_Pit_Stops.ipynb      # Jupyter notebook with full analysis
├── train.csv                          # Training dataset
├── test.csv                           # Test dataset
└── requirements.txt                   # Python dependencies
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Predicting-F1-PitStops.git
   cd Predicting-F1-PitStops
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - Or navigate there manually if it doesn't open

---

## Models

### 1. **Logistic Regression**
- **Pros:** Fast training, highly interpretable, feature importance visible
- **Cons:** Limited to linear relationships
- **Use When:** You need explanations and fast predictions

**Parameters:**
- Solver: `newton-cg` (stable for smaller datasets)
- Max iterations: 1000
- Feature scaling: StandardScaler

### 2. **Neural Network (MLP Classifier)**
- **Pros:** Can capture non-linear patterns, potentially higher accuracy
- **Cons:** Slower training, less interpretable ("black box")
- **Use When:** You need maximum accuracy

**Architecture:**
- Hidden layers: 1 layer with 50 neurons
- Activation: ReLU
- Solver: Adam optimizer
- Max iterations: 500
- Feature scaling: StandardScaler

### Model Performance Metrics

Both models are evaluated on:
- **Accuracy:** Overall correctness
- **Precision:** Accuracy of positive predictions
- **Recall:** Coverage of actual positives
- **F1 Score:** Harmonic mean of precision & recall
- **ROC AUC:** Area under receiver operating characteristic curve

---

## Usage Guide

### Step 1: Load Your Data
- **Option A (Recommended):** Upload your `train.csv` file using the sidebar uploader
  - App validates data automatically
  - Shows row count and data source indicator
  
- **Option B:** Use synthetic mock data (default if no file uploaded)
  - 3,000 synthetic F1 records
  - Good for exploring dashboard functionality

### Step 2: Apply Filters (Optional)
Use sidebar controls to filter by:
- Drivers (select specific drivers to focus on)
- Races (analyze specific tracks)
- Years (compare seasons)
- Tyre compounds (compare compound strategies)
- Stint range (analyze specific stint phases)

### Step 3: Explore EDA Dashboard
Click through 6 tabs to understand your data:
1. Overview - Quick metrics and distributions
2. Tyre Analysis - Compound performance
3. Driver & Race - Performance by driver/track
4. Position & Stint - Strategic dynamics
5. Lap Time & Degradation - Performance trends
6. Season Trends - Long-term patterns

### Step 4: Compare ML Models
- Navigate to "Model Evaluation" page
- Select between **Logistic Regression** or **Neural Network**
- App trains model on filtered data automatically
- Review:
  - Performance metrics (5 scores + ROC curve)
  - Classification report by class
  - Feature importance (LR only)
  - Confusion matrix

### Step 5: Interpret Results
- **High Recall:** Model finds most pit stops (fewer false negatives)
- **High Precision:** Model confident in predictions (fewer false positives)
- **High F1:** Balanced performance across both metrics
- **High AUC:** Good at distinguishing pit stop vs no pit stop

---

## Key Findings from Analysis

Based on the notebook analysis, key insights include:

1. **Tyre Life is Dominant:** Strongest predictor of pit stops
2. **Compound Impact:** Soft tyres pit stop 20% more frequently
3. **Stint Progression:** Later stints (3+) show higher pit stop rates
4. **Race Position:** Lower positions (struggling cars) pit slightly more often
5. **Seasonal Trends:** Pit stop strategies have evolved across recent seasons
6. **Driver Differences:** Top drivers manage tyre life differently
7. **Lap Time Delta:** Performance drop signals imminent pit stops

---

## Tech Stack

### Core Libraries
- **Streamlit:** Interactive web dashboard framework
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computations
- **Scikit-learn:** Machine learning models and preprocessing
- **Plotly:** Interactive visualizations

### Machine Learning
- Logistic Regression (linear classifier)
- MLPClassifier (neural network)
- StandardScaler (feature normalization)
- LabelEncoder (categorical encoding)

### Data Visualization
- Plotly Express & Graph Objects (interactive charts)
- Heatmaps, scatter plots, bar charts, pie charts, ROC curves

---


## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn ML Guide](https://scikit-learn.org/stable/user_guide.html)
- [Plotly Interactive Charts](https://plotly.com/python/)
- [F1 Telemetry Data](https://www.formula1.com/)

---

## Contributing

Contributions are welcome! To improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes with clear commits
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request with detailed description

### Ideas for Contribution
- Add more ML algorithms (Random Forest, XGBoost, SVM)
- Implement real-time F1 data fetching
- Add SHAP explanations for neural networks
- Create model comparison dashboard
- Add hyperparameter tuning interface

---

## License

This project is licensed under the MIT License - see LICENSE file for details.