# 🎬 MovieIQ – Predictive Analytics on Film Success

## 📌 Project Overview

MovieIQ is a Machine Learning and Data Analytics project that predicts whether a movie is likely to be commercially successful based on historical movie data.

The project combines data preprocessing, exploratory data analysis (EDA), statistical testing, machine learning, and an interactive Streamlit dashboard to provide meaningful insights into movie performance.

---

## 🎯 Objective

The objective of this project is to:

- Analyze movie data to identify factors influencing movie success.
- Predict whether a movie will be successful using a Machine Learning model.
- Provide business insights and recommendations for decision-making.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SciPy
- Streamlit

---

## 📂 Dataset

The dataset contains information about movies, including:

- Budget
- Revenue
- Popularity
- Runtime
- Vote Average
- Title
- Genres

A movie is considered **Successful** if:

```
Revenue > Budget
```

Otherwise, it is classified as **Unsuccessful**.

---

## 📊 Features

### Data Preparation

- Data Cleaning
- Missing Value Handling
- Removal of Invalid Records
- Target Variable Creation

### Exploratory Data Analysis

- Dataset Preview
- Descriptive Statistics
- Success Distribution
- Budget vs Revenue Analysis
- Genre Trends
- Vote Average Distribution
- Correlation Heatmap

### Statistical Analysis

- Pearson Correlation
- Independent T-Test
- Chi-Square Test

### Machine Learning

- Random Forest Classifier
- Train-Test Split
- Accuracy Evaluation
- Classification Report
- Confusion Matrix
- Feature Importance

### Prediction

Users can enter:

- Budget
- Popularity
- Runtime
- Vote Average

The application predicts whether the movie is likely to be:

- ✅ Successful
- ❌ Unsuccessful

along with the prediction confidence.

### Business Analytics

- Business Insights
- Business Recommendations
- Download Filtered Dataset

---

## 🚀 How to Run

### Install the required libraries

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run MovieIQ.py
```

---

## 📸 Dashboard

The dashboard includes:

- KPI Cards
- Interactive Filters
- Data Visualizations
- Statistical Analysis
- Machine Learning Results
- Movie Success Prediction
- Business Insights
- Business Recommendations

---

## 📈 Machine Learning Workflow

1. Load Dataset
2. Clean Data
3. Create Target Variable
4. Perform Exploratory Data Analysis
5. Conduct Statistical Tests
6. Train Random Forest Classifier
7. Evaluate Model Performance
8. Predict Movie Success

---

## 💡 Future Enhancements

- Compare multiple Machine Learning models.
- Add more movie features such as cast, director, and release date.
- Improve prediction accuracy using advanced algorithms.
- Deploy the application with additional interactive visualizations.

---

## 👩‍💻 Author

**Manjula**

Electronics and Communication Engineering

Machine Learning & Data Analytics Project

---

## 📄 License

This project is developed for educational and learning purposes.
