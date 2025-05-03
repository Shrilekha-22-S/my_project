# my_project
# Enhancing Road Safety with AI-driven Traffic Accident Analysis and Prediction

**Student Name:** Shrilekha S  
**Register Number:** 412723205047
**Institution:** Tagore Engineering College  
**Department:** Information Technology  
**Date of Submission:** 3rd May, 2025  

## 📌 GitHub Repository Link
> _Provide the link here after uploading the project._

---

## 📖 Table of Contents
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Workflow](#workflow)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Results & Insights](#results--insights)
- [Technologies Used](#technologies-used)
- [Team Members](#team-members)
- [How to Run the Code](#how-to-run-the-code)

---

## 🧠 Problem Statement

Road accidents cause immense loss of life and property each year. Traditional analysis is often reactive. This project utilizes AI to predict accident severity using features like weather, road type, and lighting to assist authorities in improving road safety.

---

## 🎯 Objectives

- Predict accident severity using ML models.
- Identify critical factors influencing severity.
- Improve traffic safety via data-driven decisions.
- Ensure accuracy, interpretability, and usability.

---

## 🔁 Workflow

```text
Data Acquisition → Preprocessing → EDA → Feature Engineering → Model Training → Evaluation → Interpretation → Reporting
# Handling missing values
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Merging datasets
merged_df = accidents.merge(vehicles, on='Accident_Index').merge(casualties, on='Accident_Index')

# Encoding categorical variables
df_encoded = pd.get_dummies(df, columns=['Road_Type', 'Weather_Conditions'])

# Outlier removal
df = df[(df['Speed_limit'] > df['Speed_limit'].quantile(0.01)) & 
        (df['Speed_limit'] < df['Speed_limit'].quantile(0.99))]

# Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Extract time-based features
df['Hour'] = df['Time'].dt.hour
df['Weekday'] = df['Date'].dt.weekday

# Binning speed
df['Speed_Category'] = pd.cut(df['Speed_limit'], bins=[0, 30, 50, 70], labels=['Low', 'Medium', 'High'])

# Interaction term
df['Light_Weather'] = df['Light_Conditions'] + "_" + df['Weather_Conditions']

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y)

# Model
model = XGBClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

