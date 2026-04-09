# Traffic Situation Prediction System

**Project Overview**

This project focused on developing a comprehensive Traffic Situation Prediction system using real-world traffic datasets. The objective was to predict traffic conditions based on vehicle counts, time, and day of the week. The workflow included data preprocessing, exploratory data analysis (EDA), feature engineering, model development, evaluation, hyperparameter tuning, and model comparison using both machine learning and deep learning approaches.

**Data Understanding and Preprocessing**

##### Datasets Used: 
- Two traffic datasets (Traffic.csv and TrafficTwoMonth.csv) containing:
     - Car, Bike, Bus, Truck counts
     - Total traffic
     - Time
     - Day of the week

##### Key Observations:
- Combined dataset contained duplicate rows, which were removed to ensure data quality.
- No significant missing values were observed.
- The target variable (Traffic Situation) showed class imbalance, requiring oversampling techniques.

Data Preprocessing Steps:
- Duplicate rows removed
- Label encoding applied to target variable (Traffic Situation)
- One-hot encoding applied to categorical feature (Day of the Week)
- Numerical features scaled using StandardScaler
- SMOTE applied to handle class imbalance in training data
  
**Exploratory Data Analysis (EDA)**
1. Traffic Situation Distribution:
Count plot revealed class imbalance across traffic categories

<img width="704" height="468" alt="download" src="https://github.com/user-attachments/assets/9ba6d63e-aa61-48b1-8fc1-3259b4d9f21f" />

2. Vehicle Count Distribution:
Histograms showed skewed distributions for Car, Bike, Bus, and Truck counts
Indicated peak traffic behavior during specific time periods

<img width="995" height="680" alt="download" src="https://github.com/user-attachments/assets/a451c794-b036-40eb-b0be-47b2070e6415" />

3. Traffic vs Day of Week:
Boxplots indicated higher traffic during weekdays compared to weekends

<img width="850" height="448" alt="download" src="https://github.com/user-attachments/assets/17f0a3ab-2233-4014-ac57-e9283baf11cb" />

4. Traffic Pattern Across Time:
Line plot (Hour vs Total traffic) showed morning and evening peak congestion

<img width="850" height="448" alt="download" src="https://github.com/user-attachments/assets/e19cbaaf-42fd-463c-8a27-6ecc2d3145ad" />

5. Vehicle Mix by Traffic Situation:
Different traffic classes showed distinct vehicle composition patterns

<img width="1115" height="986" alt="download" src="https://github.com/user-attachments/assets/f130a54b-e944-453f-9c6b-4d183afe5682" />

##### Key EDA Insight:

Traffic patterns are strongly influenced by time, day of week, and vehicle composition, validating their use as predictive features.

**Feature Engineering**

Cyclical Encoding for Time:
- Used sin and cos transformations to represent cyclical nature of hours:
    Time_sin, Time_cos

Vehicle Ratio Features:
- Created normalized features:
   CarRatio, BikeRatio, BusRatio, TruckRatio

Outcome:

Feature engineering improved model understanding of relative traffic composition rather than raw counts.

**Model Development and Evaluation**
1. ANN
Architecture:
- 2 hidden layers (64, 32 neurons)
- ReLU activation
- Softmax output layer
- Training: Early stopping applied (patience = 10)
- Performance: Test Accuracy: ~0.8870
- Observation: Captured non-linear relationships effectively and Performed well on majority classes but struggled slightly with minority classes

2. Random Forest
- Performance: Accuracy: ~0.8538
- Observation: Robust to noise and scaling
- Performed well on dominant classes
- Limited performance on minority classes
  
3. XGBoost
- Performance: Accuracy: ~0.8632
Observation:
- Captured complex feature interactions effectively
- Strong generalization performance
- Better balance between precision and recall compared to Random Forest
  
4. ANN with Hyperparameter Tuning (Keras Tuner)
Optimization:
- Tuned hidden layer units
- Tuned learning rate
- Used Random Search strategy
- Performance: Test Accuracy: ~0.8767
Observation:
- Improved generalization compared to baseline ANN
- Reduced manual tuning effort significantly
  
**Comparative Model Performance**

Model - Test Accuracy

ANN	       -       0.8870

Tuned ANN	-      0.8767

Random Forest  -	0.8538

XGBoost	   -       0.8632

**Best Model**
- Vanilla ANN achieved the highest test accuracy (0.8870)
- Recommended for deployment when using deep learning approach
- XGBoost is a strong alternative due to robustness and interpretability
  
**Final Insight**
- SMOTE improved minority class predictions significantly
- ANN models captured complex patterns better than traditional ML models
- Tree-based models provided strong baseline performance with interpretability
  
**Model Saving**
- Final trained ANN model, scaler, encoding saved as:
    - ann_model.keras
    - scaler.pkl
    - label_encoder.pkl
- Can be used for real-time traffic situation prediction

**Model Deployment**
The trained traffic prediction model was deployed using a Streamlit web application to enable real-time, user-friendly predictions. The deployment integrates the saved ANN model (ann_model.keras), along with preprocessing components such as the StandardScaler and LabelEncoder to ensure consistency between training and inference. Users input vehicle counts, time, and day of the week through an interactive interface, after which feature engineering (including vehicle ratios, cyclical time encoding, and day-wise encoding) is applied. The processed input is scaled and passed to the model to predict the traffic condition, and the output is converted into a readable label and displayed with a confidence score and visual indicators. This deployment demonstrates how a machine learning model can be effectively transformed into a practical, real-world application for intelligent traffic management.

<img width="1098" height="787" alt="Screenshot 2026-04-09 210031" src="https://github.com/user-attachments/assets/29f397aa-21ff-4443-b6d1-3437b01dc2dc" />

<img width="1100" height="221" alt="Screenshot 2026-04-09 210052" src="https://github.com/user-attachments/assets/e1f9aa29-19cb-4576-b5bc-9fc4d6b094ef" />

**Business Impact**

This system enables:
- Prediction of traffic congestion in advance
- Optimization of traffic signal timings
- Improved urban traffic management
- Better commuter experience
- Support for smart city planning initiatives
