Bangalore House Price Prediction

A machine learning project to predict house prices in Bangalore based on features like location, square footage, BHK, and number of bathrooms.

--------------------------------------------------
OVERVIEW
--------------------------------------------------
This project implements a regression-based machine learning model to predict house prices in Bangalore using historical real estate data.

Key features used:
- Location
- Total Square Feet
- Number of Bathrooms
- BHK (Bedrooms)

--------------------------------------------------
DATASET
--------------------------------------------------
File: bengaluru_house_prices.csv

Initial Dataset:
- Records: 13,320
- Features: 9

Columns:
- area_type
- availability
- location
- size
- society
- total_sqft
- bath
- balcony
- price (in lakhs INR)

--------------------------------------------------
DATA CLEANING & FEATURE ENGINEERING
--------------------------------------------------
1. Dropped unnecessary columns:
   area_type, society, balcony, availability

2. Handled missing values:
   Dropped rows with null values
   Remaining records: 13,246

3. Feature extraction:
   Extracted BHK from size column

4. total_sqft cleaning:
   Converted ranges to averages
   Removed non-numeric values

5. Added price_per_sqft feature:
   price_per_sqft = (price * 100000) / total_sqft

6. Location dimensionality reduction:
   Grouped locations with <=10 entries into 'other'
   Final locations: 242

7. Outlier removal:
   - Removed properties with sqft/bhk < 300
   - Removed price_per_sqft outliers (mean ± std per location)
   - Removed BHK-based and bathroom-based anomalies

Final dataset size: 7,231 records

--------------------------------------------------
EXPLORATORY DATA ANALYSIS
--------------------------------------------------
Visualizations:
- Price per square foot distribution
- Bathroom count distribution
- Scatter plots for 2 BHK vs 3 BHK comparisons

--------------------------------------------------
MODEL BUILDING
--------------------------------------------------
1. One-hot encoded location feature
2. Train-test split:
   - Training: 80%
   - Testing: 20%

Models evaluated:
- Linear Regression
- Lasso Regression
- Decision Tree

Best model: Linear Regression

--------------------------------------------------
RESULTS
--------------------------------------------------
Test Accuracy: 87.6%
Cross-validation score: 0.8467

--------------------------------------------------
PREDICTION FUNCTION
--------------------------------------------------
predict_price(location, sqft, bath, bhk)

Example:
predict_price('Whitefield', 1500, 2, 2)

--------------------------------------------------
INSTALLATION
--------------------------------------------------
1. Clone repository
2. Install dependencies:
   pip install pandas numpy matplotlib scikit-learn
3. Place dataset in project directory

--------------------------------------------------
USAGE
--------------------------------------------------
Open Jupyter Notebook:
jupyter notebook ml_model.ipynb

--------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------
bangalore-house-price-prediction/
│
├── ml_model.ipynb
├── bengaluru_house_prices.csv
├── README.txt
└── requirements.txt

--------------------------------------------------
KEY INSIGHTS
--------------------------------------------------
- Location is a strong price indicator
- Price per sqft helps detect outliers
- Feature engineering improves model accuracy

--------------------------------------------------
FUTURE IMPROVEMENTS
--------------------------------------------------
- Add amenities-based features
- Use ensemble models
- Build a web interface
- Time-series price analysis

--------------------------------------------------
TECHNOLOGIES USED
--------------------------------------------------
Python, Pandas, NumPy, Matplotlib, Scikit-learn

--------------------------------------------------
LICENSE
--------------------------------------------------
MIT License
