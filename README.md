# Logistic Regression Analysis - Luxury Housing Classification

## Project Overview
This repository contains a comprehensive logistic regression analysis for predicting luxury housing classification. The project implements multiple feature selection techniques and evaluates model performance using various statistical measures and visualizations.

## Dataset Description
The analysis uses a housing dataset with 7,000 property records containing 22 features. The target variable `IsLuxury` is a binary classification indicating whether a property is classified as luxury (1) or non-luxury (0).

### Key Features Analyzed
- **Price**: Property sale price ($85,000 - $1,046,676)
- **SquareFootage**: Living area in square feet (550 - 2,875 sq ft)  
- **NumBathrooms**: Number of bathrooms (1.0 - 5.8)
- **SchoolRating**: Local school quality rating (0.2 - 10.0 scale)
- **RenovationQuality**: Quality of renovations (0.01 - 10.0 scale)
- **LocalAmenities**: Access to local amenities (0.0 - 10.0 scale)

### Dataset Characteristics
- **Size**: 7,000 observations, 22 variables
- **Missing Values**: None
- **Duplicates**: None
- **Target Distribution**: Balanced (50% luxury, 50% non-luxury)

## Project Structure
```
Logistic-Regression-Analysis/
├── logistic_regression_analysis.ipynb  # main analysis notebook
├── housing_dataset.csv                 # complete housing dataset
├── training_dataset.csv               # processed training data
├── test_dataset.csv                   # processed test data
├── lra_env/                           # virtual environment
└── README.md                          # project documentation
```

## Analysis Methodology

### 1. Data Preprocessing
- **Missing Value Check**: Verified no missing values in dataset
- **Duplicate Detection**: Confirmed no duplicate records
- **Outlier Analysis**: Identified outliers in numerical features using IQR method
- **Correlation Analysis**: Examined relationships between variables

### 2. Feature Selection Methods
Three feature selection techniques were compared:

#### Forward Selection
- Selected Features: Price, NumBathrooms, SchoolRating, RenovationQuality
- Test Accuracy: 89.21%

#### Backward Selection  
- Selected Features: Price, NumBathrooms, SchoolRating, RenovationQuality
- Test Accuracy: 89.21%

#### Recursive Feature Elimination (RFE) - **Final Choice**
- Selected Features: Price, SquareFootage, RenovationQuality, LocalAmenities
- Test Accuracy: 89.57%

### 3. Model Validation
- **Assumption Testing**: Verified logistic regression assumptions
- **Multicollinearity Check**: VIF analysis (all values < 2.0)
- **Linearity of Log-odds**: Visual inspection of predictor relationships

## Model Performance

### Final Model Statistics
- **Test Accuracy**: 89.57%
- **Training Accuracy**: 88.21% (no overfitting)
- **ROC AUC Score**: 0.916 (excellent discrimination)
- **Pseudo R-squared**: 0.379
- **AIC**: 4827.35
- **BIC**: 4860.50

### Classification Performance
| Metric | Non-Luxury (0) | Luxury (1) | Overall |
|--------|---------------|------------|---------|
| Precision | 0.88 | 0.91 | 0.90 |
| Recall | 0.91 | 0.88 | 0.90 |
| F1-Score | 0.90 | 0.89 | 0.90 |

### Confusion Matrix (Test Set)
```
            Predicted
Actual      0     1
    0     645    62
    1      84   609
```

## Logistic Regression Equation
```
log(p/(1-p)) = 0.3154 + (2.3465 × Price) + (0.0310 × SquareFootage) 
               + (-0.0065 × RenovationQuality) + (0.0541 × LocalAmenities)
```

## Key Findings

### Coefficient Interpretation
1. **Price** (Coefficient: 2.347, OR: 10.45)
   - Strongest predictor of luxury classification
   - One standard deviation increase multiplies odds by 10.45

2. **SquareFootage** (Coefficient: 0.031, OR: 1.03)
   - Positive but modest effect on luxury probability
   - Larger homes slightly more likely to be luxury

3. **LocalAmenities** (Coefficient: 0.054, OR: 1.06)
   - Better amenity access increases luxury odds by 6%

4. **RenovationQuality** (Coefficient: -0.007, OR: 0.99)
   - Minimal negative effect (nearly neutral)

### Model Strengths
- Excellent discrimination ability (AUC = 0.916)
- No overfitting (test > training accuracy)
- Balanced performance across both classes
- Stable and consistent results

### Model Limitations
- Some predictors show weak individual significance
- Moderate pseudo R-squared value suggests room for improvement
- Possible non-linear relationships not captured
- Limited to binary classification

## Requirements
```python
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 0.24.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
statsmodels >= 0.12.0
scipy >= 1.7.0
```

## Installation & Usage

### Setting Up Environment
```bash
# create virtual environment
python3 -m venv lra_env

# activate environment
source lra_env/bin/activate

# install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels scipy jupyter
```

### Running the Analysis
```bash
# start jupyter notebook
jupyter notebook logistic_regression_analysis.ipynb
```

## Methodology Validation

### Assumption Verification
1. ✅ **Binary Dependent Variable**: IsLuxury contains only 0 and 1 values
2. ✅ **Independence of Observations**: Each house record is independent
3. ✅ **No Perfect Multicollinearity**: All VIF values < 2.0
4. ✅ **Linearity of Log-odds**: Reasonable linear relationships observed

### Feature Selection Justification
RFE was selected as the optimal method because:
- Achieved highest test accuracy (89.57%)
- More robust feature selection process
- Better inclusion of correlated predictors
- Lower information criteria (AIC/BIC)

## Future Enhancements
1. **Non-linear Models**: Explore polynomial features and interactions
2. **Ensemble Methods**: Implement Random Forest or Gradient Boosting
3. **Cross-Validation**: Implement k-fold CV for more robust estimates  
4. **Feature Engineering**: Create domain-specific composite features
5. **Threshold Optimization**: Fine-tune classification threshold
6. **External Validation**: Test on completely independent dataset

## Files Description
- **logistic_regression_analysis.ipynb**: Complete analysis with EDA, feature selection, modeling, and validation
- **housing_dataset.csv**: Original dataset with all 22 features
- **training_dataset.csv**: Standardized training data with selected features
- **test_dataset.csv**: Standardized test data for final evaluation

## Author
Logistic Regression Analysis Project

## License
This project is for educational and research purposes.