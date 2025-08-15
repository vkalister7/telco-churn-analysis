# Telco Customer Churn Analysis Dashboard

A Streamlit web application that predicts which telecom customers are likely to cancel their service and provides business insights to reduce churn.

## What It Does

This dashboard analyzes telecom customer data to:

- **Predict Customer Churn**: Uses machine learning models (Random Forest, XGBoost, Logistic Regression) to identify customers likely to cancel their service
- **Analyze Customer Behavior**: Explores patterns in age, revenue, contract types, service usage, and satisfaction scores
- **Calculate Business Impact**: Estimates revenue loss from churn and potential savings from retention campaigns
- **Generate Insights**: Identifies high-risk customer segments and key factors driving churn
- **Provide Recommendations**: Suggests prioritized actions to reduce churn with implementation timelines and ROI projections

## Key Features

- Interactive visualizations showing churn patterns across different customer segments
- Automated feature engineering creating derived metrics like revenue per month and service counts
- Model comparison with performance metrics (accuracy, precision, recall, F1-score)
- Financial impact analysis with customizable intervention costs and success rates
- Strategic recommendations ranked by priority and expected business impact

## How It Works

1. **Data Processing**: Loads customer data, handles missing values, and creates new features
2. **Model Training**: Trains multiple ML models with automated hyperparameter tuning
3. **Analysis**: Generates interactive charts and identifies key churn drivers
4. **Business Intelligence**: Calculates ROI for retention campaigns and provides actionable insights

The application is designed for business analysts and data scientists in telecommunications companies to make data-driven decisions about customer retention strategies.

## Customization

The code can be easily modified to fit different business needs:

### Modifying Models
- **New Algorithms**: Add different ML models to the `models_config` dictionary
- **Hyperparameter Tuning**: Adjust parameter grids for better performance on the data

### Dashboard Customization  
- **Branding**: Update colors, logos, and styling
- **Metrics**: Modify KPIs and success metrics

The modular code structure makes it straightforward to adapt the application for other industries or specific business requirements.