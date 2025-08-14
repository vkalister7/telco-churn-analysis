import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE

df = None
processed_df = None
models = {}
results = {}
feature_importance = {}
modeling_data = {}

@st.cache_data
def load_and_clean_data():
    global df
    
    try:
        df = pd.read_csv("telco.csv")
        
        # Fill NAs
        df["Offer"] = df["Offer"].fillna("None")
        df["Internet Type"] = df["Internet Type"].fillna("None")
        df["Churn Category"] = df["Churn Category"].fillna("None Provided")
        df["Churn Reason"] = df["Churn Reason"].fillna("None Provided")
        
        # Drop columns
        columns_to_drop = ["Under 30", "Senior Citizen", "Country", "State", "Customer ID", "Latitude", "Longitude", "Zip Code", "Population"]
        df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
        
        return df
    except FileNotFoundError:
        st.error("telco.csv file not found! Please make sure the file is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def feature_engineering(df):
    processed_df = df.copy()
    
    # Creating new features
    processed_df['Revenue_per_Month'] = processed_df['Total Revenue'] / (processed_df['Tenure in Months'] + 1)
    processed_df['Charges_to_Revenue_Ratio'] = processed_df['Monthly Charge'] / (processed_df['Total Revenue'] + 1)
    
    service_columns = ['Phone Service', 'Internet Service', 'Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music']
    
    processed_df['Services_Count'] = 0
    for col in service_columns:
        if col in processed_df.columns:
            processed_df['Services_Count'] += (processed_df[col] == 'Yes').astype(int)
    
    processed_df['Age_Group'] = pd.cut(processed_df['Age'], bins=[0, 30, 50, 65, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    processed_df['Tenure_Group'] = pd.cut(processed_df['Tenure in Months'], bins=[0, 12, 24, 48, 100], labels=['New', 'Short', 'Medium', 'Long'])
    
    revenue_75th = processed_df['Total Revenue'].quantile(0.75)
    processed_df['High_Value_Customer'] = (processed_df['Total Revenue'] >= revenue_75th).astype(int)
    
    # Encoding
    categorical_columns = processed_df.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    
    for column in categorical_columns:
        if column != 'Churn Label':
            le = LabelEncoder()
            processed_df[column] = le.fit_transform(processed_df[column])
            label_encoders[column] = le
    
    return processed_df

@st.cache_data
def prepare_modeling_data(processed_df):
    exclude_columns = ['Churn Label', 'Customer Status', 'Churn Category', 'Churn Reason', 'City']
    feature_columns = [col for col in processed_df.columns if col not in exclude_columns]
    
    X = processed_df[feature_columns]
    y = processed_df['Churn Label']
    
    # Encoding
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    return {
        'X_train': X_train_balanced,
        'X_test': X_test_scaled,
        'y_train': y_train_balanced,
        'y_test': y_test,
        'feature_names': feature_columns,
        'scaler': scaler,
        'label_encoder': le_target
    }

@st.cache_data
def train_models(_modeling_data):
    models = {}
    
    X_train = _modeling_data['X_train']
    y_train = _modeling_data['y_train']

    models_config = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2],
                'n_estimators': [100, 200]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        }
    }
    
    # Train models
    for name, config in models_config.items():
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        models[name] = grid_search.best_estimator_
    
    return models

@st.cache_data
def evaluate_models(_models, _modeling_data):
    X_test = _modeling_data['X_test']
    y_test = _modeling_data['y_test']
    
    results = []
    feature_importance = {}
    
    for name, model in _models.items():
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc
        })
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance[name] = dict(zip(_modeling_data['feature_names'], model.feature_importances_))
    
    results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
    return results_df, feature_importance

def main():
    st.set_page_config(
        page_title="Telco Customer Churn Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Telco Customer Churn Analysis Dashboard")
    st.markdown("---")
    
    with st.spinner("Loading and processing data"):
        df = load_and_clean_data()
    
    if df is None:
        st.stop()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis",
        ["Overview", "Data Exploration", "Model Performance", "Business Impact", "Customer Insights", "Recommendations"]
    )
    
    # Process data for analysis
    with st.spinner("Preparing analysis"):
        processed_df = feature_engineering(df)
        modeling_data = prepare_modeling_data(processed_df)
        models = train_models(modeling_data)
        results, feature_importance = evaluate_models(models, modeling_data)
    
    # Page content based on selection
    if page == "Overview":
        show_overview(df)
    elif page == "Data Exploration":
        show_data_exploration(df)
    elif page == "Model Performance":
        show_model_performance(results, feature_importance, modeling_data)
    elif page == "Business Impact":
        show_business_impact(df, results)
    elif page == "Customer Insights":
        show_customer_insights(df)
    elif page == "Recommendations":
        show_recommendations(df, results, feature_importance)

def show_overview(df):
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    churned_customers = len(df[df['Churn Label'] == 'Yes'])
    churn_rate = churned_customers / total_customers
    avg_revenue = df['Total Revenue'].mean()
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("Churned Customers", f"{churned_customers:,}")
    with col3:
        st.metric("Churn Rate", f"{churn_rate:.2%}")
    with col4:
        st.metric("Avg Revenue/Customer", f"${avg_revenue:,.2f}")
    
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        st.write(f"**Numeric Columns:** {len(df.select_dtypes(include=[np.number]).columns)}")
        st.write(f"**Categorical Columns:** {len(df.select_dtypes(include=['object']).columns)}")
    
    with col2:
        st.write("**Sample Data:**")
        st.dataframe(df.head())
    
    st.subheader("Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**High-Risk Segments:**")
        
        if 'Contract' in df.columns:
            contract_churn = df.groupby('Contract')['Churn Label'].apply(lambda x: (x == 'Yes').mean()).sort_values(ascending=False)
            for contract, rate in contract_churn.head(3).items():
                st.write(f"• {contract}: {rate:.2%} churn rate")
        
        short_tenure_churn = df[df['Tenure in Months'] <= 12]['Churn Label'].apply(lambda x: x == 'Yes').mean()
        st.write(f"• New customers (≤12 months): {short_tenure_churn:.2%} churn rate")
    
    with col2:
        st.write("**Revenue Impact:**")
        revenue_lost = churned_customers * df[df['Churn Label'] == 'Yes']['Total Revenue'].mean()
        avg_churned_revenue = df[df['Churn Label'] == 'Yes']['Total Revenue'].mean()
        
        st.write(f"• Total revenue lost: ${revenue_lost:,.2f}")
        st.write(f"• Avg revenue per churned customer: ${avg_churned_revenue:,.2f}")
        
        avg_retained_revenue = df[df['Churn Label'] == 'No']['Total Revenue'].mean()
        revenue_diff = avg_retained_revenue - avg_churned_revenue
        st.write(f"• Revenue difference: ${revenue_diff:,.2f} higher for retained customers")

def show_data_exploration(df):
    st.header("Data Exploration")
    
    viz_option = st.selectbox(
        "Select Analysis",
        ["Churn Distribution", "Age Analysis", "Revenue Analysis", "Contract Analysis", 
         "Service Analysis", "Satisfaction Analysis"]
    )
    
    if viz_option == "Churn Distribution":
        col1, col2 = st.columns(2)
        
        with col1:
            churn_counts = df['Churn Label'].value_counts()
            fig = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
                title="Overall Churn Distribution",
                color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            gender_churn = df.groupby(['Gender', 'Churn Label']).size().reset_index(name='Count')
            fig = px.bar(
                gender_churn, x='Gender', y='Count', color='Churn Label',
                title='Churn Distribution by Gender',
                color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Age Analysis":
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, x="Age", color="Churn Label",
                title="Age Distribution by Churn Status",
                barmode="overlay",
                color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            age_groups = pd.cut(df['Age'], bins=5)
            age_churn = df.groupby(age_groups)['Churn Label'].apply(lambda x: (x == 'Yes').mean()).reset_index()
            age_churn['Age_Range'] = age_churn['Age'].astype(str)
            
            fig = px.bar(
                age_churn, x='Age_Range', y='Churn Label',
                title='Churn Rate by Age Group'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Revenue Analysis":
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                df, x="Churn Label", y="Total Revenue",
                title="Total Revenue Distribution by Churn Status",
                color="Churn Label",
                color_discrete_map={'Yes':'#ff6b6b', 'No': '#4ecdc4'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df, x="Churn Label", y="Monthly Charge",
                title="Monthly Charges by Churn Status",
                color="Churn Label",
                color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Contract Analysis":
        contract_data = df.groupby(['Contract', 'Churn Label']).size().reset_index(name='Count')
        fig = px.bar(
            contract_data, x="Contract", y="Count", color="Churn Label",
            title="Contract Type vs Churn",
            color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        contract_churn_rate = df.groupby('Contract')['Churn Label'].apply(lambda x: (x == 'Yes').mean()).reset_index()
        contract_churn_rate.columns = ['Contract', 'Churn_Rate']
        
        fig = px.bar(
            contract_churn_rate, x='Contract', y='Churn_Rate',
            title='Churn Rate by Contract Type',
            color='Churn_Rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Service Analysis":
        services = ['Phone Service', 'Internet Service', 'Online Security', 'Online Backup',
                   'Device Protection Plan', 'Premium Tech Support']
        
        service_data = []
        for service in services:
            if service in df.columns:
                churn_rate_yes = df[df[service] == 'Yes']['Churn Label'].apply(lambda x: x == 'Yes').mean()
                churn_rate_no = df[df[service] == 'No']['Churn Label'].apply(lambda x: x == 'Yes').mean()
                service_data.append({'Service': service, 'Has_Service': 'Yes', 'Churn_Rate': churn_rate_yes})
                service_data.append({'Service': service, 'Has_Service': 'No', 'Churn_Rate': churn_rate_no})
        
        if service_data:
            service_df = pd.DataFrame(service_data)
            fig = px.bar(
                service_df, x='Service', y='Churn_Rate', color='Has_Service',
                title='Churn Rate by Service Usage',
                barmode='group'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Satisfaction Analysis":
        if 'Satisfaction Score' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(
                    df, x="Churn Label", y="Satisfaction Score",
                    title="Satisfaction Score by Churn Status",
                    color="Churn Label",
                    color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                satisfaction_churn = df.groupby('Satisfaction Score')['Churn Label'].apply(
                    lambda x: (x == 'Yes').mean()
                ).reset_index()
                satisfaction_churn.columns = ['Satisfaction_Score', 'Churn_Rate']
                
                fig = px.bar(
                    satisfaction_churn, x='Satisfaction_Score', y='Churn_Rate',
                    title='Churn Rate by Satisfaction Score'
                )
                st.plotly_chart(fig, use_container_width=True)

def show_model_performance(results, feature_importance, _modeling_data):
    st.header("Model Performance")
    
    st.subheader("Model Comparison")
    st.dataframe(results.round(4), use_container_width=True)
    
    best_model = results.iloc[0]
    st.success(f"Best Model: **{best_model['Model']}** with F1-Score: **{best_model['F1-Score']:.4f}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        
        fig = go.Figure()
        
        for _, row in results.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[metric] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        melted_results = results.melt(
            id_vars=['Model'], 
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            var_name='Metric', 
            value_name='Score'
        )
        
        fig = px.bar(
            melted_results, x='Model', y='Score', color='Metric',
            title='Detailed Model Metrics',
            barmode='group'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    if feature_importance and best_model['Model'] in feature_importance:
        st.subheader("Feature Importance Analysis")
        
        importance_dict = feature_importance[best_model['Model']]
        importance_df = pd.DataFrame(
            list(importance_dict.items()), 
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(
            importance_df, x='Importance', y='Feature',
            orientation='h', 
            title=f'Top 15 Most Important Features - {best_model["Model"]}'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Importance Table")
        st.dataframe(importance_df, use_container_width=True)

def show_business_impact(df, results):
    st.header("Business Impact Analysis")
    
    total_customers = len(df)
    churned_customers = len(df[df['Churn Label'] == 'Yes'])
    churn_rate = churned_customers / total_customers
    avg_revenue_churned = df[df['Churn Label'] == 'Yes']['Total Revenue'].mean()
    revenue_lost = churned_customers * avg_revenue_churned
    
    st.subheader("Current Business Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Customers Lost", f"{churned_customers:,}")
    with col2:
        st.metric("Churn Rate", f"{churn_rate:.2%}")
    with col3:
        st.metric("Revenue Lost", f"${revenue_lost:,.2f}")
    with col4:
        st.metric("Avg Revenue/Churned", f"${avg_revenue_churned:,.2f}")
    
    if results is not None and not results.empty:
        st.subheader("Projected Model Impact")
        
        best_model = results.iloc[0]
        precision = best_model['Precision']
        recall = best_model['Recall']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            intervention_cost = st.slider(
                "Intervention Cost per Customer ($)", 
                min_value=10, max_value=200, value=50, step=10
            )
        
        with col2:
            success_rate = st.slider(
                "Intervention Success Rate", 
                min_value=0.1, max_value=0.8, value=0.3, step=0.05
            )
        
        with col3:
            model_threshold = st.slider(
                "Model Prediction Threshold", 
                min_value=0.3, max_value=0.9, value=0.5, step=0.05
            )
        
        predicted_churners = int(churned_customers / recall)
        true_churners_identified = int(predicted_churners * precision)
        customers_retained = int(true_churners_identified * success_rate)
        revenue_saved = customers_retained * avg_revenue_churned
        total_intervention_cost = predicted_churners * intervention_cost
        net_benefit = revenue_saved - total_intervention_cost
        roi = (net_benefit / total_intervention_cost * 100) if total_intervention_cost > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Customers Flagged", f"{predicted_churners:,}")
        with col2:
            st.metric("True Churners Found", f"{true_churners_identified:,}")
        with col3:
            st.metric("Customers Retained", f"{customers_retained:,}")
        with col4:
            st.metric("ROI", f"{roi:.1f}%")
        
        st.subheader("Financial Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Revenue Saved", f"${revenue_saved:,.2f}", f"{revenue_saved/revenue_lost:.1%} of total lost")
        
        with col2:
            st.metric("Net Benefit", f"${net_benefit:,.2f}", f"Cost: ${total_intervention_cost:,.2f}")
        
        impact_data = pd.DataFrame({
            'Scenario': ['Current State', 'With Model'],
            'Revenue Lost': [revenue_lost, revenue_lost - revenue_saved],
            'Customers Lost': [churned_customers, churned_customers - customers_retained]
        })
        
        fig = px.bar(
            impact_data, x='Scenario', y='Revenue Lost',
            title='Revenue Impact Comparison',
            color='Scenario'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_customer_insights(df):
    st.header("Customer Insights")
    st.subheader("Customer Segmentation Analysis")
    
    high_value_threshold = df['Total Revenue'].quantile(0.75)
    high_value_customers = df[df['Total Revenue'] >= high_value_threshold]
    high_value_churn_rate = (high_value_customers['Churn Label'] == 'Yes').mean()
    
    low_value_customers = df[df['Total Revenue'] < high_value_threshold]
    low_value_churn_rate = (low_value_customers['Churn Label'] == 'Yes').mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("High-Value Customer Churn", f"{high_value_churn_rate:.2%}")
        st.metric("High-Value Customers", f"{len(high_value_customers):,}")
    
    with col2:
        st.metric("Low-Value Customer Churn", f"{low_value_churn_rate:.2%}")
        st.metric("Low-Value Customers", f"{len(low_value_customers):,}")
    
    st.subheader("Customer Tenure Analysis")
    
    tenure_bins = [0, 6, 12, 24, 48, float('inf')]
    tenure_labels = ['0-6 months', '6-12 months', '1-2 years', '2-4 years', '4+ years']
    df['Tenure_Segment'] = pd.cut(df['Tenure in Months'], bins=tenure_bins, labels=tenure_labels)
    
    tenure_analysis = df.groupby('Tenure_Segment').agg({
        'Churn Label': lambda x: (x == 'Yes').mean(),
        'Total Revenue': 'mean',
        df.columns[0]: 'count'
    }).round(4)
    
    tenure_analysis.columns = ['Churn Rate', 'Avg Revenue', 'Customer Count']
    st.dataframe(tenure_analysis, use_container_width=True)
    
    tenure_churn = df.groupby('Tenure_Segment')['Churn Label'].apply(lambda x: (x == 'Yes').mean()).reset_index()
    tenure_churn.columns = ['Tenure_Segment', 'Churn_Rate']
    
    fig = px.bar(
        tenure_churn, x='Tenure_Segment', y='Churn_Rate',
        title='Churn Rate by Customer Tenure',
        color='Churn_Rate',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if 'Payment Method' in df.columns:
        st.subheader("Payment Method Analysis")
        
        payment_churn = df.groupby('Payment Method')['Churn Label'].apply(
            lambda x: (x == 'Yes').mean()
        ).sort_values(ascending=False).reset_index()
        payment_churn.columns = ['Payment_Method', 'Churn_Rate']
        
        fig = px.bar(
            payment_churn, x='Payment_Method', y='Churn_Rate',
            title='Churn Rate by Payment Method',
            color='Churn_Rate',
            color_continuous_scale='Reds'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Service Usage Patterns")
    
    streaming_services = ['Streaming TV', 'Streaming Movies', 'Streaming Music']
    available_streaming = [col for col in streaming_services if col in df.columns]
    
    if available_streaming:
        streaming_analysis = []
        
        for service in available_streaming:
            churn_with = df[df[service] == 'Yes']['Churn Label'].apply(lambda x: x == 'Yes').mean()
            churn_without = df[df[service] == 'No']['Churn Label'].apply(lambda x: x == 'Yes').mean()
            count_with = len(df[df[service] == 'Yes'])
            
            streaming_analysis.append({
                'Service': service,
                'Churn Rate (With)': churn_with,
                'Churn Rate (Without)': churn_without,
                'Customers With Service': count_with,
                'Difference': churn_with - churn_without
            })
        
        streaming_df = pd.DataFrame(streaming_analysis)
        st.dataframe(streaming_df.round(4), use_container_width=True)

def show_recommendations(df, results, feature_importance):
    st.header("Strategic Recommendations")
    st.subheader("High Priority Actions")
    
    recommendations = []
    
    if 'Contract' in df.columns:
        contract_churn = df.groupby('Contract')['Churn Label'].apply(lambda x: (x == 'Yes').mean())
        worst_contract = contract_churn.idxmax()
        recommendations.append({
            'Priority': 'HIGH',
            'Category': 'Contract Management',
            'Recommendation': f'Address {worst_contract} contract churn rate of {contract_churn[worst_contract]:.2%}',
            'Impact': 'Revenue Protection',
            'Timeline': 'Immediate'
        })
    
    short_tenure_churn = df[df['Tenure in Months'] <= 12]['Churn Label'].apply(lambda x: x == 'Yes').mean()
    recommendations.append({
        'Priority': 'HIGH',
        'Category': 'Customer Onboarding',
        'Recommendation': f'Improve first-year customer experience - {short_tenure_churn:.2%} churn rate',
        'Impact': 'Customer Retention',
        'Timeline': '3-6 months'
    })
    
    if 'Satisfaction Score' in df.columns:
        low_satisfaction_churn = df[df['Satisfaction Score'] <= 2]['Churn Label'].apply(lambda x: x == 'Yes').mean()
        recommendations.append({
            'Priority': 'HIGH',
            'Category': 'Customer Experience',
            'Recommendation': f'Address low satisfaction customers - {low_satisfaction_churn:.2%} churn rate',
            'Impact': 'Customer Experience',
            'Timeline': '1-3 months'
        })
    
    if 'Payment Method' in df.columns:
        payment_churn = df.groupby('Payment Method')['Churn Label'].apply(lambda x: (x == 'Yes').mean())
        worst_payment = payment_churn.idxmax()
        recommendations.append({
            'Priority': 'MEDIUM',
            'Category': 'Payment Strategy',
            'Recommendation': f'Incentivize better payment methods - {worst_payment} has {payment_churn[worst_payment]:.2%} churn rate',
            'Impact': 'Process Improvement',
            'Timeline': '2-4 months'
        })
    
    if results is not None and not results.empty:
        best_model = results.iloc[0]['Model']
        recommendations.append({
            'Priority': 'HIGH',
            'Category': 'Technology Implementation',
            'Recommendation': f'Deploy {best_model} model for real-time churn prediction',
            'Impact': 'Proactive Retention',
            'Timeline': '1-2 months'
        })
    
    if feature_importance and results is not None and not results.empty:
        best_model_name = results.iloc[0]['Model']
        if best_model_name in feature_importance:
            top_features = sorted(feature_importance[best_model_name].items(), key=lambda x: x[1], reverse=True)[:3]
            feature_names = ', '.join([f[0] for f in top_features])
            recommendations.append({
                'Priority': 'MEDIUM',
                'Category': 'Data-Driven Strategy',
                'Recommendation': f'Focus monitoring on key factors: {feature_names}',
                'Impact': 'Targeted Interventions',
                'Timeline': '2-3 months'
            })
    
    recommendations_df = pd.DataFrame(recommendations)
    
    def highlight_priority(row):
        if row['Priority'] == 'HIGH':
            return ['background-color: #ffcdd2; color: #000000'] * len(row)
        elif row['Priority'] == 'MEDIUM':
            return ['background-color: #ffe0b2; color: #000000'] * len(row)
        else:
            return ['background-color: #c8e6c9; color: #000000'] * len(row)
    
    st.dataframe(
        recommendations_df.style.apply(highlight_priority, axis=1),
        use_container_width=True
    )
    
    st.subheader("Implementation Roadmap")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Phase 1 (Months 1-2): Immediate Actions**")
        st.write("• Deploy predictive model")
        st.write("• Set up automated alerts for high-risk customers")
        st.write("• Launch customer satisfaction improvement program")
        st.write("• Implement retention campaigns for contract renewal")
    
    with col2:
        st.write("**Phase 2 (Months 3-6): Strategic Initiatives**")
        st.write("• Redesign customer onboarding process")
        st.write("• A/B test intervention strategies")
        st.write("• Implement payment method incentives")
        st.write("• Develop segment-specific retention offers")
    
    st.subheader("Success Metrics & KPIs")
    
    metrics_data = {
        'Metric': [
            'Overall Churn Rate',
            'Model Precision',
            'Customer Satisfaction Score',
            'Revenue Retention',
            'Intervention Success Rate'
        ],
        'Current': [
            f"{df['Churn Label'].apply(lambda x: x == 'Yes').mean():.2%}",
            f"{results.iloc[0]['Precision']:.3f}" if results is not None and not results.empty else "N/A",
            f"{df['Satisfaction Score'].mean():.1f}" if 'Satisfaction Score' in df.columns else "N/A",
            "Baseline",
            "30% (Assumed)"
        ],
        'Target (6 months)': [
            f"{df['Churn Label'].apply(lambda x: x == 'Yes').mean() * 0.8:.2%}",
            ">0.800",
            f"{df['Satisfaction Score'].mean() + 0.5:.1f}" if 'Satisfaction Score' in df.columns else "4.0",
            "+15%",
            "45%"
        ],
        'Target (12 months)': [
            f"{df['Churn Label'].apply(lambda x: x == 'Yes').mean() * 0.7:.2%}",
            ">0.850",
            f"{df['Satisfaction Score'].mean() + 1.0:.1f}" if 'Satisfaction Score' in df.columns else "4.5",
            "+25%",
            "60%"
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    st.subheader("Additional Strategic Considerations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Technology & Analytics:**")
        st.write("• Real-time customer health scoring")
        st.write("• Automated intervention workflows")
        st.write("• Regular model retraining (monthly)")
        st.write("• Customer feedback loop integration")
        st.write("• Predictive analytics dashboard")
    
    with col2:
        st.write("**Business Process:**")
        st.write("• Cross-functional retention team")
        st.write("• Customer success manager assignment")
        st.write("• Proactive customer outreach program")
        st.write("• Service quality improvement initiatives")
        st.write("• Competitive pricing analysis")

if __name__ == "__main__":
    main()