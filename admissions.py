import streamlit as st
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from mapie.metrics import regression_coverage_score


# Set page configuration
st.set_page_config(page_title="Graduate Admission Predictor", layout="wide")

st.title("Graduate Admission Predictor")
st.image("images/admission.jpg", width=500)
# Load the trained model
@st.cache_resource
def load_model():
    with open('reg_admission.pickle', 'rb') as file:
        model = pickle.load(file)
    return model

# Load and prepare data for model insights
@st.cache_data
def load_data():
    df = pd.read_csv('data/Admission_Predict.csv')
    X = df.drop(columns=['Chance of Admit'])
    y = df['Chance of Admit']
    X_encoded = pd.get_dummies(X)
    train_X, test_X, train_y, test_y = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    return df, X_encoded, train_X, test_X, train_y, test_y

# Train base Random Forest model for visualizations
@st.cache_resource
def train_base_model(_train_X, _train_y):
    from sklearn.ensemble import RandomForestRegressor
    reg = RandomForestRegressor(random_state=42)
    reg.fit(_train_X, _train_y)
    return reg

# Load model and data
mapie_model = load_model()
df, X_encoded, train_X, test_X, train_y, test_y = load_data()
reg_model = train_base_model(train_X, train_y)

# Sidebar for inputs
st.sidebar.header("Enter Your Profile")
st.sidebar.markdown("Adjust the sliders below to input your academic credentials.")
st.sidebar.divider()

# Academic Test Scores Section
st.sidebar.subheader("Test Scores")
gre_score = st.sidebar.slider(
    "GRE Score",
    min_value=260,
    max_value=340,
    value=320,
    step=1,
    help="Enter your GRE score out of 340"
)

toefl_score = st.sidebar.slider(
    "TOEFL Score",
    min_value=0,
    max_value=120,
    value=110,
    step=1,
    help="Enter your TOEFL score out of 120"
)

st.sidebar.divider()

# Academic Background Section
st.sidebar.subheader("Academic Background")
university_rating = st.sidebar.select_slider(
    "University Rating",
    options=[1, 2, 3, 4, 5],
    value=3,
    help="Rate your undergraduate university (1=lowest, 5=highest)"
)

cgpa = st.sidebar.number_input(
    "Undergraduate CGPA",
    min_value=0.0,
    max_value=10.0,
    value=8.0,
    step=0.01,
    help="Enter your CGPA out of 10.0"
)

research = st.sidebar.radio(
    "Research Experience",
    options=["Yes", "No"],
    index=0,
    help="Do you have research experience?"
)

st.sidebar.divider()

# Application Materials Section
st.sidebar.subheader("Application Quality")
sop = st.sidebar.select_slider(
    "Statement of Purpose",
    options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    value=3.5,
    help="Rate the quality of your SOP (1=lowest, 5=highest)"
)

lor = st.sidebar.select_slider(
    "Letter of Recommendation",
    options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    value=3.5,
    help="Rate the quality of your LOR (1=lowest, 5=highest)"
)

st.sidebar.divider()

# Predict button in sidebar
predict_button = st.sidebar.button("Calculate Admission Chances", type="primary", use_container_width=True)

# Add helpful info in sidebar
with st.sidebar.expander("About This Tool"):
    st.markdown("""
    This tool uses a **Random Forest Regression model** trained on historical admission data to predict your chances of getting admitted to graduate programs.

    **Key Features:**
    - Prediction with 90% confidence intervals
    - Based on multiple academic factors
    - Uses MAPIE for uncertainty quantification

    **Note:** This is an estimation tool. Actual admission decisions depend on many factors not captured by this model.
    """)

with st.sidebar.expander("Typical Score Ranges"):
    st.markdown(f"""
    **From Training Data:**
    - GRE: {df['GRE Score'].min():.0f} - {df['GRE Score'].max():.0f} (Avg: {df['GRE Score'].mean():.0f})
    - TOEFL: {df['TOEFL Score'].min():.0f} - {df['TOEFL Score'].max():.0f} (Avg: {df['TOEFL Score'].mean():.0f})
    - CGPA: {df['CGPA'].min():.1f} - {df['CGPA'].max():.1f} (Avg: {df['CGPA'].mean():.1f})
    """)

# Main content area
st.markdown("### Predict your chances of admission with confidence intervals")
st.markdown("Use the sidebar to enter your profile details, then click the button to get your prediction.")

# Prediction logic
if predict_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'GRE Score': [gre_score],
        'TOEFL Score': [toefl_score],
        'University Rating': [university_rating],
        'SOP': [sop],
        'LOR': [lor],
        'CGPA': [cgpa],
        'Research': [research]
    })

    # One-hot encode the input
    input_encoded = pd.get_dummies(input_data)

    # Ensure all columns match training data
    for col in train_X.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = False

    input_encoded = input_encoded[train_X.columns]

    # Make prediction with 90% confidence interval (alpha = 0.1)
    alpha = 0.1
    y_pred, y_pis = mapie_model.predict(input_encoded, alpha=alpha)

    predicted_value = y_pred[0]
    lower_bound = y_pis[0, 0, 0]
    upper_bound = y_pis[0, 1, 0]

    # Display prediction with enhanced UI
    st.divider()
    st.subheader("Your Predicted Admission Chances")

    # Main prediction with larger display
    col_main1, col_main2, col_main3 = st.columns([1, 2, 1])

    with col_main2:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
            <h1 style='color: #1f77b4; margin: 0;'>{predicted_value*100:.1f}%</h1>
            <p style='color: #666; font-size: 18px; margin: 5px 0;'>Predicted Admission Chance</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")  # Spacing

    # Confidence interval display
    st.markdown("#### 90% Confidence Interval")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Lower Bound", f"{lower_bound*100:.2f}%", help="Lower limit of the 90% confidence interval")

    with col2:
        st.metric("Expected Range", f"{upper_bound*100 - lower_bound*100:.2f}%", help="Width of the confidence interval")

    with col3:
        st.metric("Upper Bound", f"{upper_bound*100:.2f}%", help="Upper limit of the 90% confidence interval")

    # Interpretation
    st.info(f"**Interpretation:** We are 90% confident that your actual admission chance falls between **{lower_bound*100:.1f}%** and **{upper_bound*100:.1f}%**. This range accounts for model uncertainty and variability in the admissions process.")

    # Visual representation of the interval
    import matplotlib.pyplot as plt
    fig_pred, ax_pred = plt.subplots(figsize=(10, 2))
    ax_pred.barh([0], [upper_bound - lower_bound], left=lower_bound, height=0.3, color='lightblue', alpha=0.7)
    ax_pred.plot([predicted_value], [0], 'ro', markersize=12, label='Predicted Value')
    ax_pred.set_xlim([0, 1])
    ax_pred.set_ylim([-0.5, 0.5])
    ax_pred.set_xlabel('Admission Chance', fontsize=12)
    ax_pred.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_pred.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax_pred.set_yticks([])
    ax_pred.legend(loc='upper right')
    ax_pred.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_pred)
else:
    # Show placeholder when no prediction has been made
    st.divider()
    st.info("Enter your academic profile in the sidebar and click **'Calculate Admission Chances'** to see your prediction.")

st.divider()

# Model Insights Section
st.header("Model Insights")

# Pre-compute predictions and metrics for all tabs (to avoid scope issues)
y_test_pred = reg_model.predict(test_X)
all_residuals = test_y - y_test_pred
r2 = sklearn.metrics.r2_score(test_y, y_test_pred)
rmse = sklearn.metrics.root_mean_squared_error(test_y, y_test_pred)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted vs Actual", "Coverage Plot"])

# Tab 1: Feature Importance
with tab1:
    st.subheader("Which features are the most important for predicting admission chances?")
    st.markdown("This chart shows how much each feature contributes to the model's predictions. Higher importance means the feature has a stronger influence on admission predictions.")

    # Calculate feature importance
    importance = reg_model.feature_importances_
    feature_imp = pd.DataFrame(list(zip(train_X.columns, importance)),
                               columns=['Feature', 'Importance'])
    feature_imp = feature_imp.sort_values('Importance', ascending=False).reset_index(drop=True)

    # Create plot
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    colors = ['red' if i < 2 else 'lime' for i in range(len(feature_imp))]
    ax1.barh(feature_imp['Feature'], feature_imp['Importance'], color=colors)
    ax1.set_xlabel("Importance", fontsize=12)
    ax1.set_ylabel("Input Feature", fontsize=12)
    ax1.tick_params(axis='both', labelsize=10)
    plt.tight_layout()

    st.pyplot(fig1)

# Tab 2: Histogram of Residuals
with tab2:
    st.subheader("Distribution of Residuals")
    st.markdown("Residuals represent the difference between predicted and actual values. A well-performing model should have residuals centered around zero with a normal distribution.")

    # Create plot
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.hist(all_residuals, bins=25, color='lime', edgecolor='black')
    ax2.set_xlabel('Residuals', fontsize=14)
    ax2.set_ylabel('# of Test Datapoints', fontsize=14)
    ax2.set_title('Distribution of Residuals', fontsize=16)
    ax2.tick_params(axis='both', labelsize=10)
    plt.tight_layout()

    st.pyplot(fig2)

    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R-squared on Test Set", f"{r2:.2f}")
    with col2:
        st.metric("RMSE on Test Set", f"{rmse:.2f}")

# Tab 3: Predicted vs Actual
with tab3:
    st.subheader("Predicted vs Actual Values")
    st.markdown("This scatter plot compares the model's predictions against actual admission chances. Points closer to the red diagonal line indicate better predictions.")

    # Create plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(test_y, y_test_pred, color='blue', alpha=0.6, edgecolor='black', s=40)
    ax3.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)],
             color='red', linestyle='--', lw=2, label='Perfect Predictions')
    ax3.set_xlabel('Actual Values', fontsize=12)
    ax3.set_ylabel('Predicted Values', fontsize=12)
    ax3.set_title('Predicted vs Actual Values', fontsize=14)
    ax3.tick_params(axis='both', labelsize=10)
    ax3.legend()
    plt.tight_layout()

    st.pyplot(fig3)

# Tab 4: Coverage Plot
with tab4:
    st.subheader("Prediction Intervals and Coverage")
    st.markdown("This plot shows how well the 90% confidence intervals capture the actual values. The green band represents the prediction intervals, and green dots are actual values. Good coverage means most actual values fall within the predicted intervals.")

    # Get predictions with intervals on test set
    alpha = 0.1
    y_test_pred_mapie, y_test_pis = mapie_model.predict(test_X, alpha=alpha)

    # Calculate coverage
    coverage = regression_coverage_score(test_y, y_test_pis[:, 0], y_test_pis[:, 1])
    coverage_percentage = coverage * 100

    # Create dataframe for visualization
    predictions = test_y.to_frame()
    predictions.columns = ['Actual Value']
    predictions["Predicted Value"] = y_test_pred_mapie.round(2)
    predictions["Lower Value"] = y_test_pis[:, 0].round(2)
    predictions["Upper Value"] = y_test_pis[:, 1].round(2)

    # Sort by actual value
    sorted_predictions = predictions.sort_values(by=['Actual Value']).reset_index(drop=True)

    # Create plot
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(sorted_predictions["Actual Value"], 'go', markersize=3, label="Actual Value")
    ax4.fill_between(np.arange(len(sorted_predictions)),
                     sorted_predictions["Lower Value"],
                     sorted_predictions["Upper Value"],
                     alpha=0.2, color="green", label="Prediction Interval")
    ax4.set_xlim([0, len(sorted_predictions)])
    ax4.set_xlabel("Samples", fontsize=12)
    ax4.set_ylabel("Target", fontsize=12)
    ax4.set_title(f"Prediction Intervals and Coverage: {coverage_percentage:.2f}%",
                  fontsize=14, fontweight="bold")
    ax4.legend(loc="upper left", fontsize=10)
    ax4.tick_params(axis='both', labelsize=10)
    plt.tight_layout()

    st.pyplot(fig4)

    st.metric("Model Coverage", f"{coverage_percentage:.2f}%")
    st.info(f"The model's prediction intervals contain {coverage_percentage:.2f}% of the actual values, " +
            "which aligns well with our 90% confidence level.")
