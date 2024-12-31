import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="Health Assistant Pro",
    layout="wide",
    page_icon="🧑‍⚕️",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    /* Input field styles */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        background-color: white !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 16px !important;
        color: #333 !important;
    }

    /* Additional number input specific styles */
    div[data-baseweb="input"] {
        background-color: white !important;
    }
    
    div[data-baseweb="input"] input {
        background-color: white !important;
        border: none !important;
    }

    div[data-baseweb="base-input"] {
        background-color: white !important;
    }
    
    div[data-baseweb="base-input"] input {
        background-color: white !important;
        border: none !important;
    }

    /* Number input container */
    .stNumberInput > div {
        background-color: white !important;
    }

    /* Number input wrapper */
    .stNumberInput > div > div {
        background-color: white !important;
    }

    /* Actual number input field */
    .stNumberInput input[type="number"] {
        background-color: white !important;
        border: none !important;
    }

    /* Input field focus state */
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus,
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="base-input"]:focus-within {
        border-color: #2196F3 !important;
        box-shadow: 0 0 0 1px #2196F3 !important;
    }

    /* Input field hover state */
    .stTextInput > div > div > input:hover,
    .stNumberInput > div > div > input:hover,
    .stSelectbox > div > div:hover,
    div[data-baseweb="input"]:hover,
    div[data-baseweb="base-input"]:hover {
        border-color: #2196F3 !important;
    }

    /* Label styles */
    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label {
        color: #1565C0 !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        padding-bottom: 5px !important;
    }

    /* Selectbox specific styles */
    .stSelectbox > div > div > div {
        background-color: white !important;
    }

    /* Button styles */
    .stButton > button {
        background-color: #2196F3 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        border: none !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background-color: #1976D2 !important;
    }

    /* Container styles */
    .block-container {
        padding: 20px !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }

    /* Header styles */
    h1 {
        color: #1565C0 !important;
        font-size: 36px !important;
        font-weight: bold !important;
        margin-bottom: 24px !important;
        text-align: center !important;
    }

    h2 {
        color: #1976D2 !important;
        font-size: 28px !important;
        font-weight: bold !important;
        margin-bottom: 16px !important;
    }

    h3 {
        color: #2196F3 !important;
        font-size: 24px !important;
        font-weight: bold !important;
    }

    h4 {
        color: #1565C0 !important;
        font-size: 20px !important;
        font-weight: bold !important;
        margin-bottom: 16px !important;
    }

    /* Alert styles */
    .stAlert {
        background-color: white !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
        padding: 16px !important;
        margin: 16px 0 !important;
    }

    /* Info box styles */
    .stInfo {
        background-color: #E3F2FD !important;
        color: #1565C0 !important;
        border: none !important;
    }

    /* Success message styles */
    .stSuccess {
        background-color: #E8F5E9 !important;
        color: #388E3C !important;
        border: none !important;
    }

    /* Warning message styles */
    .stWarning {
        background-color: #FFF3E0 !important;
        color: #F57C00 !important;
        border: none !important;
    }

    /* Error message styles */
    .stError {
        background-color: #FFEBEE !important;
        color: #D32F2F !important;
        border: none !important;
    }

    /* Markdown text styles */
    .element-container div[data-testid="stMarkdownContainer"] {
        color: #333 !important;
    }

    /* Sidebar styles */
    .css-1d391kg {
        background-color: white !important;
    }

    /* Expander styles */
    .streamlit-expanderHeader {
        background-color: white !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)



# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Function to get prediction result with confidence
def get_prediction_result(model, input_data, threshold=0.5):
    try:
        # Try to get probability predictions
        prediction_proba = model.predict_proba([input_data])
        prediction = prediction_proba[0][1] >= threshold
        confidence = prediction_proba[0][1] if prediction else prediction_proba[0][0]
        return prediction, confidence * 100
    except (AttributeError, NotImplementedError):
        # If probabilities are not available, return binary prediction with None confidence
        prediction = model.predict([input_data])[0]
        return prediction, None

# sidebar for navigation
with st.sidebar:


    selected = option_menu(
        'Navigation Menu',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        menu_icon='hospital',
        icons=['activity', 'heart-pulse', 'person-circle'],
        styles={
            "container": {
                "padding": "20px",
                "background-color": "white",
                "border-radius": "8px",
                "margin": "8px 0"
            },
            "icon": {
                "color": "#1976D2",
                "font-size": "20px"
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "8px 0",
                "padding": "12px 16px",
                "border-radius": "8px"
            },
            "nav-link-selected": {
                "background-color": "#2196F3",
                "color": "white"
            }
        }
    )
    st.markdown("<div style='height: 2px; background-color: #2196F3; margin: 16px 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 8px;'>
            <h3 style='color: #1565C0; font-size: 20px; margin-bottom: 16px;'>About</h3>
            <p style='color: #333; font-size: 16px; line-height: 1.5;'>
                This application leverages advanced machine learning models to predict various diseases based on clinical parameters. Our system provides accurate risk assessments for diabetes, heart disease, and Parkinson's disease.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.markdown("""
        <div style='background-color: #E3F2FD; padding: 24px; border-radius: 8px; margin-bottom: 24px;'>
            <h2 style='color: #1565C0; text-align: center; margin-bottom: 16px;'>Diabetes Risk Assessment</h2>
            <p style='text-align: center; color: #1976D2; font-size: 18px;'>Enter your clinical parameters below for an accurate diabetes risk prediction</p>
        </div>
    """, unsafe_allow_html=True)

    # Information box about diabetes
    st.info("""
    ℹ️ **About Diabetes Risk Assessment**
    
    This tool uses machine learning to assess diabetes risk based on clinical parameters. The model considers various factors 
    including glucose levels, blood pressure, and other health indicators to provide a risk assessment.
    
    Please enter accurate values for the most reliable prediction.
    """)

    # Create three columns with equal width
    col1, col2, col3 = st.columns(3)

    # Input validation function
    def validate_input(value, min_val, max_val, name):
        if value < min_val or value > max_val:
            st.warning(f"⚠️ {name} should be between {min_val} and {max_val}")
            return False
        return True

    with col1:
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 8px;'>
            <h4 style='color: #1565C0;'>Patient Information</h4>
        """, unsafe_allow_html=True)
        
        Pregnancies = st.number_input(
            'Number of Pregnancies',
            min_value=0,
            max_value=20,
            value=0,
            help="Number of times pregnant"
        )
        
        Glucose = st.number_input(
            'Glucose Level (mg/dL)',
            min_value=0,
            max_value=500,
            value=100,
            help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test"
        )
        
        BloodPressure = st.number_input(
            'Blood Pressure (mm Hg)',
            min_value=0,
            max_value=300,
            value=70,
            help="Diastolic blood pressure (mm Hg)"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 8px;'>
            <h4 style='color: #1565C0;'>Physical Measurements</h4>
        """, unsafe_allow_html=True)
        
        SkinThickness = st.number_input(
            'Skin Thickness (mm)',
            min_value=0,
            max_value=100,
            value=20,
            help="Triceps skin fold thickness (mm)"
        )
        
        Insulin = st.number_input(
            'Insulin Level (µU/mL)',
            min_value=0,
            max_value=1000,
            value=79,
            help="2-Hour serum insulin (µU/mL)"
        )
        
        BMI = st.number_input(
            'BMI Value',
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            help="Body Mass Index (weight in kg/(height in m)²)"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 8px;'>
            <h4 style='color: #1565C0;'>Additional Factors</h4>
        """, unsafe_allow_html=True)
        
        DiabetesPedigreeFunction = st.number_input(
            'Diabetes Pedigree Function',
            min_value=0.0,
            max_value=4.0,
            value=0.47,
            help="A function that scores likelihood of diabetes based on family history"
        )
        
        Age = st.number_input(
            'Age (years)',
            min_value=0,
            max_value=120,
            value=30,
            help="Age in years"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Add a separator
    st.markdown("<hr style='margin: 24px 0;'>", unsafe_allow_html=True)

    # Create columns for the prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button('Predict Diabetes Risk', use_container_width=True)

    # code for Prediction
    if predict_button:
        # Input validation
        valid_inputs = True
        valid_inputs &= validate_input(Glucose, 0, 500, "Glucose Level")
        valid_inputs &= validate_input(BloodPressure, 0, 300, "Blood Pressure")
        valid_inputs &= validate_input(BMI, 0, 70, "BMI")

        if valid_inputs:
            try:
                # Prepare input data
                user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, 
                            Insulin, BMI, DiabetesPedigreeFunction, Age]
                user_input = [float(x) for x in user_input]
                
                # Get prediction and confidence
                prediction, confidence = get_prediction_result(diabetes_model, user_input)
                
                # Display result in a nice format
                st.markdown("<div style='background-color: white; padding: 24px; border-radius: 8px; margin-top: 24px;'>", unsafe_allow_html=True)
                
                if prediction:
                    st.markdown("""
                        <h3 style='color: #d32f2f; text-align: center; margin-bottom: 16px;'>
                            ⚠️ Higher Risk of Diabetes Detected
                        </h3>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <h3 style='color: #388e3c; text-align: center; margin-bottom: 16px;'>
                            ✅ Lower Risk of Diabetes Detected
                        </h3>
                    """, unsafe_allow_html=True)

                # Display confidence if available
                if confidence is not None:
                    st.markdown(f"""
                        <p style='text-align: center; font-size: 18px; color: #666;'>
                            Confidence Level: {confidence:.1f}%
                        </p>
                    """, unsafe_allow_html=True)

                # Add recommendations based on risk factors
                st.markdown("<h4 style='color: #1565C0; margin-top: 24px;'>Key Observations:</h4>", unsafe_allow_html=True)
                
                recommendations = []
                if Glucose > 140:
                    recommendations.append("• High glucose levels detected. Consider consulting a healthcare provider.")
                if BMI > 30:
                    recommendations.append("• BMI indicates obesity. Consider lifestyle modifications.")
                if BloodPressure > 90:
                    recommendations.append("• Elevated blood pressure detected. Regular monitoring recommended.")
                
                if recommendations:
                    for rec in recommendations:
                        st.markdown(f"<p style='color: #666;'>{rec}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #666;'>• All parameters are within normal ranges.</p>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # Disclaimer
                st.markdown("""
                    <p style='color: #666; font-size: 14px; text-align: center; margin-top: 16px;'>
                        ⚠️ This prediction is based on machine learning analysis and should not replace professional medical advice.
                        Please consult with a healthcare provider for proper diagnosis and treatment.
                    </p>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error("An error occurred during prediction. Please check your input values and try again.")
                st.exception(e)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.markdown("""
        <div style='background-color: #E3F2FD; padding: 24px; border-radius: 8px; margin-bottom: 24px;'>
            <h2 style='color: #1565C0; text-align: center; margin-bottom: 16px;'>Heart Disease Risk Assessment</h2>
            <p style='text-align: center; color: #1976D2; font-size: 18px;'>Enter your clinical parameters for heart disease risk evaluation</p>
        </div>
    """, unsafe_allow_html=True)

    # Information box about heart disease
    st.info("""
    ℹ️ **About Heart Disease Risk Assessment**
    
    This tool evaluates your risk of heart disease using machine learning analysis of various cardiovascular parameters. 
    The assessment considers multiple factors including blood pressure, cholesterol levels, and other clinical indicators.
    
    Please provide accurate information for the most reliable assessment.
    """)

    # Input validation function
    def validate_heart_input(value, min_val, max_val, name):
        if value < min_val or value > max_val:
            st.warning(f"⚠️ {name} should be between {min_val} and {max_val}")
            return False
        return True

    # Create three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 8px;'>
            <h4 style='color: #1565C0;'>Personal Information</h4>
        """, unsafe_allow_html=True)
        
        age = st.number_input(
            'Age',
            min_value=1,
            max_value=120,
            value=45,
            help="Age in years"
        )
        
        sex = st.selectbox(
            'Sex',
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
            help="Biological sex"
        )
        
        cp = st.selectbox(
            'Chest Pain Type',
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-anginal Pain",
                3: "Asymptomatic"
            }[x],
            help="Type of chest pain experienced"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 8px;'>
            <h4 style='color: #1565C0;'>Clinical Measurements</h4>
        """, unsafe_allow_html=True)
        
        trestbps = st.number_input(
            'Resting Blood Pressure (mm Hg)',
            min_value=50,
            max_value=250,
            value=120,
            help="Resting blood pressure in mm Hg"
        )
        
        chol = st.number_input(
            'Cholesterol (mg/dl)',
            min_value=100,
            max_value=600,
            value=200,
            help="Serum cholesterol in mg/dl"
        )
        
        fbs = st.selectbox(
            'Fasting Blood Sugar > 120 mg/dl',
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Fasting blood sugar > 120 mg/dl"
        )
        
        restecg = st.selectbox(
            'Resting ECG Results',
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }[x],
            help="Resting electrocardiographic results"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 8px;'>
            <h4 style='color: #1565C0;'>Exercise Test Results</h4>
        """, unsafe_allow_html=True)
        
        thalach = st.number_input(
            'Maximum Heart Rate',
            min_value=50,
            max_value=250,
            value=150,
            help="Maximum heart rate achieved during exercise"
        )
        
        exang = st.selectbox(
            'Exercise Induced Angina',
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Exercise induced angina"
        )
        
        oldpeak = st.number_input(
            'ST Depression',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="ST depression induced by exercise relative to rest"
        )
        
        slope = st.selectbox(
            'Slope of Peak Exercise ST Segment',
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }[x],
            help="The slope of the peak exercise ST segment"
        )
        
        ca = st.selectbox(
            'Number of Major Vessels',
            options=[0, 1, 2, 3, 4],
            help="Number of major vessels colored by fluoroscopy"
        )
        
        thal = st.selectbox(
            'Thalassemia',
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Normal",
                1: "Fixed Defect",
                2: "Reversible Defect",
                3: "Unknown"
            }[x],
            help="Type of thalassemia"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Add a separator
    st.markdown("<hr style='margin: 24px 0;'>", unsafe_allow_html=True)

    # Create columns for the prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button('Predict Heart Disease Risk', use_container_width=True)

    # Prediction code
    if predict_button:
        # Input validation
        valid_inputs = True
        valid_inputs &= validate_heart_input(age, 1, 120, "Age")
        valid_inputs &= validate_heart_input(trestbps, 50, 250, "Blood Pressure")
        valid_inputs &= validate_heart_input(chol, 100, 600, "Cholesterol")

        if valid_inputs:
            try:
                # Prepare input data
                user_input = [age, sex, cp, trestbps, chol, fbs, restecg, 
                            thalach, exang, oldpeak, slope, ca, thal]
                user_input = [float(x) for x in user_input]
                
                # Get prediction and confidence
                prediction, confidence = get_prediction_result(heart_disease_model, user_input)
                
                # Display result
                st.markdown("<div style='background-color: white; padding: 24px; border-radius: 8px; margin-top: 24px;'>", unsafe_allow_html=True)
                
                if prediction:
                    st.markdown("""
                        <h3 style='color: #d32f2f; text-align: center; margin-bottom: 16px;'>
                            ⚠️ Higher Risk of Heart Disease Detected
                        </h3>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <h3 style='color: #388e3c; text-align: center; margin-bottom: 16px;'>
                            ✅ Lower Risk of Heart Disease Detected
                        </h3>
                    """, unsafe_allow_html=True)

                # Display confidence if available
                if confidence is not None:
                    st.markdown(f"""
                        <p style='text-align: center; font-size: 18px; color: #666;'>
                            Confidence Level: {confidence:.1f}%
                        </p>
                    """, unsafe_allow_html=True)

                # Add recommendations based on risk factors
                st.markdown("<h4 style='color: #1565C0; margin-top: 24px;'>Key Observations:</h4>", unsafe_allow_html=True)
                
                recommendations = []
                if trestbps > 140:
                    recommendations.append("• High blood pressure detected. Regular monitoring and lifestyle modifications recommended.")
                if chol > 200:
                    recommendations.append("• Elevated cholesterol levels. Consider dietary changes and consultation with a healthcare provider.")
                if thalach > 220:
                    recommendations.append("• High maximum heart rate. Further cardiac evaluation may be needed.")
                if oldpeak > 2:
                    recommendations.append("• Significant ST depression observed. Cardiac evaluation recommended.")
                
                if recommendations:
                    for rec in recommendations:
                        st.markdown(f"<p style='color: #666;'>{rec}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #666;'>• All parameters are within normal ranges.</p>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # Disclaimer
                st.markdown("""
                    <p style='color: #666; font-size: 14px; text-align: center; margin-top: 16px;'>
                        ⚠️ This prediction is based on machine learning analysis and should not replace professional medical advice.
                        Please consult with a healthcare provider for proper diagnosis and treatment.
                    </p>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error("An error occurred during prediction. Please check your input values and try again.")
                st.exception(e)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.markdown("""
        <div style='background-color: #E3F2FD; padding: 24px; border-radius: 8px; margin-bottom: 24px;'>
            <h2 style='color: #1565C0; text-align: center; margin-bottom: 16px;'>Parkinson's Disease Risk Assessment</h2>
            <p style='text-align: center; color: #1976D2; font-size: 18px;'>Enter voice recording measurements for Parkinson's disease risk evaluation</p>
        </div>
    """, unsafe_allow_html=True)

    # Information box about Parkinson's disease
    st.info("""
    ℹ️ **About Parkinson's Disease Risk Assessment**
    
    This tool analyzes voice recording measurements to assess the risk of Parkinson's disease. The assessment uses various 
    acoustic parameters that can indicate early signs of the condition. These measurements are obtained from sustained 
    phonations of the vowel sound.
    
    The analysis requires specific voice measurements that are typically collected by medical professionals.
    """)

    # Input validation function
    def validate_parkinsons_input(value, min_val, max_val, name):
        if value < min_val or value > max_val:
            st.warning(f"⚠️ {name} should be between {min_val} and {max_val}")
            return False
        return True

    # Create columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 8px;'>
            <h4 style='color: #1565C0;'>Frequency Measurements</h4>
        """, unsafe_allow_html=True)
        
        fo = st.number_input(
            'Average Vocal Fundamental Frequency',
            min_value=0.0,
            max_value=500.0,
            value=120.0,
            help="Average vocal fundamental frequency in Hz"
        )
        
        fhi = st.number_input(
            'Maximum Vocal Fundamental Frequency',
            min_value=0.0,
            max_value=1000.0,
            value=200.0,
            help="Maximum vocal fundamental frequency in Hz"
        )
        
        flo = st.number_input(
            'Minimum Vocal Fundamental Frequency',
            min_value=0.0,
            max_value=500.0,
            value=100.0,
            help="Minimum vocal fundamental frequency in Hz"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 8px;'>
            <h4 style='color: #1565C0;'>Jitter Measurements</h4>
        """, unsafe_allow_html=True)
        
        Jitter_percent = st.number_input(
            'Jitter Percentage',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Measure of frequency instability in %"
        )
        
        Jitter_Abs = st.number_input(
            'Absolute Jitter',
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            help="Absolute measure of frequency instability"
        )
        
        RAP = st.number_input(
            'Relative Amplitude Perturbation',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Relative measure of period-to-period variability"
        )
        
        PPQ = st.number_input(
            'Period Perturbation Quotient',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Five-point period perturbation quotient"
        )
        
        DDP = st.number_input(
            'Average Absolute Difference',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Average absolute difference of differences between cycles"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 8px;'>
            <h4 style='color: #1565C0;'>Shimmer Measurements</h4>
        """, unsafe_allow_html=True)
        
        Shimmer = st.number_input(
            'Shimmer',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Local shimmer"
        )
        
        Shimmer_dB = st.number_input(
            'Shimmer in dB',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Shimmer in decibels"
        )
        
        APQ3 = st.number_input(
            'Three-point APQ',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Three-point amplitude perturbation quotient"
        )
        
        APQ5 = st.number_input(
            'Five-point APQ',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Five-point amplitude perturbation quotient"
        )
        
        APQ = st.number_input(
            'APQ',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="11-point amplitude perturbation quotient"
        )
        
        DDA = st.number_input(
            'Average Absolute Differences',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Average absolute differences between consecutive differences"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 8px;'>
            <h4 style='color: #1565C0;'>Harmonic Measurements</h4>
        """, unsafe_allow_html=True)
        
        NHR = st.number_input(
            'Noise to Harmonic Ratio',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Ratio of noise to tonal components"
        )
        
        HNR = st.number_input(
            'Harmonic to Noise Ratio',
            min_value=0.0,
            max_value=50.0,
            value=20.0,
            help="Ratio of harmonic to noise components"
        )
        
        RPDE = st.number_input(
            'RPDE',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Recurrence period density entropy measure"
        )
        
        DFA = st.number_input(
            'DFA',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Signal fractal scaling exponent"
        )
        
        spread1 = st.number_input(
            'Spread1',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Nonlinear measure of fundamental frequency variation"
        )
        
        spread2 = st.number_input(
            'Spread2',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Nonlinear measure of fundamental frequency variation"
        )
        
        D2 = st.number_input(
            'D2',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Correlation dimension"
        )
        
        PPE = st.number_input(
            'PPE',
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            help="Pitch period entropy"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Add a separator
    st.markdown("<hr style='margin: 24px 0;'>", unsafe_allow_html=True)

    # Create columns for the prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button('Predict Parkinson\'s Disease Risk', use_container_width=True)

    # Prediction code
    if predict_button:
        try:
            # Prepare input data
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                         Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                         RPDE, DFA, spread1, spread2, D2, PPE]
            
            user_input = [float(x) for x in user_input]
            
            # Get prediction and confidence
            prediction, confidence = get_prediction_result(parkinsons_model, user_input)
            
            # Display result
            st.markdown("<div style='background-color: white; padding: 24px; border-radius: 8px; margin-top: 24px;'>", unsafe_allow_html=True)
            
            if prediction:
                st.markdown("""
                    <h3 style='color: #d32f2f; text-align: center; margin-bottom: 16px;'>
                        ⚠️ Higher Risk of Parkinson's Disease Detected
                    </h3>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <h3 style='color: #388e3c; text-align: center; margin-bottom: 16px;'>
                        ✅ Lower Risk of Parkinson's Disease Detected
                    </h3>
                """, unsafe_allow_html=True)

            # Display confidence if available
            if confidence is not None:
                st.markdown(f"""
                    <p style='text-align: center; font-size: 18px; color: #666;'>
                        Confidence Level: {confidence:.1f}%
                    </p>
                """, unsafe_allow_html=True)

            # Add analysis of key measurements
            st.markdown("<h4 style='color: #1565C0; margin-top: 24px;'>Key Observations:</h4>", unsafe_allow_html=True)
            
            observations = []
            if Jitter_percent > 1.0:
                observations.append("• Elevated jitter percentage indicates increased vocal instability")
            if Shimmer > 0.5:
                observations.append("• Higher shimmer values suggest potential voice irregularities")
            if HNR < 20:
                observations.append("• Low harmonic-to-noise ratio may indicate voice quality issues")
            if DFA > 0.7:
                observations.append("• Elevated DFA value suggests changes in voice complexity")
            
            if observations:
                for obs in observations:
                    st.markdown(f"<p style='color: #666;'>{obs}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #666;'>• All voice measurements are within typical ranges</p>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Disclaimer
            st.markdown("""
                <p style='color: #666; font-size: 14px; text-align: center; margin-top: 16px;'>
                    ⚠️ This prediction is based on machine learning analysis of voice measurements and should not replace professional medical diagnosis.
                    Please consult with a healthcare provider for proper evaluation and diagnosis.
                </p>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error("An error occurred during prediction. Please check your input values and try again.")
            st.exception(e)
