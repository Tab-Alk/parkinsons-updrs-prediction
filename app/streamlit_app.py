"""
Parkinson's Disease Assessment Tool
===================================

Clinical tool for UPDRS prediction using voice biomarkers.
For clinicians, patients, and families.
"""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.model import UPDRSPredictor
from app.preprocessing import create_sample_input

# Page configuration
st.set_page_config(
    page_title="Parkinson's Assessment Tool",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def load_predictor():
    """Load the UPDRS predictor."""
    try:
        predictor = UPDRSPredictor()
        return predictor, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


@st.cache_data
def load_sample_inputs():
    """Load sample inputs from JSON."""
    sample_file = Path("data/sample_inputs.json")
    if sample_file.exists():
        with open(sample_file, 'r') as f:
            return json.load(f)
    return {}


def render_home_page():
    """Render the HOME page - educational content about Parkinson's."""

    # Hero Section
    st.title("Parkinson's Disease Assessment")
    st.markdown("### Advanced voice biomarker analysis for precise monitoring of Parkinson's disease progression")
    st.markdown("---")

    # Understanding Parkinson's Section
    st.header("Understanding Parkinson's Disease")
    st.write("""
    Parkinson's disease is a progressive neurological disorder that affects movement control,
    caused by the gradual breakdown of dopamine-producing neurons in the brain. This condition
    affects approximately 1% of individuals over 60 years old, with symptoms typically developing
    gradually over years.
    """)

    st.subheader("Key Characteristics")
    st.write("""
    The disease manifests through several hallmark symptoms that typically begin gradually and
    progress over time. One of the most recognizable symptoms is **tremors**, often starting
    in the hands or fingers and appearing as a "pill-rolling" motion. **Bradykinesia** (slowed
    movement) makes simple tasks difficult and time-consuming, while **muscle rigidity** causes
    stiffness and can limit the range of motion.
    """)

    # Motor and Non-Motor Symptoms in columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Motor Symptoms")
        st.write("""
        - Tremors at rest
        - Muscle stiffness
        - Slowed movement (bradykinesia)
        - Impaired posture and balance
        - Loss of automatic movements
        """)

    with col2:
        st.markdown("#### Non-Motor Symptoms")
        st.write("""
        - Sleep disturbances
        - Loss of smell
        - Cognitive changes
        - Mood disorders
        - Autonomic dysfunction
        """)

    st.write("""
    Early detection and ongoing monitoring are crucial for managing Parkinson's disease effectively.
    Our assessment tool provides a non-invasive method to track disease progression through
    voice analysis, offering valuable insights for both patients and healthcare providers.
    """)

    st.markdown("---")

    # Voice Analysis Section
    st.header("Voice Analysis in Parkinson's Assessment")
    st.write("""
    Research has demonstrated that Parkinson's disease affects speech production in predictable ways,
    often before other motor symptoms become apparent. Our voice analysis technology captures these
    subtle changes, providing objective measurements that correlate with disease progression.
    """)

    st.subheader("How It Works")
    st.write("The assessment analyzes various acoustic features of the voice, including:")

    st.write("""
    - **Jitter:** Measures frequency variations in vocal cord vibrations
    - **Shimmer:** Quantifies amplitude variations in speech
    - **Harmonic-to-Noise Ratio (HNR):** Assesses voice quality
    - **Fundamental Frequency (F0):** Tracks changes in pitch
    """)

    st.info("""
    **Clinical Benefits**

    - **Early Detection:** Identify subtle changes before they're clinically apparent
    - **Objective Measurement:** Reduce subjectivity in symptom assessment
    - **Remote Monitoring:** Enable frequent assessments from home
    - **Treatment Tracking:** Monitor response to medication and therapy
    """)

    st.markdown("---")

    # UPDRS Scale Explanation
    st.header("Understanding the UPDRS Scale")
    st.write("""
    The **Unified Parkinson's Disease Rating Scale (UPDRS)** is the gold standard
    for assessing Parkinson's disease severity. Our tool predicts the motor examination
    section of the UPDRS, which evaluates key motor functions affected by Parkinson's.
    """)

    st.subheader("UPDRS Score Ranges")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("**Mild: 7-25**")
        st.write("Early-stage symptoms with minimal functional impact")

    with col2:
        st.warning("**Moderate: 25-40**")
        st.write("Noticeable symptoms affecting daily activities")

    with col3:
        st.error("**Severe: 40+**")
        st.write("Advanced symptoms requiring significant assistance")

    st.write("""
    The UPDRS evaluates various aspects of motor function, including speech, facial expression,
    tremor, rigidity, finger taps, hand movements, rapid alternating movements, leg agility,
    arising from chair, posture, gait, postural stability, and body bradykinesia.
    """)

    st.markdown("---")

    # About the Tool
    st.header("About This Assessment Tool")
    st.write("""
    Our Parkinson's Disease Assessment Tool leverages advanced machine learning algorithms to
    analyze voice patterns and predict UPDRS scores with high accuracy. The system was developed
    using a comprehensive dataset of voice recordings from Parkinson's patients across different
    stages of the disease.
    """)

    st.subheader("Intended Users")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**For Healthcare Providers**")
        st.write("""
        - Objective assessment of disease progression
        - Treatment effectiveness monitoring
        - Remote patient monitoring capabilities
        """)

    with col2:
        st.markdown("**For Patients & Caregivers**")
        st.write("""
        - Track symptoms over time
        - Share progress with healthcare team
        - Better understand disease progression
        """)

    st.info("""
    **Getting Started**

    Ready to assess Parkinson's symptoms? Navigate to the **Assessment** tab to start the
    voice analysis. The process is simple, non-invasive, and takes just a few minutes.

    **Note:** This tool is intended for informational purposes only and should
    not replace professional medical advice, diagnosis, or treatment.
    """)


def render_assessment_page():
    """Render the ASSESSMENT page - input form and results."""

    st.title("Clinical Assessment")
    st.caption("Enter voice analysis measurements to calculate UPDRS severity score")

    # Note about measurements
    st.info("""
    **Note for clinicians:** These measurements are obtained from specialized voice analysis equipment.
    14 acoustic features are required for accurate prediction. While this may seem like many inputs,
    each measurement contributes to the model's high accuracy (Test R² = 0.80, MAE = 3.8 points).
    """)

    # Input form
    with st.form("voice_measurement_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Patient Demographics")
            age = st.number_input("Age (years)", min_value=30, max_value=90, value=65,
                                help="Patient's age in years", key="age_input")
            sex = st.radio("Sex", options=[0, 1],
                         format_func=lambda x: "Male" if x == 0 else "Female",
                         horizontal=True,
                         help="0 = Male, 1 = Female", key="sex_input")

            st.markdown("#### Temporal")
            test_time = st.number_input("Days Since Baseline", min_value=-5.0, max_value=250.0,
                                      value=92.0, help="Days since initial measurement", key="test_time_input")

            st.markdown("#### Jitter Measurements")
            st.caption("Frequency variation indicators")
            jitter_pct = st.number_input("Jitter (%)", min_value=0.0001, max_value=0.15,
                                        value=0.0049, format="%.6f", key="jitter_pct_input")
            jitter_abs = st.number_input("Jitter (Absolute)", min_value=0.000001, max_value=0.0005,
                                        value=0.000035, format="%.6f", key="jitter_abs_input")

            st.markdown("#### Shimmer Measurements")
            st.caption("Amplitude variation indicators")
            shimmer = st.number_input("Shimmer", min_value=0.005, max_value=0.3,
                                    value=0.0253, format="%.4f", key="shimmer_input")
            shimmer_apq11 = st.number_input("Shimmer APQ11", min_value=0.002, max_value=0.3,
                                          value=0.0227, format="%.4f", key="shimmer_apq11_input")

        with col2:
            st.markdown("#### Voice Quality")
            st.caption("Harmonic-to-noise ratios")
            nhr = st.number_input("NHR (Noise-to-Harmonics)", min_value=0.0001, max_value=0.8,
                                value=0.0184, format="%.4f", key="nhr_input")
            hnr = st.number_input("HNR (Harmonics-to-Noise)", min_value=1.0, max_value=40.0,
                                value=21.92, format="%.2f", key="hnr_input")

            st.markdown("#### Nonlinear Complexity")
            st.caption("Dynamic complexity measures")
            rpde = st.number_input("RPDE (Recurrence Period Density Entropy)",
                                 min_value=0.1, max_value=1.0, value=0.5422, format="%.4f", key="rpde_input")
            dfa = st.number_input("DFA (Detrended Fluctuation Analysis)",
                                min_value=0.5, max_value=0.9, value=0.6436, format="%.4f", key="dfa_input")
            ppe = st.number_input("PPE (Pitch Period Entropy)",
                              min_value=0.02, max_value=0.8, value=0.2055, format="%.4f", key="ppe_input")

        submitted = st.form_submit_button("Calculate UPDRS Score", use_container_width=True)

    # Process submission
    if submitted:
        input_data = {
            'age': age, 'sex': sex, 'test_time': test_time,
            'Jitter(%)': jitter_pct, 'Jitter(Abs)': jitter_abs,
            'Shimmer': shimmer, 'Shimmer:APQ11': shimmer_apq11,
            'NHR': nhr, 'HNR': hnr,
            'RPDE': rpde, 'DFA': dfa, 'PPE': ppe
        }

        predictor, error = load_predictor()
        if error:
            st.error(f"Error: {error}")
            return

        with st.spinner("Calculating..."):
            result = predictor.predict(input_data)

        # Display validation feedback
        if result['validation']:
            validation = result['validation']

            # Show errors (red)
            if validation.errors:
                for error in validation.errors:
                    st.error(f"❌ {error}")

            # Show warnings (yellow)
            if validation.warnings:
                for warning in validation.warnings:
                    st.warning(f"⚠️ {warning}")

            # Show info messages (blue)
            if validation.info:
                for info in validation.info:
                    st.info(f"ℹ️ {info}")

        # Show prediction error if exists
        if result['error']:
            st.error(result['error'])
        else:
            # Store result in session state for persistence
            st.session_state.prediction_result = result

    # Display results if available (persists across reruns)
    if 'prediction_result' in st.session_state and st.session_state.prediction_result:
        display_results(st.session_state.prediction_result)


def display_results(result):
    """Display prediction results using native Streamlit components."""

    st.markdown("---")
    st.subheader("Assessment Results")

    # Main result display using columns for centering
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # UPDRS Score as large metric
        st.metric(label="UPDRS Score", value=f"{result['prediction']:.1f}",
                 help="Predicted UPDRS motor examination score")

        # Severity badge with appropriate styling
        severity = result['severity']
        if severity == 'Mild':
            st.success(f"**{severity} Severity** (7-25 points)")
        elif severity == 'Moderate':
            st.warning(f"**{severity} Severity** (25-40 points)")
        else:  # Severe
            st.error(f"**{severity} Severity** (40+ points)")

        # Confidence message
        st.write("")
        st.write(result['confidence'])

    st.markdown("---")

    # Personalized Patient Interpretation
    score = result['prediction']
    st.subheader("Patient-Specific Assessment")

    # Generate personalized interpretation based on exact score
    if score < 7:
        interpretation = f"""
        **Score: {score:.1f}** - This score is below the typical UPDRS range, which may indicate
        very minimal motor symptoms or measurement variability. Consider verifying the voice measurements
        and conducting a comprehensive clinical examination for accurate assessment.
        """
    elif score < 15:
        interpretation = f"""
        **Score: {score:.1f}** - This patient is in the **early mild range** of Parkinson's symptoms.
        Motor symptoms are present but minimal and typically do not significantly interfere with daily activities.
        This is an excellent time for patient education about the disease, establishing a baseline for future
        monitoring, and implementing lifestyle modifications (exercise, diet) that may slow progression.
        """
    elif score < 25:
        interpretation = f"""
        **Score: {score:.1f}** - This patient shows **mild Parkinson's symptoms** with noticeable but
        manageable motor impairments. Daily activities can generally be performed independently, though some
        tasks may take longer or require more effort. Regular monitoring every 3-6 months is recommended
        to track progression and adjust treatment as needed.
        """
    elif score < 32:
        interpretation = f"""
        **Score: {score:.1f}** - This patient is experiencing **moderate Parkinson's symptoms** in the
        lower-moderate range. Motor symptoms are affecting daily function and quality of life. The patient
        may benefit from medication optimization, physical therapy, and occupational therapy to maintain
        independence. Consider discussing adaptive strategies and assistive devices if needed.
        """
    elif score < 40:
        interpretation = f"""
        **Score: {score:.1f}** - This patient shows **significant moderate symptoms** approaching the
        severe range. Motor impairments notably impact daily activities and independence. This score suggests
        the need for comprehensive treatment review, possible medication adjustments, and evaluation for
        advanced therapies. Multidisciplinary care coordination is strongly recommended.
        """
    else:
        interpretation = f"""
        **Score: {score:.1f}** - This patient has **severe Parkinson's symptoms** requiring comprehensive
        care management. Significant motor impairments substantially limit daily activities and independence.
        Immediate review of current treatment regimen is recommended. Consider evaluation for advanced therapies
        (DBS, medication pump therapy), comprehensive rehabilitation services, and caregiver support resources.
        """

    st.info(interpretation)

    st.markdown("---")

    # Clinical Interpretation
    st.subheader("General Clinical Guidelines")

    with st.expander("For Clinicians", expanded=True):
        st.markdown("""
        **Mild (7-25):** Patient shows early-stage symptoms. Consider baseline assessment,
        patient education, and lifestyle modifications. Monitor progression with regular follow-ups.

        **Moderate (25-40):** Patient exhibits noticeable symptoms affecting daily function.
        Evaluate current treatment effectiveness, consider medication adjustments, and assess need for
        supportive therapies.

        **Severe (40+):** Advanced disease requiring comprehensive care coordination.
        Review medication regimen, consider advanced therapies, arrange multidisciplinary support.
        """)

    with st.expander("For Patients and Families"):
        st.write("""
        This score helps your healthcare team understand how Parkinson's is affecting you right now.
        The number itself is less important than tracking how it changes over time. Regular monitoring
        helps your doctor make the best decisions about your care. If you have questions about what
        this score means for you personally, please discuss with your neurologist or care team.
        """)

    # Validation warnings if any
    if result['validation']:
        if result['validation'].warnings:
            with st.expander("Clinical Warnings", expanded=True):
                for warning in result['validation'].warnings:
                    st.warning(warning)


def render_batch_page():
    """Render the BATCH UPLOAD page."""

    st.title("Batch Assessment")
    st.caption("Upload CSV file with multiple patient measurements for batch processing")

    st.info("""
    **What is Batch Processing?**

    Instead of entering patient data one at a time, upload a CSV file containing multiple patients'
    voice measurements. The system will:
    - Process all patients simultaneously
    - Generate UPDRS predictions for each patient
    - Provide summary statistics (average scores, severity distribution)
    - Allow you to download all results as a CSV file

    **Use Case:** Ideal for clinics managing multiple Parkinson's patients who need regular monitoring.
    """)

    with st.expander("View Required CSV Format"):
        st.markdown("**Your CSV file must include these 12 columns (in any order):**")
        st.markdown("age, sex, test_time, Jitter(%), Jitter(Abs), Shimmer, Shimmer:APQ11, NHR, HNR, RPDE, DFA, PPE")

        st.markdown("**Example CSV format:**")
        st.code("""age,sex,test_time,Jitter(%),Jitter(Abs),Shimmer,Shimmer:APQ11,NHR,HNR,RPDE,DFA,PPE
65,0,92.0,0.0049,0.000035,0.0253,0.0227,0.0184,21.92,0.5422,0.6436,0.2055
78,1,150.0,0.012,0.00008,0.065,0.055,0.045,18.5,0.62,0.72,0.35""", language="csv")

    uploaded_file = st.file_uploader("Choose CSV File", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully: {len(df)} records found")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Process All Records", use_container_width=True):
                predictor, error = load_predictor()
                if error:
                    st.error(f"Error: {error}")
                    return

                with st.spinner("Processing batch..."):
                    results_df = predictor.predict_batch(df)

                st.success(f"Batch processing complete: {len(results_df)} records processed")

                # Check for validation issues
                error_count = results_df['has_errors'].sum()
                warning_count = sum(len(w) for w in results_df['has_warnings'])

                if error_count > 0:
                    st.error(f"⚠️ {error_count} records had validation errors and were not processed.")
                if warning_count > 0:
                    st.warning(f"⚠️ {warning_count} validation warnings detected across records. Review results carefully.")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average UPDRS", f"{results_df['prediction'].mean():.1f}")
                with col2:
                    st.metric("Mild Cases", (results_df['severity'] == 'Mild').sum())
                with col3:
                    st.metric("Moderate Cases", (results_df['severity'] == 'Moderate').sum())
                with col4:
                    st.metric("Severe Cases", (results_df['severity'] == 'Severe').sum())

                st.markdown("#### Detailed Results")
                st.dataframe(results_df, use_container_width=True)

                # Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results (CSV)",
                    data=csv,
                    file_name="updrs_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


def main():
    """Main application."""

    # Initialize session state for result persistence
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None

    # Check if predictor loads (for setup errors)
    predictor, error = load_predictor()
    if error:
        st.error(f"Setup Error: {error}")
        st.info("""
        **Required Setup:**
        1. Run `python save_model.py` to generate model files
        2. Ensure these files exist:
           - models/random_forest_model.pkl
           - models/scaler.pkl
           - models/feature_config.json
        3. Refresh this page
        """)
        return

    # 3-Tab Navigation
    tab1, tab2, tab3 = st.tabs(["Home", "Assessment", "Batch Upload"])

    with tab1:
        render_home_page()

    with tab2:
        render_assessment_page()

    with tab3:
        render_batch_page()


if __name__ == "__main__":
    main()
