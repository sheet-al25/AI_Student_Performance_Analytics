

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("student_performance_model.pkl")


model = load_model()


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/student_performance_demo.csv")


df = load_data()


# =========================================================
# FEATURE NAMES
# =========================================================

FEATURES = [
    "attendance_percent",
    "internal_marks_30",
    "assignment_percent",
    "study_hours_per_day",
    "previous_score_percent"
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_prediction(data):
    prediction = model.predict(data)
    return prediction[0]


def get_prediction_confidence(data):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)[0]
        return max(probabilities) * 100

    return None


def get_prediction_probabilities(data):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)[0]
        classes = model.classes_

        probability_data = pd.DataFrame({
            "Performance": classes,
            "Probability": probabilities * 100
        })

        return probability_data

    return pd.DataFrame()


def get_risk_from_prediction(prediction):

    if prediction == "High":
        return "Low Risk"

    elif prediction == "Medium":
        return "Medium Risk"

    else:
        return "High Risk"


def calculate_risk_score(
    attendance,
    internal_marks,
    assignment,
    study_hours,
    previous_score
):

    risk_score = 0

    if attendance < 75:
        risk_score += 1

    if internal_marks < 18:
        risk_score += 1

    if assignment < 60:
        risk_score += 1

    if study_hours < 2:
        risk_score += 1

    if previous_score < 60:
        risk_score += 1

    return risk_score


def get_risk_level(score):

    if score == 0:
        return "Low Risk"

    elif score <= 2:
        return "Medium Risk"

    else:
        return "High Risk"


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 AI Student Analytics")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "🤖 Performance Prediction",
        "🚨 Risk & Early Warning",
        "📋 Performance Breakdown",
        "🔮 What-If Simulator",
        "👥 Multi-Student Prediction",
        "🧠 Explainable AI",
        "📈 Dataset Analytics",
        "🧠 Model Analytics",
        "ℹ️ About Project"
    ]
)


# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Student Inputs")

attendance = st.sidebar.slider(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)

internal_marks = st.sidebar.slider(
    "Internal Marks (30)",
    min_value=0,
    max_value=30,
    value=20
)

assignment = st.sidebar.slider(
    "Assignment (%)",
    min_value=0,
    max_value=100,
    value=70
)

study_hours = st.sidebar.slider(
    "Study Hours / Day",
    min_value=0.0,
    max_value=12.0,
    value=2.0,
    step=0.5
)

previous_score = st.sidebar.slider(
    "Previous Score (%)",
    min_value=0,
    max_value=100,
    value=65
)


# =========================================================
# STUDENT DATA
# =========================================================

student_data = pd.DataFrame({
    "attendance_percent": [attendance],
    "internal_marks_30": [internal_marks],
    "assignment_percent": [assignment],
    "study_hours_per_day": [study_hours],
    "previous_score_percent": [previous_score]
})


# =========================================================
# DASHBOARD
# =========================================================

def show_dashboard():

    st.title("🎓 AI Student Performance Analytics")

    st.markdown(
        "### AI-powered student performance prediction, "
        "risk analysis and academic insights"
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👨‍🎓 Students",
            len(df)
        )

    with col2:
        st.metric(
            "📊 Features",
            len(FEATURES)
        )

    with col3:
        st.metric(
            "🤖 Best Model",
            "Logistic Regression"
        )

    with col4:
        st.metric(
            "🎯 Accuracy",
            "76%"
        )

    st.markdown("---")

    st.subheader("✨ Project Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🤖 Performance Prediction

        Predict whether a student is likely to perform at:

        - Low
        - Medium
        - High
        """)

    with col2:
        st.markdown("""
        ### 🚨 Risk Detection

        Identify students who may require academic support using:

        - Attendance
        - Internal marks
        - Assignments
        - Study hours
        - Previous score
        """)

    with col3:
        st.markdown("""
        ### 🧠 Explainable AI

        Understand which academic indicators are helping
        or negatively affecting the student's performance.
        """)

    st.markdown("---")

    st.subheader("📌 Current Student Input")

    st.dataframe(
        student_data,
        width="stretch",
        hide_index=True
    )


# =========================================================
# PERFORMANCE PREDICTION
# =========================================================

def show_prediction():

    st.title("🤖 Performance Prediction")

    st.write(
        "Enter student information using the sidebar "
        "to generate an AI-based performance prediction."
    )

    prediction = get_prediction(student_data)

    confidence = get_prediction_confidence(student_data)

    risk = get_risk_from_prediction(prediction)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Performance",
            prediction
        )

    with col2:

        if confidence is not None:
            st.metric(
                "Prediction Confidence",
                f"{confidence:.1f}%"
            )
        else:
            st.metric(
                "Prediction Confidence",
                "N/A"
            )

    with col3:
        st.metric(
            "Risk Level",
            risk
        )

    st.markdown("---")

    # -----------------------------------------------------
    # PERFORMANCE MESSAGE
    # -----------------------------------------------------

    if prediction == "High":

        st.success(
            "🌟 The student is predicted to have High performance."
        )

    elif prediction == "Medium":

        st.warning(
            "⚠️ The student is predicted to have Medium performance."
        )

    else:

        st.error(
            "🚨 The student is predicted to have Low performance."
        )

    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    st.subheader("📊 Prediction Probability")

    probability_data = get_prediction_probabilities(
        student_data
    )

    if not probability_data.empty:

        probability_display = probability_data.copy()

        probability_display["Probability"] = (
            probability_display["Probability"]
            .round(2)
            .astype(str)
            + "%"
        )

        st.dataframe(
            probability_display,
            width="stretch",
            hide_index=True
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        bars = ax.bar(
            probability_data["Performance"],
            probability_data["Probability"]
        )

        ax.set_ylabel("Probability (%)")
        ax.set_xlabel("Performance Level")
        ax.set_title("Model Prediction Probability")
        ax.set_ylim(0, 100)

        for bar, probability in zip(
            bars,
            probability_data["Probability"]
        ):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{probability:.1f}%",
                ha="center",
                fontweight="bold"
            )

        st.pyplot(fig)

        plt.close(fig)

    # -----------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------

    st.subheader("💡 Recommendations")

    recommendations = []

    if attendance < 75:
        recommendations.append(
            "📅 Improve attendance and maintain regular class participation."
        )

    if internal_marks < 18:
        recommendations.append(
            "📝 Focus on internal assessments and exam preparation."
        )

    if assignment < 60:
        recommendations.append(
            "📚 Complete assignments regularly and improve submission quality."
        )

    if study_hours < 2:
        recommendations.append(
            "⏰ Increase daily study time gradually."
        )

    if previous_score < 60:
        recommendations.append(
            "📈 Revise previous topics and work on weak academic areas."
        )

    if not recommendations:

        st.success(
            "✅ Current academic indicators look healthy. "
            "Continue maintaining the same consistency."
        )

    else:

        for recommendation in recommendations:
            st.write(recommendation)

    st.markdown("---")

    st.subheader("📋 Input Summary")

    st.dataframe(
        student_data,
        width="stretch",
        hide_index=True
    )


# =========================================================
# RISK & EARLY WARNING
# =========================================================

def show_risk():

    st.title("🚨 Risk & Early Warning")

    st.write(
        "This section identifies possible academic risk "
        "using a rule-based early warning system."
    )

    risk_score = calculate_risk_score(
        attendance,
        internal_marks,
        assignment,
        study_hours,
        previous_score
    )

    risk_level = get_risk_level(risk_score)

    prediction = get_prediction(student_data)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Risk Score",
            f"{risk_score}/5"
        )

    with col2:
        st.metric(
            "Risk Level",
            risk_level
        )

    with col3:
        st.metric(
            "ML Prediction",
            prediction
        )

    st.markdown("---")

    if risk_level == "Low Risk":

        st.success(
            "🟢 Student currently shows low academic risk."
        )

    elif risk_level == "Medium Risk":

        st.warning(
            "🟡 Student shows some indicators that may require attention."
        )

    else:

        st.error(
            "🔴 Student shows multiple risk indicators."
        )

    st.subheader("⚠️ Risk Indicators")

    risk_indicators = []

    if attendance < 75:
        risk_indicators.append(
            "Attendance is below 75%."
        )

    if internal_marks < 18:
        risk_indicators.append(
            "Internal marks are below 18/30."
        )

    if assignment < 60:
        risk_indicators.append(
            "Assignment performance is below 60%."
        )

    if study_hours < 2:
        risk_indicators.append(
            "Daily study time is below 2 hours."
        )

    if previous_score < 60:
        risk_indicators.append(
            "Previous academic score is below 60%."
        )

    if not risk_indicators:

        st.success(
            "✅ No major risk indicators detected."
        )

    else:

        for item in risk_indicators:
            st.warning(item)

    st.markdown("---")

    st.subheader("🎯 Suggested Actions")

    actions = []

    if attendance < 75:
        actions.append(
            "Increase class attendance."
        )

    if internal_marks < 18:
        actions.append(
            "Practice internal assessment topics."
        )

    if assignment < 60:
        actions.append(
            "Complete pending assignments."
        )

    if study_hours < 2:
        actions.append(
            "Create a consistent daily study schedule."
        )

    if previous_score < 60:
        actions.append(
            "Revise weak topics from previous assessments."
        )

    if not actions:

        st.info(
            "Maintain the current academic routine."
        )

    else:

        for action in actions:
            st.write(f"• {action}")


# =========================================================
# PERFORMANCE BREAKDOWN
# =========================================================

def show_breakdown():

    st.title("📋 Performance Breakdown")

    st.write(
        "Detailed analysis of the student's academic indicators."
    )

    indicators = pd.DataFrame({
        "Indicator": [
            "Attendance",
            "Internal Marks",
            "Assignment",
            "Study Hours",
            "Previous Score"
        ],
        "Value": [
            attendance,
            internal_marks,
            assignment,
            study_hours,
            previous_score
        ]
    })

    st.subheader("📊 Indicator Values")

    st.dataframe(
        indicators,
        width="stretch",
        hide_index=True
    )

    st.subheader("📈 Performance Indicators")

    normalized_values = [
        attendance,
        (internal_marks / 30) * 100,
        assignment,
        min((study_hours / 8) * 100, 100),
        previous_score
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        indicators["Indicator"],
        normalized_values
    )

    ax.set_ylabel("Normalized Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Student Performance Indicators")

    plt.xticks(rotation=20)

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# WHAT-IF SIMULATOR
# =========================================================

def show_what_if():

    st.title("🔮 What-If Simulator")

    st.write(
        "Change the academic indicators to see how the "
        "predicted performance may change."
    )

    st.subheader("🎯 What-If Inputs")

    col1, col2 = st.columns(2)

    with col1:

        what_attendance = st.slider(
            "What-if Attendance (%)",
            0,
            100,
            attendance
        )

        what_internal = st.slider(
            "What-if Internal Marks",
            0,
            30,
            internal_marks
        )

        what_assignment = st.slider(
            "What-if Assignment (%)",
            0,
            100,
            assignment
        )

    with col2:

        what_study = st.slider(
            "What-if Study Hours",
            0.0,
            12.0,
            study_hours,
            0.5
        )

        what_previous = st.slider(
            "What-if Previous Score (%)",
            0,
            100,
            previous_score
        )

    what_if_data = pd.DataFrame({
        "attendance_percent": [what_attendance],
        "internal_marks_30": [what_internal],
        "assignment_percent": [what_assignment],
        "study_hours_per_day": [what_study],
        "previous_score_percent": [what_previous]
    })

    current_prediction = get_prediction(student_data)

    what_if_prediction = get_prediction(
        what_if_data
    )

    current_confidence = get_prediction_confidence(
        student_data
    )

    what_if_confidence = get_prediction_confidence(
        what_if_data
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Current Prediction")

        st.metric(
            "Performance",
            current_prediction
        )

        if current_confidence is not None:
            st.metric(
                "Confidence",
                f"{current_confidence:.1f}%"
            )

    with col2:

        st.subheader("What-If Prediction")

        st.metric(
            "Performance",
            what_if_prediction
        )

        if what_if_confidence is not None:
            st.metric(
                "Confidence",
                f"{what_if_confidence:.1f}%"
            )

    st.markdown("---")

    comparison = pd.DataFrame({
        "Metric": [
            "Attendance",
            "Internal Marks",
            "Assignment",
            "Study Hours",
            "Previous Score"
        ],
        "Current": [
            attendance,
            internal_marks,
            assignment,
            study_hours,
            previous_score
        ],
        "What-If": [
            what_attendance,
            what_internal,
            what_assignment,
            what_study,
            what_previous
        ]
    })

    st.subheader("📊 Scenario Comparison")

    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True
    )


# =========================================================
# MULTI-STUDENT PREDICTION
# =========================================================

def show_bulk_prediction():

    st.title("👥 Multi-Student Prediction")

    st.write(
        "Upload a CSV file containing multiple students "
        "to generate predictions."
    )

    uploaded_file = st.file_uploader(
        "Upload Student CSV",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV file to begin multi-student prediction."
        )

        st.markdown("### Required Columns")

        for feature in FEATURES:
            st.write(f"• `{feature}`")

        return

    try:

        bulk_df = pd.read_csv(uploaded_file)

        missing_columns = [
            column
            for column in FEATURES
            if column not in bulk_df.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

            return

        for feature in FEATURES:

            bulk_df[feature] = pd.to_numeric(
                bulk_df[feature],
                errors="coerce"
            )

        if bulk_df[FEATURES].isnull().any().any():

            st.error(
                "Some required columns contain invalid or missing values."
            )

            return

        predictions = model.predict(
            bulk_df[FEATURES]
        )

        bulk_df["Predicted Performance"] = predictions

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                bulk_df[FEATURES]
            )

            bulk_df["Confidence (%)"] = (
                probabilities.max(axis=1) * 100
            ).round(2)

        bulk_df["Risk Level"] = bulk_df[
            "Predicted Performance"
        ].apply(
            get_risk_from_prediction
        )

        st.success(
            f"Successfully analyzed {len(bulk_df)} students."
        )

        st.subheader("📊 Prediction Results")

        st.dataframe(
            bulk_df,
            width="stretch",
            hide_index=True
        )

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Students",
                len(bulk_df)
            )

        with col2:
            high_risk_count = (
                bulk_df["Risk Level"] == "High Risk"
            ).sum()

            st.metric(
                "High Risk Students",
                high_risk_count
            )

        with col3:
            high_performance = (
                bulk_df["Predicted Performance"] == "High"
            ).sum()

            st.metric(
                "High Performers",
                high_performance
            )

        st.markdown("---")

        st.subheader("🚨 High-Risk Students")

        high_risk_students = bulk_df[
            bulk_df["Risk Level"] == "High Risk"
        ]

        if high_risk_students.empty:

            st.success(
                "No high-risk students detected."
            )

        else:

            st.dataframe(
                high_risk_students,
                width="stretch",
                hide_index=True
            )

        csv = bulk_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction Results",
            data=csv,
            file_name="student_predictions.csv",
            mime="text/csv",
            width="stretch"
        )

    except Exception as e:

        st.error(
            f"Error while processing file: {e}"
        )


# =========================================================
# EXPLAINABLE AI
# =========================================================

def show_explainable_ai():

    st.title("🧠 Explainable AI")

    st.write(
        "This section explains the student's academic "
        "indicators using simple, understandable rules."
    )

    st.markdown("---")

    indicators = {
        "Attendance": (
            attendance,
            75,
            "%"
        ),
        "Internal Marks": (
            internal_marks,
            18,
            "/30"
        ),
        "Assignment": (
            assignment,
            60,
            "%"
        ),
        "Study Hours": (
            study_hours,
            2,
            "hours/day"
        ),
        "Previous Score": (
            previous_score,
            60,
            "%"
        )
    }

    for name, values in indicators.items():

        value, threshold, unit = values

        if value >= threshold:

            st.success(
                f"✅ {name}: {value}{unit} — Healthy indicator"
            )

        else:

            st.warning(
                f"⚠️ {name}: {value}{unit} — Needs improvement"
            )

    st.markdown("---")

    st.subheader("🔎 Key Observations")

    observations = []

    if attendance >= 75:
        observations.append(
            "Attendance is at or above the recommended threshold."
        )
    else:
        observations.append(
            "Attendance is below the recommended threshold."
        )

    if internal_marks >= 18:
        observations.append(
            "Internal assessment performance is satisfactory."
        )
    else:
        observations.append(
            "Internal assessment performance needs improvement."
        )

    if assignment >= 60:
        observations.append(
            "Assignment performance is satisfactory."
        )
    else:
        observations.append(
            "Assignment performance needs improvement."
        )

    if study_hours >= 2:
        observations.append(
            "Study time meets the basic target."
        )
    else:
        observations.append(
            "Study time is below the basic target."
        )

    if previous_score >= 60:
        observations.append(
            "Previous academic performance is satisfactory."
        )
    else:
        observations.append(
            "Previous academic performance indicates an area for improvement."
        )

    for observation in observations:
        st.write(f"• {observation}")


# =========================================================
# DATASET ANALYTICS
# =========================================================

def show_dataset_analytics():

    st.title("📈 Dataset Analytics")

    st.write(
        "Overview of the student dataset used by the application."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Students",
            len(df)
        )

    with col2:
        st.metric(
            "Input Features",
            len(FEATURES)
        )

    with col3:
        st.metric(
            "Best Model Accuracy",
            "76%"
        )

    st.markdown("---")

    st.subheader("📊 Performance Distribution")

    if "performance" in df.columns:

        distribution = df[
            "performance"
        ].value_counts()

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.bar(
            distribution.index,
            distribution.values
        )

        ax.set_xlabel("Performance")
        ax.set_ylabel("Number of Students")
        ax.set_title("Student Performance Distribution")

        st.pyplot(fig)

        plt.close(fig)

    elif "Performance" in df.columns:

        distribution = df[
            "Performance"
        ].value_counts()

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.bar(
            distribution.index,
            distribution.values
        )

        ax.set_xlabel("Performance")
        ax.set_ylabel("Number of Students")
        ax.set_title("Student Performance Distribution")

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "Performance label column not found in dataset."
        )

    st.markdown("---")

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(20),
        width="stretch",
        hide_index=True
    )


# =========================================================
# MODEL ANALYTICS
# =========================================================

def show_model_analytics():

    st.title("🧠 Model Analytics")

    st.write(
        "Comparison of machine learning models used in the project."
    )

    model_comparison = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "Random Forest",
            "Logistic Regression"
        ],
        "Accuracy (%)": [
            66,
            68,
            76
        ]
    })

    st.subheader("🤖 Model Comparison")

    st.dataframe(
        model_comparison,
        width="stretch",
        hide_index=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        model_comparison["Model"],
        model_comparison["Accuracy (%)"]
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Machine Learning Model Accuracy")

    plt.xticks(rotation=15)

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("---")

    st.subheader("📌 Confusion Matrix")

    try:

        st.image(
            "confusion_matrix.png",
            width="stretch"
        )

    except Exception:

        st.info(
            "confusion_matrix.png not found."
        )

    st.markdown("---")

    st.subheader("📊 Feature Importance")

    try:

        st.image(
            "feature_importance.png",
            width="stretch"
        )

    except Exception:

        st.info(
            "feature_importance.png not found."
        )

    st.markdown("---")

    st.subheader("🔎 Key Insights")

    st.write(
        "• Logistic Regression achieved the highest accuracy among the tested models."
    )

    st.write(
        "• Attendance, internal marks, assignments, study hours and previous score are important academic indicators."
    )

    st.write(
        "• Model predictions can support early identification of students who may require academic attention."
    )


# =========================================================
# ABOUT PROJECT
# =========================================================

def show_about():

    st.title("ℹ️ About Project")

    st.write(
        "AI Student Performance Analytics is a machine learning "
        "based application designed to analyze and predict student "
        "academic performance."
    )

    st.markdown("---")

    st.subheader("🎯 Project Objective")

    st.write(
        "The main objective is to use academic indicators to "
        "predict student performance and identify students who "
        "may require additional academic support."
    )

    st.subheader("🤖 Machine Learning Models")

    st.write(
        "The project compares the following models:"
    )

    st.write("• Decision Tree")
    st.write("• Random Forest")
    st.write("• Logistic Regression")

    st.subheader("🏆 Best Model")

    st.success(
        "Logistic Regression — 76% Accuracy"
    )

    st.subheader("📊 Input Features")

    for feature in FEATURES:
        st.write(f"• {feature}")

    st.subheader("🎓 Performance Categories")

    st.write("🟢 High")
    st.write("🟡 Medium")
    st.write("🔴 Low")

    st.subheader("✨ Additional Features")

    st.write("• Risk and Early Warning System")
    st.write("• What-If Simulator")
    st.write("• Multi-Student Prediction")
    st.write("• Explainable AI")
    st.write("• Dataset Analytics")
    st.write("• Model Analytics")
    st.write("• Prediction Probability")

    st.subheader("⚠️ Important Note")

    st.info(
        "This application is designed as an academic analytics "
        "and decision-support system. Predictions should be used "
        "as supportive insights rather than as the sole basis for "
        "academic decisions."
    )


# =========================================================
# PAGE ROUTING
# =========================================================

if page == "🏠 Dashboard":

    show_dashboard()

elif page == "🤖 Performance Prediction":

    show_prediction()

elif page == "🚨 Risk & Early Warning":

    show_risk()

elif page == "📋 Performance Breakdown":

    show_breakdown()

elif page == "🔮 What-If Simulator":

    show_what_if()

elif page == "👥 Multi-Student Prediction":

    show_bulk_prediction()

elif page == "🧠 Explainable AI":

    show_explainable_ai()

elif page == "📈 Dataset Analytics":

    show_dataset_analytics()

elif page == "🧠 Model Analytics":

    show_model_analytics()

elif page == "ℹ️ About Project":

    show_about()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🎓 AI Student Performance Analytics | "
    "Machine Learning Capstone Project"
)