import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Churn Analysis", layout="wide")

# ==============================
# SESSION STATE
# ==============================
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

# ==============================
# PREMIUM CSS 🔥
# ==============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.metric-card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    text-align: center;
    font-size: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: 0.3s;
}
.metric-card:hover {
    transform: scale(1.05);
}

.stButton>button {
    background: linear-gradient(45deg, #ff512f, #dd2476);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}

h1 {
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown("<h1>🚀 Customer Churn Analysis</h1>", unsafe_allow_html=True)
# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model/churn_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

# ==============================
# FILE UPLOAD
# ==============================
file = st.file_uploader("📁 Upload Customer CSV", type=["csv"])

if file:
    data = pd.read_csv(file)

    st.subheader("📄 Data Preview")
    st.dataframe(data.head())

    # ==============================
    # PREPROCESS (SAME LOGIC)
    # ==============================
    model_data = data.drop(
        ['Customer_ID','Customer_Status','Churn_Category','Churn_Reason'],
        axis=1,
        errors='ignore'
    )

    model_data.fillna("Missing", inplace=True)

    for col in model_data.select_dtypes(include='object'):
        if col in label_encoders:
            model_data[col] = model_data[col].astype(str)
            model_data[col] = model_data[col].apply(
                lambda x: x if x in label_encoders[col].classes_
                else label_encoders[col].classes_[0]
            )
            model_data[col] = label_encoders[col].transform(model_data[col])

    model_data = model_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # ==============================
    # PREDICT
    # ==============================
    if st.button("🔮 Run Prediction"):
        data['Prediction'] = model.predict(model_data)
        data['Churn_Probability'] = model.predict_proba(model_data)[:,1]

        data['Risk_Level'] = data['Churn_Probability'].apply(
            lambda x: "High Risk" if x > 0.7 else "Medium Risk" if x > 0.4 else "Low Risk"
        )

        data['Risk_Level'] = data['Risk_Level'].str.strip()

        data['Offer'] = data['Risk_Level'].map({
            "High Risk": "🔥 30% OFF + Personal Call",
            "Medium Risk": "🎁 15% OFF + Upgrade",
            "Low Risk": "⭐ Loyalty Bonus"
        })

        st.session_state.processed_data = data

    # ==============================
    # AFTER PREDICTION
    # ==============================
    if st.session_state.processed_data is not None:

        data = st.session_state.processed_data

        # ==============================
        # FILTER (GLOBAL - SAME LOGIC)
        # ==============================
        st.subheader("🎯 Filter Customers")

        risk_filter = st.selectbox("Select Risk Level", ["All", "High Risk", "Medium Risk", "Low Risk"])

        if risk_filter == "All":
            filtered = data
        else:
            filtered = data[data['Risk_Level'] == risk_filter]

        # ==============================
        # TABS UI 🔥
        # ==============================
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Search", "📋 Data"])

        # ==============================
        # TAB 1: DASHBOARD
        # ==============================
        with tab1:

            col1, col2, col3 = st.columns(3)

            emoji_map = {
                "High Risk": "🔥",
                "Medium Risk": "⚠️",
                "Low Risk": "✅"
            }

            if risk_filter == "All":
                label = "High Risk"
                value = (filtered['Risk_Level'] == "High Risk").sum()
                emoji = "🔥"
            else:
                label = risk_filter
                value = len(filtered)
                emoji = emoji_map.get(label, "📊")

            col1.markdown(f"<div class='metric-card'>👥 Total<br>{len(filtered)}</div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric-card'>{emoji} {label}<br>{value}</div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric-card'>📊 Avg Prob<br>{round(filtered['Churn_Probability'].mean(),2)}</div>", unsafe_allow_html=True)

            st.subheader("🔥 Top Customers")
            top10 = filtered.sort_values(by='Churn_Probability', ascending=False).head(10)
            st.dataframe(top10)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(filtered, names='Risk_Level', title="Risk Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(filtered, x='Churn_Probability', nbins=30, title="Churn Probability")
                st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(
                top10,
                x='Customer_ID',
                y='Churn_Probability',
                title="Top Risk Customers"
            )
            st.plotly_chart(fig, use_container_width=True)

        # ==============================
        # TAB 2: SEARCH
        # ==============================
        with tab2:
            search = st.text_input("Enter Customer ID")

            if search:
                result = data[data['Customer_ID'].astype(str).str.contains(search, case=False, na=False)]
                st.dataframe(result)

        # ==============================
        # TAB 3: DATA
        # ==============================
        with tab3:
            st.subheader("📋 Filtered Results")

            if filtered.empty:
                st.warning("⚠️ No customers found")
            else:
                st.dataframe(filtered)

        # ==============================
        # DOWNLOAD
        # ==============================
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "churn_results.csv")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown(" Churn Analytics 🚀")