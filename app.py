import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Page config (should be first Streamlit call)
st.set_page_config(page_title="COVID-19 Data Analysis & Prediction", layout="wide")

# Load dataset with caching
@st.cache_data
def load_data():
    return pd.read_csv("covid_dataset_sample_1000.csv")

df = load_data()

# Load trained model
with open("trained_covid_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("🦠 COVID-19 Data Analysis & Prediction Dashboard")

# Sidebar menu
menu = st.sidebar.radio("Navigation", ["Dataset Overview", "Visual Analysis", "Prediction"])

# --- Dataset Overview ---
if menu == "Dataset Overview":
    st.header("📊 Dataset Overview")
    st.write("Here is the first part of the dataset used for analysis:")
    st.dataframe(df.head(20))

    st.subheader("Summary Statistics")
    st.write(df.describe())

# --- Visual Analysis ---
elif menu == "Visual Analysis":
    st.header("📈 Visual Analysis of COVID-19 Data")

    # --- Total Confirmed Cases by Country ---
    st.subheader("Total Confirmed Cases by Country")
    country_cases = df.groupby("country")[["confirmed", "deaths", "recovered"]].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=country_cases, x="country", y="confirmed", palette="viridis", ax=ax)
    ax.set_title("Total Confirmed Cases by Country")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[['confirmed', 'deaths', 'recovered', 'tests', 'vaccinations']].corr(),
                annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # --- Confirmed Cases Over Time (Top 5 Countries) ---
    st.subheader("Confirmed Cases Over Time (Top 5 Countries)")
    top_countries = df.groupby("country")["confirmed"].max().nlargest(5).index
    filtered_df = df[df["country"].isin(top_countries)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=filtered_df, x="date", y="confirmed", hue="country", marker="o", ax=ax)
    ax.set_title("Confirmed COVID-19 Cases Over Time by Country")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

    # --- Scatter Plot: Confirmed vs Deaths ---
    st.subheader("Scatter Plot: Confirmed Cases vs Deaths")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["confirmed"], df["deaths"], alpha=0.5, c="red")
    ax.set_title("Confirmed Cases vs Deaths")
    ax.set_xlabel("Confirmed Cases")
    ax.set_ylabel("Deaths")
    plt.grid(True)
    st.pyplot(fig)

    # --- Pie Chart: Distribution of Confirmed Cases by Country ---
    st.subheader("Distribution of Confirmed Cases by Country")
    country_cases_total = df.groupby("country")["confirmed"].sum()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(country_cases_total, labels=country_cases_total.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("Distribution of Confirmed COVID-19 Cases by Country")
    st.pyplot(fig)
# --- Prediction ---
elif menu == "Prediction":
    st.header("🧮 COVID-19 Confirmed Cases Prediction")

    st.write("Enter values below to predict the number of confirmed cases:")

    deaths = st.number_input("Number of Deaths", min_value=0, step=1)
    recovered = st.number_input("Number of Recovered Cases", min_value=0, step=1)
    tests = st.number_input("Number of Tests", min_value=0, step=100)
    vaccinations = st.number_input("Number of Vaccinations", min_value=0, step=100)

    if st.button("Predict"):
        # Prepare input
        input_data = pd.DataFrame([[deaths, recovered, tests, vaccinations]],
                                  columns=["deaths", "recovered", "tests", "vaccinations"])
        
        # (Optional) Scaling step if you used a scaler when training
        # with open("scaler.pkl", "rb") as f:
        #     scaler = pickle.load(f)
        # input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)

        st.success(f"✅ Predicted Confirmed Cases: {int(prediction[0])}")
