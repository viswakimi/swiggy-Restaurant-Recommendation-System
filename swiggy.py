import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("D:\\App\\vscode1\\swiggy\\cleaned_df.csv")

# Load Models
@st.cache_resource
def load_models():
    try:
        with open("D:\\App\\vscode1\\swiggy\\encoders_scalers.pkl", "rb") as f:
            encoders = pickle.load(f)
    except (FileNotFoundError, EOFError):
        encoders = {}

    try:
        with open("D:\\App\\vscode1\\swiggy\\kmeans_model.pkl", "rb") as f:
            kmeans = pickle.load(f)
    except (FileNotFoundError, EOFError):
        st.error("K-Means model not found. Please train and save the model.")
        st.stop()

    return encoders, kmeans

# Load Data and Models
cleaned_df = load_data()
encoders, kmeans = load_models()

# Load or Create Cost Scaler
if "cost_scaler" not in encoders:
    st.warning("⚠️ Cost scaler is missing. Training a new scaler.")
    cost_scaler = StandardScaler()
    cost_scaler.fit(cleaned_df[["cost"]])
    encoders["cost_scaler"] = cost_scaler

    # Save the updated encoders
    with open("D:\\App\\vscode1\\swiggy\\encoders_scalers.pkl", "wb") as f:
        pickle.dump(encoders, f)
else:
    cost_scaler = encoders["cost_scaler"]

# Load Other Encoders
if "city_encoder" in encoders and "cuisine_encoder" in encoders and "rating_scaler" in encoders:
    city_encoder = encoders["city_encoder"]
    cuisine_encoder = encoders["cuisine_encoder"]
    rating_scaler = encoders["rating_scaler"]
else:
    st.error("Encoders are missing. Please train and save them properly.")
    st.stop()

# UI Layout
st.title("🍽️ Restaurant Recommendation System")
st.header("Find the Best Restaurants for You!")

# User Input
city = st.selectbox("Select City", options=cleaned_df["city"].unique())
cuisine = st.multiselect("Select Cuisine", options=cleaned_df["cuisine"].unique())  # Use multiselect for multiple cuisines
cost = st.number_input("Enter Cost", min_value=0.0, step=0.1)
rating = st.slider("Select a Rating", min_value=0.0, max_value=5.0, step=0.1)

if st.button("Get Recommendations"):
    # Encode and scale features
    city_encoded = pd.DataFrame([[city_encoder.transform([city])[0]]], columns=["city_encoded"])

    # Cuisine Encoding Fix (One-Hot Encoding)
    try:
        cuisine_input = cuisine_encoder.transform([[cuisine]]).toarray()  # Convert categorical to numerical
        cuisine_encoded = pd.DataFrame(cuisine_input, columns=cuisine_encoder.get_feature_names_out(["cuisine"]))
    except ValueError:
        st.error("Invalid cuisine selection. Please choose a valid option.")
        st.stop()

    rating_scaled = pd.DataFrame(rating_scaler.transform([[rating]]), columns=["rating_scaled"])
    cost_scaled = pd.DataFrame(cost_scaler.transform([[cost]]), columns=["cost_scaled"])

    # Combine features
    user_input_df = pd.concat([cuisine_encoded, rating_scaled, cost_scaled, city_encoded], axis=1)

    # Predict cluster
    predicted_cluster = kmeans.predict(user_input_df)[0]
    st.write(f"Predicted Cluster: {predicted_cluster}")

    # Filter and recommend restaurants
    recommended_restaurants = cleaned_df[cleaned_df["cluster"] == predicted_cluster]
    recommended_restaurants = recommended_restaurants[["name", "link"]].rename(
        columns={"name": "Restaurant Name", "link": "Link"}
    ).reset_index(drop=True)

    # Display recommendations
    if recommended_restaurants.empty:
        st.warning("No matching restaurants found. Try adjusting your preferences!")
    else:
        st.subheader("🍽️ Recommended Restaurants:")
        st.table(recommended_restaurants.head(10))
