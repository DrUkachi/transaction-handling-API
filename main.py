import streamlit as st
import requests

# Define FastAPI base URL
BASE_URL = "http://localhost:8000"  # Update with your FastAPI server URL

# Function to call match-users endpoint
def call_match_users_api(transaction_id):
    url = f"{BASE_URL}/match-users/?transaction_id={transaction_id}"
    response = requests.post(url)
    payload={}
    return response.json()

# Function to call similar-transactions endpoint
def call_similar_transactions_api(input_string):
    url = f"{BASE_URL}/similar-transactions/?input_string={input_string}"
    payload = {}
    response = requests.post(url, json=payload)
    return response.json()

# Streamlit UI
def main():
    st.title("Deel Transaction Handling API with Streamlit")

    st.sidebar.title("Select an Option")
    options = ["Match Users", "Find Similar Transactions"]
    selected_option = st.sidebar.selectbox("Choose an option", options)

    if selected_option == "Match Users":
        st.subheader("Match Users")
        transaction_id = st.text_input("Enter Transaction ID")
        if st.button("Match"):
            if transaction_id:
                result = call_match_users_api(transaction_id)
                st.json(result)
            else:
                st.warning("Please enter a Transaction ID.")

    elif selected_option == "Find Similar Transactions":
        st.subheader("Find Similar Transactions")
        input_string = st.text_area("Enter Input String", height=100)
        if st.button("Find Similar"):
            if input_string:
                result = call_similar_transactions_api(input_string)
                st.json(result)
            else:
                st.warning("Please enter an Input String.")

if __name__ == "__main__":
    main()
