import pandas as pd
import requests
import streamlit as st

# Define API Base URL
API_BASE_URL = "http://nanoporous-service:8000"


# Streamlit App
st.title("Nanoporous Materials Database")


# Add Material Section
st.header("1. Add Material")
uploaded_file = st.file_uploader("Upload a CIF file", type=["cif"])
material_name = st.text_input("Material Name")
if st.button("Add Material"):
    if uploaded_file and material_name:
        files = {"file": uploaded_file.getvalue()}
        params = {"name": material_name}
        response = requests.post(
            f"{API_BASE_URL}/add_material/", params=params, files=files
        )
        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(response.json().get("detail", "An error occurred"))
    else:
        st.warning("Please provide a material name and upload a CIF file.")


# Query Material Section
st.header("2. Query Material by Name")
query_name = st.text_input("Enter Material Name to Query")
if st.button("Query Material"):
    if query_name:
        params = {"name": query_name}
        response = requests.get(f"{API_BASE_URL}/query/", params=params)
        if response.status_code == 200:
            results = response.json()
            if results:
                st.write("Query Results:")
                st.json(results)
            else:
                st.warning("No materials found with the given name.")
        else:
            st.error(response.json().get("detail", "An error occurred"))
    else:
        st.warning("Please enter a material name.")


# Add List of Materials Section
st.header("3. Add a List of Materials")
multi_uploaded_files = st.file_uploader(
    "Upload Multiple CIF Files", type=["cif"], accept_multiple_files=True
)
if st.button("Add Multiple Materials"):
    if multi_uploaded_files:
        files = [
            ("files", (file.name, file.getvalue()))
            for file in multi_uploaded_files
        ]
        response = requests.post(f"{API_BASE_URL}/populate_db/", files=files)
        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(response.json().get("detail", "An error occurred"))
    else:
        st.warning("Please upload at least one CIF file.")


# Run SQL Query Section
st.header("4. Run SQL Query")
sql_query = st.text_area("Enter an SQL Query")
if st.button("Run SQL"):
    if sql_query:
        payload = {"query": sql_query}
        response = requests.post(f"{API_BASE_URL}/execute_sql/", json=payload)
        if response.status_code == 200:
            results = response.json()
            if "result" in results:
                st.write("Query Results:")
                st.dataframe(pd.DataFrame(results["result"]))
            else:
                st.success(results["message"])
        else:
            st.error(response.json().get("detail", "An error occurred"))
    else:
        st.warning("Please enter a valid SQL query.")


# Display Summary Statistics
st.header("5. Summary Statistics")
if st.button("Generate Summary Statistics"):
    query = "SELECT * FROM materials"
    payload = {"query": query}
    response = requests.post(f"{API_BASE_URL}/execute_sql/", json=payload)
    if response.status_code == 200:
        results = response.json()
        if "result" in results:
            df = pd.DataFrame(results["result"])
            numeric_cols = df.select_dtypes(include=["float", "int"]).columns
            summary = df[numeric_cols].describe(
                percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
            st.write("Summary Statistics:")
            st.dataframe(summary)
        else:
            st.warning("No data available for summary statistics.")
    else:
        st.error(response.json().get("detail", "An error occurred"))
