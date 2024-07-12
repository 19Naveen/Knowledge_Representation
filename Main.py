import streamlit as st
import KnowRep
import Tools
import Model
import Processing
import chat_with_csv
import os

# Set page config
st.set_page_config(
    page_title="KnowRep",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/19Naveen/Knowledge_Representation/blob/Master/README.md',
        'About': "# One spot to know everything about your CSV!"
    }
)

if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
Tools.make_folders()

# Sidebar
with st.sidebar:
    st.image("https://static.vecteezy.com/system/resources/previews/010/794/341/non_2x/purple-artificial-intelligence-technology-circuit-file-free-png.png", width=200)
    st.title("KnowRep")
    st.write(st.session_state)
    st.session_state.api_key = st.text_input("Enter your API Key", type="password", value=st.session_state.api_key)
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None and st.session_state.api_key and not st.session_state.file_uploaded:
        if st.button("Process File"):
            with st.spinner("Processing..."):
                try:
                    if Tools.save_file(uploaded_file, Tools.ORIGINAL_PATH) == 1:
                        Processing.preprocess_dataset()
                        st.session_state.file_uploaded = True
                        KnowRep.make_llm(st.session_state.api_key)
                        st.success("File processed successfully!")
                    else:
                        raise Exception("Failed to save file")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Reset button
    if st.button("Reset Application"):
        with st.spinner("Resetting..."):
            try:
                Tools.delete_files()
                st.session_state.file_uploaded = False
                st.session_state.insights = ''
                st.session_state.display_insights = True
                st.success("Reset successful!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error during reset: {e}")

# Main content
st.markdown("# **KNOWLEDGE REPRESENTATION ON STRUCTURED DATASETS**")
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Insights Generation", "Chat with CSV", "ML Prediction"])

with tab1:
    st.header("Welcome to KnowRep")
    st.markdown("This tool helps you analyze and interact with your CSV data. One spot to know everything about your CSV!")
    st.markdown('')
    st.markdown('')
    st.markdown("##### :red[TO GET STARTED]")
    st.markdown("1. Enter your API key in the sidebar\n2. Upload a CSV file\n3. Click **Process File**\n4. Use the Options above to access different features\n5. Click **Reset Application** to reset the current workflow so you can start Analyzing your new CSV file")
    

with tab2:
    st.markdown("## Insights Generation")
    st.markdown('''This feature analyzes the uploaded CSV file to provide valuable insights about the data. 
                    It processes the dataset, generates descriptive statistics, identifies patterns, and creates  visualizations. The output includes textual insights and charts that help you quickly understand 
                    key characteristics and trends in your data.''')
    if 'insights' not in st.session_state:
        st.session_state['insights'] = 'Error Generating Insights Try Refreshing the Page'
    if 'display_insights' not in st.session_state:
        st.session_state.display_insights = False

    if st.session_state.file_uploaded:
        if st.button("Generate Insights", key="generate_insights", use_container_width=True):
            with st.spinner("Analyzing data..."):
                try:
                    sample = Tools.load_csv_files(Tools.PATH, key='string')
                    st.session_state.insights = KnowRep.generate_insights(sample)
                    st.session_state.display_insights = True
                    sample = Tools.load_csv_files(Tools.PATH, key='dataframe')
                    charts = KnowRep.generate_and_extract_charts(sample)
                    Processing.Visualize_charts(charts)

                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("Please upload and process a CSV file first.")
    if st.session_state.display_insights == True:
        st.markdown("### 📊 Insights")
        st.markdown(st.session_state.insights)
        st.markdown("### 📈 Visualizations")
        for file in os.listdir(Tools.VISUALIZE_PATH):
            if file.endswith(".png"):
                st.image(os.path.join(Tools.VISUALIZE_PATH, file), use_column_width=True)

with tab3:
    st.header("Chat with CSV")
    st.markdown('''This interactive feature allows you to ask questions about your CSV data in natural language. 
                It uses the uploaded dataset to provide answers to your questions. 
                you can inquire about specific data points, relationships between variables, or summary statistics, 
                making it easier to explore and understand their data without writing complex queries.''')
    if st.session_state.file_uploaded:
        user_question = None
        user_question = st.text_input("Ask a question about your data:")
        try:
            if user_question:
                chat_with_csv.initChat()
            if user_question:
                with st.spinner("Processing question..."):
                    chat_with_csv.handle_userinput(user_question)
                    print("User Question: ", user_question)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please upload and process a CSV file first.")

with tab4:
    st.header("ML Prediction")
    st.markdown('''This feature leverages machine learning algorithms to make predictions based on the uploaded CSV data. 
                Users can select a target column, and the system will attempt to predict values for that column using 
                other columns as features. This can be useful for forecasting, classification tasks, or identifying 
                influential factors in the dataset.''')
    if st.session_state.file_uploaded:
        if st.button("Run ML Prediction", use_container_width=True):
            with st.spinner("Running prediction model..."):
                try:
                    loaded_csv = Tools.load_csv_files(Tools.PATH)
                    Model.prediction_model(loaded_csv)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("Please upload and process a CSV file first.")


st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Bit Bandits")