import streamlit as st
import KnowRep
import Tools
import Model
import Processing
import chat_with_csv
import os


def predict():
    """
    Function to predict using the ML model.

    This function takes the user input from the session state and uses it as input for the prediction model.
    The result of the prediction is stored in the session state.

    Parameters:
        None

    Returns:
        None
    """
    user_input = st.session_state.user_input
    if user_input.strip() != '':
        st.write("2 User Input: ", user_input)
        with st.spinner("Creating prediction ML model..."):
            result = Model.prediction_model(df, target_variable, data_type, user_input)
            st.session_state.result = result
            print(result)

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
                st.session_state.result = ''
                st.success("Reset successful!")
                st.rerun()
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
        chat_container = st.container()
        
        user_question = st.text_input("Ask a question about your data:", key="user_question")
        
        if user_question:  # Check if there is a question submitted
            try:
                chat_with_csv.initChat()  # Initialize chat
                
                with chat_container:
                    with st.spinner("Processing question..."):
                        chat_with_csv.handle_userinput(user_question)  
            except Exception as e:
                st.error(f"Error: {e}")
                st.write(chat_with_csv.ui.bot_template("Sorry, Something went Wrong. Please Try Again"), unsafe_allow_html=True)
    else:
        st.warning("Please upload and process a CSV file first.")
            
with tab4:
    st.header("ML Prediction")
    st.markdown('''This feature leverages machine learning algorithms to make predictions based on the uploaded CSV data. 
                Users can select a target column, and the system will attempt to predict values for that column using 
                other columns as features. This can be useful for forecasting, classification tasks, or identifying 
                influential factors in the dataset.''')
    if st.session_state.file_uploaded:
        if 'result' not in st.session_state:
            st.session_state.result = ''
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ''
        if st.button("Run ML Prediction", use_container_width=True):
            with st.spinner("Running prediction model..."):
                try: 
                    sample_file = Tools.load_csv_files(Tools.PATH)
                    sample_file = sample_file[:5]
                    df = Tools.load_csv_files(Tools.PATH, key='dataframe')
                    target_variable = KnowRep.get_target(sample_file)  
                    data_type = KnowRep.dataset_type(sample_file)  
                    st.markdown('Enter values for the following features, separated by commas:')
                    st.write(', '.join(df.columns.drop(target_variable) if data_type != 'clustering' else df.columns))

                    st.text_input("Enter your input seperated by commas[,]", 
                                               key="user_input", 
                                               on_change= predict)
                
                except Exception as e:
                    st.error(f"Error: {e}")
        st.markdown(st.session_state.result)
    else:
        st.warning("Please upload and process a CSV file first.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Bit Bandits")
