import os
import shutil
import streamlit as st
import src.KnowRep as KnowRep
import src.Tools as Tools
import src.Model as Model
import src.Processing as Processing
import src.chat_with_csv.chat_with_csv as chat_with_csv
import src.chat_with_csv.ui_template as ui
from src.Model import create_model, predict_model

st.set_page_config(
    page_title="KnowRep",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/19Naveen/Knowledge_Representation/blob/Master/README.md',
        'About': "# One spot to know everyth ing about your CSV!"
    }
)

if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
Tools.make_folders()


# Sidebar
with st.sidebar:
    
    st.image("https://i.ibb.co/vx7frqM8/purple-artificial-intelligence-technology-circuit-file-free-png.webp", width=200, caption="AI Image")
    st.title("KnowRep")
    st.session_state.api_key = st.text_input("Enter your API Key", type="password", value=st.session_state.api_key)
    isExampleFileSelected = st.toggle("Use Example File", value=False, key="example_file_toggle")
    uploaded_file = None
    if isExampleFileSelected:
        selectedExample = st.selectbox("Select Example", Tools.AVAILABLE_EXAMPLES.keys(), index=0)
        print("Selected Example File", selectedExample)
    else:
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        
    if (uploaded_file or isExampleFileSelected) and st.session_state.api_key and (not st.session_state.file_uploaded or isExampleFileSelected):
        if st.button("Process File"):
            with st.spinner("Processing..."):
                try:
                    if isExampleFileSelected:
                        source_path = Tools.AVAILABLE_EXAMPLES[selectedExample]
                        destination_path = f'{Tools.ORIGINAL_PATH}{selectedExample}.csv'
                        shutil.copy(source_path, destination_path)
                        print("Example File Uploaded!")
                    elif Tools.save_file(uploaded_file, Tools.ORIGINAL_PATH) == 1:
                        # File save is success
                        print("File Uploaded!")
                        pass
                    else:
                        raise Exception("Failed to save file")
                    
                    Processing.preprocess_dataset()
                    st.session_state.file_uploaded = True
                    KnowRep.make_llm(st.session_state.api_key)
                    sample_file = Tools.load_csv_files(Tools.PATH)
                    sample_file = sample_file[:5]
                    st.success("File processed successfully!")
                    
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
                st.session_state.chat_started = False
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
    st.markdown(
            """
                1. Enter your Gemini API key in the sidebar. [Click here to obtain one.](https://aistudio.google.com/app/apikey)
                2. Upload a CSV file.
                3. Click :orange[Process File].
                4. Use the options above to access different features.
                5. Click :orange[Reset Application] to reset the current workflow and start analyzing a new CSV file.
            """)
    if st.session_state.file_uploaded:
        st.markdown("##### Uploaded DataFrame")
        st.markdown("This is a preview of the first 5 rows of the uploaded CSV file.")
        st.dataframe(Tools.load_csv_files(Tools.PATH, key='dataframe').head(5), use_container_width=True)
    
    
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
    if st.session_state.display_insights:
        st.markdown("### 📊 Insights")
        st.markdown(st.session_state.insights)
        st.markdown("### 📈 Visualizations")
        for file in os.listdir(Tools.VISUALIZE_PATH):
            if file.endswith(".png"):
                st.image(os.path.join(Tools.VISUALIZE_PATH, file), use_container_width=True)

with tab3:
    st.header("Chat with CSV")
    st.markdown('''This interactive feature allows you to ask questions about your CSV data in natural language. 
                It uses the uploaded dataset to provide answers to your questions. 
                you can inquire about specific data points, relationships between variables, or summary statistics, 
                making it easier to explore and understand their data without writing complex queries.''')
    st.markdown(":red[EG] : **Can you describe the dataset?**")
               
    if 'chat_started' not in st.session_state:
        st.session_state['chat_started'] = False
    if st.session_state.file_uploaded:
        if st.button("Start Chat", use_container_width=True, disabled=st.session_state.chat_started):
            st.session_state.chat_started = True
            chat_with_csv.initChat()
        chat_container = st.container()

            
            
        if st.session_state['chat_started']:
            user_question = st.chat_input("Ask a question about your data:", key="user_question")
            if user_question:  
                try:
                    # st.sidebar.write("Chat history")
                    # st.sidebar.write(st.session_state['chat_history'])
                    with chat_container:
                        st.write(ui.CSS, unsafe_allow_html=True)
                        with st.spinner("Processing question..."):
                            chat_with_csv.handle_userinput(user_question)  
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.write(chat_with_csv.ui.bot_template("Sorry, Something went Wrong. Please Try Again"), unsafe_allow_html=True)
    else:
        st.warning("Please upload and process a CSV file first.")

            
with tab4:
    for key, default in {
        'model_accuracy': '',
        'submitted': None,
        'target_column': 'Auto',
        'prediction_type': 'Auto',
        'model_trained': None,
        'result': None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    st.header("ML Prediction")
    st.markdown('''This feature leverages machine learning algorithms to make predictions based on the uploaded CSV data. 
                Users can select a target column, and the system will attempt to predict values for that column using 
                other columns as features. This can be useful for forecasting, classification tasks, or identifying 
                influential factors in the dataset.''')

    if st.session_state.file_uploaded:
        st.markdown("##### Sample DataFrame")
        st.markdown("This is a preview of the first 5 rows of the uploaded CSV file.")
        st.dataframe(Tools.load_csv_files(Tools.PATH, key='dataframe').head(5), use_container_width=True)

        if not st.session_state.model_trained:
            columns_display = ['Auto'] + [col for col in Tools.fetch_columns()]
            st.session_state.target_column = st.selectbox('Select Target Column', columns_display, index=0)
            st.session_state.prediction_type = st.selectbox('Select Prediction Type', ['Auto', 'Classification', 'Regression'], index=0)

        if not st.session_state.model_trained and st.button('Train ML Model', use_container_width=True):
            with st.spinner("Training the model..."):
                try:
                    df = Tools.load_csv_files(Tools.PATH, key='dataframe')
                    if st.session_state.target_column == 'Auto':
                        st.session_state.target_column = KnowRep.get_target(df.head(5).to_string())
                    print('Target Column:', st.session_state.target_column)
                    if st.session_state.prediction_type == 'Auto':
                        st.session_state.prediction_type = KnowRep.dataset_type(df.head(5).to_string())
                    print('Prediction Type:', st.session_state.prediction_type)

                    accuracy, le = create_model(
                        df=df,
                        target_variable=st.session_state.target_column,
                        data_type=st.session_state.prediction_type
                    )
                    st.session_state.model_accuracy = accuracy
                    st.session_state.labelEncoder = le
                    st.session_state.model_trained = True
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")

        print('ml', st.session_state)
        if st.session_state.model_accuracy:
            st.success(f"Model trained successfully with accuracy: {st.session_state.model_accuracy}")

        if st.session_state.model_trained:
            with st.form("Prediction Input Form"):
                st.write("Enter data for each column:")
                input_data = {}
                prediction_columns = [col for col in Tools.fetch_columns()]
                prediction_columns.remove(st.session_state.target_column)
                column_types = Tools.column_dtype(prediction_columns)
                for col in prediction_columns:
                    if col in column_types['Numerical']:
                        input_data[col] = st.number_input(f"{col}:", key=f"num_{col}")
                    elif col in column_types['DateTime']:
                        input_data[col] = st.date_input(f"{col}:", key=f"date_{col}")
                    else:
                        input_data[col] = st.text_input(f"{col}:", key=f"text_{col}")
                submitted = st.form_submit_button("Predict")

            if submitted:
                print('Submitted:', input_data)
                with st.spinner("Running prediction..."):
                    try:
                        st.session_state.result = predict_model(
                            user_input=input_data,
                            column_dropped=st.session_state.target_column,
                            data_type=st.session_state.prediction_type,
                            le=st.session_state.labelEncoder,
                            columns=prediction_columns
                        )
                        st.success("Prediction completed successfully!")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

        if st.session_state.result:
            st.markdown("### Prediction Result")
            st.markdown(f"**Prediction:** {st.session_state.result}")

    else:
        st.warning("Please upload and process a CSV file first.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Bit Bandits")