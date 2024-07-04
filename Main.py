import streamlit as st
import KnowRep
import Tools
import Model
import Processing
import os

image_url = "https://static.vecteezy.com/system/resources/previews/010/794/341/non_2x/purple-artificial-intelligence-technology-circuit-file-free-png.png"
st.sidebar.image(image_url, caption="", use_column_width=True)

# Streamlit UI
st.title("KNOWLEDGE REPRESENTATION ON STRUCTURED DATASET")
st.markdown('---')

# Sidebar with options

API_KEY = st.sidebar.text_input("Enter your API Key", type="password")
st.sidebar.subheader("SELECT FEATURES")
selected_feature = st.sidebar.radio("Choose a feature:",
                                    ["Insights Generation", "Chat with CSV", "ML Prediction"])

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("UPLOAD A CSV FILE", type="csv")

if selected_feature == "Insights Generation":
    st.subheader("Insights Generation")
    st.markdown('''This feature analyzes the uploaded CSV file to provide valuable insights about the data. 
                    It processes the dataset, generates descriptive statistics, identifies patterns, and creates  
                    visualizations. The output includes textual insights and charts that help users quickly understand 
                    key characteristics and trends in their data.''')
    st.markdown('---')


elif selected_feature == "Chat with CSV":
    st.subheader("Chat with CSV")
    st.markdown('''This interactive feature allows users to ask questions about their CSV data in natural language."
                " It uses the uploaded dataset to provide answers based on the content of the CSV file. "
                "Users can inquire about specific data points, relationships between variables, or summary statistics, "
                "making it easier to explore and understand their data without writing complex queries.''')
    st.markdown('---')
else:
    st.subheader("ML Prediction")
    st.markdown("This feature leverages machine learning algorithms to make predictions based on the uploaded CSV data."
                "Users can select a target column, and the system will attempt to predict values for that column using "
                "other columns as features. This can be useful for forecasting, classification tasks, or identifying "
                "influential factors in the dataset.")
    st.markdown('---')
Tools.make_folders()
# Main area
if uploaded_file is None:
    st.subheader("INSTRUCTIONS")
    st.markdown("1. Enter your API Key.")
    st.markdown("2. Select a feature.")
    st.markdown("3. Upload a CSV file.")
    st.markdown("4. Click 'Process' to run the selected feature.")
    st.markdown('---')

if uploaded_file is not None and API_KEY:
    with st.spinner("Saving uploaded file..."):
        try:
            if Tools.save_file(uploaded_file, Tools.ORIGINAL_PATH) == 1:
                st.success("File uploaded successfully!")
                Processing.preprocess_dataset()
            else:
                raise Exception
        except Exception as e:
            st.error(f"Failed to upload file: {e}")

    if st.button("Process"):
        if selected_feature == "Insights Generation":
            with st.spinner("Generating Insights for Your Dataset..."):
                try:
                    loaded_csv = Tools.load_csv_files(Tools.PATH)
                    sample = loaded_csv[0]
                    insights = KnowRep.generate_insights(sample)
                    st.subheader("Insights")
                    st.write(insights)
                except Exception as e:
                    st.error(f"Failed to generate insights: {e}")

            with st.spinner("Generating and visualizing charts..."):
                try:
                    sample = '\n'.join(loaded_csv[:3])
                    charts = KnowRep.generate_graph(sample)
                    Processing.Visualize_charts(charts)
                    st.success("Charts Created successfully!")
                except Exception as e:
                    st.error(f"Failed to visualize charts: {e}")

            with st.spinner("Listing the visualized charts..."):
                try:
                    visualized_files = os.listdir(Tools.VISUALIZE_PATH)
                    st.subheader("Visual Representation")
                    for file in visualized_files:
                        if file.endswith(".png"):
                            file_path = os.path.join(Tools.VISUALIZE_PATH, file)
                            st.image(file_path, caption=file, use_column_width=True)
                except Exception as e:
                    st.error(f"Failed to list visualized charts: {e}")

        elif selected_feature == "Chat with CSV":
            st.subheader("Chat with CSV")
            user_question = st.text_input("Ask a question about your CSV data:")
            if user_question:
                with st.spinner("Processing your question..."):
                    try:
                        loaded_csv = Tools.load_csv_files(Tools.PATH)
                        sample = loaded_csv[0]
                        answer = KnowRep.chat_with_csv(sample, user_question)
                        st.write("Answer:", answer)
                    except Exception as e:
                        st.error(f"Failed to process question: {e}")

        elif selected_feature == "ML Prediction":
            st.subheader("ML Prediction")

            with st.spinner("Generating ML prediction..."):
                try:
                    loaded_csv = Tools.load_csv_files(Tools.PATH)
                    sample = loaded_csv[0]
                    prediction = Model.prediction_model(sample)
                    st.write("Prediction Results:", prediction)
                except Exception as e:
                    st.error(f"Failed to generate prediction: {e}")

        with st.spinner("Cleaning up temporary files..."):
            try:
                Tools.delete_files()
                st.success("Temporary files deleted successfully!")
            except Exception as e:
                st.error(f"Failed to delete temporary files: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Bit Bandits")
