import streamlit as st
import KnowRep
import Model
import Tools
import Processing
import os

# Streamlit UI
st.title("Knowledge Representation on Structured Dataset")

# Sidebar with options
st.sidebar.header("Menu")
st.sidebar.markdown("---")
st.sidebar.markdown("\n")
st.sidebar.subheader("Instructions")
st.sidebar.markdown("1. Upload a CSV file.")
st.sidebar.markdown("2. Click 'Generate Insights' to process the data and generate insights.")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

# Main area
if uploaded_file is not None:
    with st.spinner("Saving uploaded file..."):
        try:
            if Tools.save_file(uploaded_file, Tools.ORIGINAL_PATH) == 1:
                st.success("File uploaded successfully!")
            else:
                raise Exception
        except Exception as e:
            st.error(f"Failed to upload file: {e}")

    if st.button("Generate Insights"):
        with st.spinner("Generating Insights for Your Dataset..."):
            try:
                Processing.preprocess_dataset()
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

        with st.spinner("Cleaning up temporary files..."):
            try:
                Tools.delete_files()
                st.success("Temporary files deleted successfully!")
            except Exception as e:
                st.error(f"Failed to delete temporary files: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Bit Bandits")
