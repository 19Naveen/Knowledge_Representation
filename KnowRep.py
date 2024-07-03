from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import Tools

GOOGLE_PALM_API_KEY = 'AIzaSyCe-7cr2qgoxBS5LyVa_-fd2Fngh4Xwv2U'
llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_PALM_API_KEY,
    model="gemini-pro",
    temperature=0.5
)

def get_target(csv):
    columns = Tools.fetch_columns(csv)
    prompt = f'''
        You are a top-tier Data Analyst renowned for extracting valuable insights from datasets. Given a dataset, determ            ine which column is the target/Label column. You can only return the name of the column.
        Here is the dataset:{csv} and the columns: {columns}'''
    query_template = PromptTemplate(template=prompt, input_variables=["csv", "columns"])
    query = query_template.format(csv=csv, columns=columns)
    response = llm.invoke(query)
    return response.content


def dataset_type(csv):
    prompt = '''
        You are a top-tier Data Analyst renowned for extracting valuable insights from datasets. Given a dataset, determine whether it is best suited for classification, regression, or clustering. You can only return one of these three options: classification, regression, or clustering.
        Here is the dataset:{dataset}'''
    query_template = PromptTemplate(template=prompt, input_variables=["dataset"])
    query = query_template.format(dataset=csv)
    response = llm.invoke(query)
    return response.content


def generate_insights(csv):
    try:
        columns = Tools.fetch_columns(csv)
        stat = Tools.get_statistical_details()

        prompt = '''
        As a top-tier Data Analyst specializing in extracting insights, your task is to analyze the provided structured dataset to identify key patterns and trends. 
        The dataset at hand ({dataset}) contains the following columns: {Columns}. Please provide a comprehensive analysis including statistical details and actionable recommendations based on the following insights:
        
        About the Dataset:
        - Explain the dataset and its features in brief
        
        Key Patterns and Trends:
        - Identify significant trends observed in the data.
        - Highlight any correlations between columns.

        Actionable Recommendations:
        - Based on the analysis, propose actionable insights that can enhance decision-making processes.
        '''

        query_template = PromptTemplate(template=prompt, input_variables=["dataset", "Columns", "Stats"])
        query = query_template.format(dataset=csv, Columns=columns, Stats=stat)

        response = llm.invoke(query)
        return response.content

    except Exception as e:
        return f"Error generating insights: {e}"


def generate_graph(csv):
    columns = Tools.fetch_columns(csv)
    stats = Tools.get_statistical_details()

    prompt = f'''
        As a senior Data Analyst specializing in extracting and visualizing insights, your task is to analyze the provided structured dataset and identify key patterns and trends.

        Dataset: {csv}
        Columns: {columns}
        Statistical Summary: {stats}

        Based on your analysis, suggest the most appropriate types of visualizations for these columns. Your goal is to generate at least 5 different graphs from the following options: Line Chart, Pie Chart, Bar Chart, Histogram, Scatter Plot, Box Plot, Heatmap, and Area Chart.

        For each visualization, please:
        1. Specify the columns to be used for the x and y axes (if applicable)
        2. Recommend the chart type
        3. Provide a brief rationale for your choice (1-2 sentences)
        4. Suggest a key insight that might be gleaned from this visualization

        Please return your recommendations in the following format (note that the first index should be the x-axis, the second index should be the y-axis, and the third index should be the chart type):

        Example:
        1. ['YearBuilt', 'Price', 'Scatter Plot']
           - Rationale: Scatter plots are useful for identifying relationships between two numerical variables.
           - Potential Insight: This visualization could reveal trends such as whether newer properties tend to have higher prices.

        2. ['Category', None, 'Pie Chart']
           - Rationale: Pie charts effectively show the proportion of categories within a dataset.
           - Potential Insight: This chart can show the distribution of different categories within the dataset.

        3. ['NumericalColumn', None, 'Histogram']
           - Rationale: Histograms are ideal for showing the distribution of a single numerical variable.
           - Potential Insight: This chart can reveal the frequency distribution of the values in the numerical column.
           - Note: Ensure to specify an appropriate number of bins based on the data distribution.

        4. ['Date', 'Value', 'Line Chart']
           - Rationale: Line charts are perfect for showing trends over time.
           - Potential Insight: This visualization can highlight patterns and trends in the values over time.
           - Note: Ensure that the Date column is in a proper datetime format and sorted chronologically.

        5. ['NumericalColumn1', 'NumericalColumn2', 'Box Plot']
           - Rationale: Box plots are useful for summarizing the distribution of a dataset.
           - Potential Insight: This chart can reveal the spread and outliers in the data for the numerical column.

        (Continue for at least 5 visualizations)
        '''

    query_template = PromptTemplate(template=prompt, input_variables=["dataset", "Columns", "Stats"])
    query = query_template.format(dataset=csv, Columns=columns, Stats=stats)
    response = llm.invoke(query).content

    return Tools.Visual_rep(response)