import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Streamlit app
st.title("Anova / Significance Tests")
st.subheader("Differences between Ratings of Categories (men/women..)")

infoExpander = st.expander("Info about the app")
with infoExpander:
    st.markdown("""

             This app is valuable for researchers or analysts interested in understanding how ratings vary across different categories and identifying specific variables that show significant differences.
                 The summary table provides a quick overview of significant findings, aiding in decision-making and further investigation.

              Here's more about its functionality:

 - User Input:

The user can select a categorical variable (ordinal/categorical nonmetric) from the available columns in the dataset.
The user can choose one or more numeric variables for analysis.
- Data Filtering:

The app filters the dataset based on the user's selections, creating a subset of data for analysis.
- Summary Statistics:

The app displays summary statistics (e.g., mean, standard deviation) for the selected numeric variables, grouped by the chosen categorical variable.
- Statistical Analysis:

For each selected numeric variable, the app performs a one-way ANOVA test to determine if there are significant differences between groups defined by the categorical variable.
Tukey's HSD Test:

If significant differences are found, the app performs Tukey's HSD (Honestly Significant Difference) test for pairwise comparisons, calculating confidence intervals for the differences between group means.
 - Results Interpretation:

The app interprets the results of the statistical tests, indicating whether there is a significant difference between groups for each numeric variable.
 - Summary Table:

The app generates a summary table at the end, listing all selected numeric variables and indicating whether significant differences were found for each.


             """)

# Load your market research data
# Replace this with your actual data loading logic
# Upload Excel file
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Read data from Excel file
    data = pd.read_excel(uploaded_file)
    st.write(data.describe())
    st.dataframe(data)

    # Filter numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    # Filter ordinal/categorical nonmetric columns
    nonmetric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()

    # Sidebar for user input
    # Sidebar for user input
    selected_category = st.sidebar.selectbox("Select Categorical Variable:", nonmetric_columns)

    st.sidebar.subheader("")

    selected_numerics = st.sidebar.multiselect("Select Numeric Variables:", numeric_columns)

    # Filter data based on user input
    selected_data = data[[selected_category] + selected_numerics]

    # Group data by the selected category
    grouped_data = selected_data.groupby(selected_category)

    # Display summary statistics for each group
    st.write("Summary Statistics of selected numeric Variables - Mean values:")
    st.write(grouped_data[selected_numerics].mean())

    # st.write("Summary Statistics:")
    # st.write(grouped_data[selected_numerics].describe())
    st.sidebar.subheader("")

    anovaStart = st.sidebar.button("Start ANOVA!")

    if anovaStart:
        st.info("Analysis of Variance (ANOVA) and Tukey's HSD:")

        summary_data = {"Variable": [], "Significant Difference": []}

        for numeric_column in selected_numerics:
            f_statistic, p_value = f_oneway(
                *[grouped_data.get_group(name)[numeric_column] for name, group in grouped_data])
            st.subheader(f"Variable: {numeric_column}")
            st.write(f"F-statistic: {f_statistic}")
            st.write(f"P-value: {p_value}")

            # Interpretation of results
            if p_value < 0.05:
                st.success("There is a significant difference between the groups.")
                # Calculate Tukey's HSD for pairwise comparisons
                tukey_result = pairwise_tukeyhsd(selected_data[numeric_column], selected_data[selected_category])
                # st.write("Tukey's HSD:")
                # st.write(tukey_result.summary())

                # Display confidence intervals for pairwise comparisons
                st.write("Confidence Intervals:")
                ci_df = pd.DataFrame(data=tukey_result.confint, columns=["Lower CI", "Upper CI"])
                ci_df.index.name = "Pairwise Comparison"
                st.write(ci_df)
                summary_data["Variable"].append(numeric_column)
                summary_data["Significant Difference"].append("Yes")



            else:
                st.warning("There is no significant difference between the groups.")
                tukey_result = pairwise_tukeyhsd(selected_data[numeric_column], selected_data[selected_category])
                st.write("Confidence Intervals:")
                ci_df = pd.DataFrame(data=tukey_result.confint, columns=["Lower CI", "Upper CI"])
                ci_df.index.name = "Pairwise Comparison"
                st.write(ci_df)
                summary_data["Variable"].append(numeric_column)
                summary_data["Significant Difference"].append("No")
            st.divider()

        # Display summary table at the end
        st.subheader("Summary of Significant Differences for " + selected_category + ":")
        summary_table = pd.DataFrame(summary_data)
        st.write(summary_table)


