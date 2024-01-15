import streamlit as st

#Z-TEST
import pandas as pd
import numpy as np
from scipy.stats import t

import matplotlib.pyplot as plt

from statsmodels.stats.proportion import proportions_ztest, proportion_confint

st.set_option('deprecation.showPyplotGlobalUse', False)

#Anova
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#Heatmap
import seaborn as sns
from scipy.stats import f_oneway



from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Simple Statistictools",
    page_icon="🧊",
    #layout="wide",
    #initial_sidebar_state="expanded",
)

#---Option Menu -------------------

option = option_menu(
	menu_title="Simple Statistics",
	options=["Z-Test", "Anova","Heatmap"],
	icons=["1-circle", "2-circle","3-circle"], #https://icons.getbootstrap.com/
	orientation="horizontal",
)

#Code um den Button-Design anzupassen
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ce1126;
    color: white;
    height: 3em;
    width: 14em;
    border-radius:10px;
    border:3px solid #000000;
    font-size:20px;
    font-weight: bold;
    margin: auto;
    display: block;
}
div.stButton > button:hover {
	background:linear-gradient(to bottom, #ce1126 5%, #ff5a5a 100%);
	background-color:#ce1126;
}
div.stButton > button:active {
	position:relative;
	top:3px;
}
</style>""", unsafe_allow_html=True)




def perform_z_test(data, column1, column2, alpha=0.05):
    # Extracting counts of successes and failures
    successes1 = int(data[column1].mean() * len(data))
    successes2 = int(data[column2].mean() * len(data))
    failures1 = len(data) - successes1
    failures2 = len(data) - successes2

    # Performing the z-test
    z_stat, p_value = proportions_ztest([successes1, successes2], [len(data), len(data)], alternative='two-sided')

    # Calculate confidence interval
    conf_interval1 = proportion_confint(successes1, len(data), alpha=alpha, method='normal')
    conf_interval2 = proportion_confint(successes2, len(data), alpha=alpha, method='normal')

    return z_stat, p_value, conf_interval1, conf_interval2







########################### Z-TEST ###############################################################################################################################################
if option =="Z-Test":
	# Streamlit app
	st.title("Proportions Z-Test App")
	st.subheader("Determine whether two population means are different")
	st.info("Upload one or two Excelfiles, and explore if the values of two columns with metric values differ significantly")

	zTestinfoExpander = st.expander("Info about Z-Tests")
	with zTestinfoExpander:
		st.write("""
	       A z-test is a statistical test used to determine whether two population means are different when the variances are known and the sample size is large.

	    The test statistic is assumed to have a normal distribution, and nuisance parameters such as standard deviation should be known in order for an accurate z-test to be performed.

	    KEY TAKEAWAYS
	    - A z-test is a statistical test to determine whether two population means are different when the variances are known and the sample size is large.
	    - A z-test is a hypothesis test in which the z-statistic follows a normal distribution. 
	    - A z-statistic, or z-score, is a number representing the result from the z-test.
	    - Z-tests are closely related to t-tests, but t-tests are best performed when an experiment has a small sample size.
	    - Z-tests assume the standard deviation is known, while t-tests assume it is unknown.      

	             """)

	infoExpander = st.expander("Info about this app")
	with infoExpander:
		st.write("""
	    The functionality of this Streamlit app is to conduct a two-sample z-test for proportions based on user-uploaded survey data in the form of an Excel file. The key features and the user flow are described below:

	    Functionality:
	    File Upload:

	    Users can upload an Excel file containing survey data.
	    The app assumes that each column in the Excel file represents responses from a different survey or group.
	    Column Selection:

	    Users can select two columns (surveys) from the uploaded data for comparison.
	    Analysis:

	    The app automatically calculates the proportions for each selected column.
	    It performs a two-sample z-test for proportions, evaluating whether there is a significant difference between the proportions of the selected columns.
	    Results Display:

	    The app displays the calculated z-statistic, p-value, and confidence intervals for each selected column.
	    It interprets the results, indicating whether the observed difference between proportions is statistically significant based on a predefined significance level (default is 0.05).
	    User Considerations:
	    Data Format:

	    The app assumes that the uploaded data is in Excel format.
	    Each column in the Excel file is considered a separate survey or group.
	    Column Selection:

	    Users need to choose two columns (surveys) for comparison.
	    It's essential to understand the nature of the data in each column and ensure they are comparable.
	    Interpretation:

	    The results include a z-statistic, p-value, and confidence intervals.
	    Users should interpret the p-value in the context of their chosen significance level (e.g., 0.05). A p-value below this level suggests a significant difference.
	    Assumptions:

	    The app assumes that the underlying data follows a binomial distribution, which is appropriate for proportions.
	    Expected Proportions:

	    The app calculates proportions automatically, removing the need for users to input expected proportions. This can be beneficial if users are unsure about the expected proportions.
	    Statistical Significance:

	    Users should be cautious in interpreting statistical significance. A non-significant result doesn't necessarily mean that the proportions are equal; it suggests that there isn't enough evidence to claim a significant difference.
	    Sample Size:

	    Larger sample sizes generally provide more reliable results. Users should consider the sample sizes of their surveys when interpreting the outcomes.
	    Data Quality:

	    Users should review their data for outliers, missing values, or other factors that might impact the reliability of statistical tests.
	    Feedback:

	    Users should carefully read the app's feedback on statistical significance and confidence intervals to make informed decisions based on their data.
	    Educational Purpose:

	    This app is designed for educational and exploratory purposes. It is not a substitute for a comprehensive statistical analysis tailored to specific research or business needs.
	    Users are encouraged to be familiar with statistical concepts, especially hypothesis testing and confidence intervals, to effectively interpret the results generated by the app. Additionally, consulting with a statistician or data analyst may be beneficial for more complex analyses or interpretations.       





	            """)

	############## Variables ##################

	data = pd.DataFrame()
	# st.write(data)

	# Upload Excel file
	uploaded_file1 = st.file_uploader("Upload an Excel file from Survey 1", type=["xlsx", "xls"])

	if uploaded_file1 is not None:
		# Read data from Excel file
		data1 = pd.read_excel(uploaded_file1)
		infoData1Expander = st.expander("Survey 1 - Dataset:")

		with infoData1Expander:
			st.write(data1.describe())
			st.dataframe(data1)

		# Choose numeric columns for the two surveys
		numeric_columns1 = data1.select_dtypes(exclude='object').columns
		st.sidebar.subheader("Column Selection - Survey 1")
		column1 = st.sidebar.selectbox("Select column for Survey 1:", numeric_columns1)

	uploaded_file2 = st.file_uploader("Upload an Excel file from Survey 2", type=["xlsx", "xls"])

	if uploaded_file2 is not None:
		# Read data from Excel file
		data2 = pd.read_excel(uploaded_file2)
		infoData2Expander = st.expander("Survey 2 - Dataset:")
		with infoData2Expander:
			st.write(data2.describe())
			st.dataframe(data2)

		# Choose columns for the two surveys
		st.sidebar.divider()
		st.sidebar.subheader("")
		st.sidebar.subheader("Column Selection - Survey 2")
		numeric_columns2 = data2.select_dtypes(exclude='object').columns
		column2 = st.sidebar.selectbox("Select column for Survey 2:", numeric_columns2)

		dataColumns1, dataColumns2 = st.columns(2)
		with dataColumns1:
			st.write("Column1 - from Survey 1")
			st.write(data1[column1])
			# st.write(data1[column1].mean())
			st.write(data1[column1].describe())
		# successes1 = int(data1[column1].mean() * len(data1))
		# st.write("successes: ",successes1)

		with dataColumns2:
			st.write("Column2 - from Survey 2")
			st.write(data2[column2])
			# st.write(data2[column2].mean())
			st.write(data2[column2].describe())
		# successes2 = int(data2[column2].mean() * len(data2))
		# st.write("successes: ",successes2)

		data = pd.DataFrame({
			'column1': data1[column1],
			'column2': data2[column2]
		})

		# st.write("Chosen Variables")
		# st.dataframe(data)

		# st.write(data['column1'].mean())

		st.subheader("")
		st.sidebar.subheader("")
		st.sidebar.subheader("")
		startZTest = st.sidebar.button("Start Z-Test!")

		if startZTest:

			from statsmodels.stats.weightstats import ztest as ztest

			st.divider()
			st.write("")
			st.subheader("Results:")
			# perform two sample z-test
			twoSampleZTest = ztest(data.column1, data.column2, value=0)

			thomasPValue = twoSampleZTest[1]
			st.write("twoSampleZTest: ", twoSampleZTest)

			st.metric(label="P-Value", value=thomasPValue.round(2))

			# Interpretation
			alpha = 0.05

			if thomasPValue < alpha:
				st.success("The difference between the proportions is statistically significant.")
			else:
				st.warning("The difference between the proportions is not statistically significant.")

			resultscol1, resultscols2 = st.columns(2)

			with resultscol1:
				st.write("Survey 1")
				m1 = data.column1.mean()
				s1 = data.column1.std()
				len1 = len(data1)
				st.write("Mean1: ", m1)
				st.write("Std 1: ", s1)
				st.write("Cases 1:", len1)
				dof1 = len(data.column1) - 1
				confidence = 0.95
				t_crit = np.abs(t.ppf((1 - confidence) / 2, dof1))
				confInterval1_left = (m1 - s1 * t_crit / np.sqrt(len1))
				confInterval1_right = (m1 + s1 * t_crit / np.sqrt(len1))
				st.write("Confidence Interval 1 left:", confInterval1_left)
				st.write("Confidence Interval 1 right:", confInterval1_right)
				# arr = np.random.normal(1, 1, size=100)
				arr1 = data.column1
				fig1, ax1 = plt.subplots()
				ax1.hist(arr1, bins=20)

				st.pyplot(fig1)

			with resultscols2:
				st.write("Survey 2")
				m2 = data.column2.mean()
				s2 = data.column2.std()
				len2 = len(data2)
				st.write("Mean2: ", m2)
				st.write("Std 2: ", s2)
				st.write("Cases 2:", len2)
				dof2 = len(data.column2) - 1
				confidence = 0.95
				t_crit = np.abs(t.ppf((1 - confidence) / 2, dof2))
				confInterval2_left = (m2 - s2 * t_crit / np.sqrt(len2))
				confInterval2_right = (m2 + s2 * t_crit / np.sqrt(len2))
				st.write("Confidence Interval 2 left:", confInterval2_left)
				st.write("Confidence Interval 2 right:", confInterval2_right)
				# arr = np.random.normal(2, 2, size=200)
				arr2 = data.column2
				fig2, ax2 = plt.subplots()
				ax2.hist(arr2, bins=20)

				st.pyplot(fig2)






########################### Anova ###############################################################################################################################################
if option =="Anova":

	# Streamlit app
	st.title("Anova / Significance Tests")
	st.subheader("Differences between Ratings of Categories (men/women..)")
	st.info("Upload an Excelfile and chose one categorical variable and one or more numeric variables and find out if the average values of the numeric variable(s) differ significantly between the categories. ")


	infoExpander = st.expander("Info about the app")
	with infoExpander:
		st.markdown("""

	             This app is for researchers or analysts interested in understanding how ratings vary across different categories and identifying specific variables that show significant differences.
	                 The summary table provides a quick overview of significant findings, aiding in decision-making and further investigation.

	              Here's more about its functionality:

	 - User Input:

	The user can select categorical variable(s) (ordinal/categorical nonmetric) from the available columns in the dataset.
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

	summary_data = {"Category": [], "Variable": [], "Significant Difference": []}

	if uploaded_file is not None:
		# Read data from Excel file
		data = pd.read_excel(uploaded_file)
		st.info("Descripton of the numeric variables:")
		st.write(data.describe())

		dataFrameExpander = st.expander("Show dataset >>")
		with dataFrameExpander:
			st.dataframe(data)

		# Filter numeric columns
		numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

		# Filter ordinal/categorical nonmetric columns
		nonmetric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()

		# Sidebar for user input
		# Sidebar for user input
		selected_categories = st.sidebar.multiselect("Select Categorical Variable:", nonmetric_columns)
		st.sidebar.subheader("")
		selected_numerics = st.sidebar.multiselect("Select Numeric Variables:", numeric_columns)
		st.sidebar.subheader("")

		anovaStart = st.sidebar.button("Start ANOVA!")

		if anovaStart:
			st.write("")
			st.info("Analysis of Variance (ANOVA) and Tukey's HSD:")
			st.write("")

			for nonmetric_column in selected_categories:

				# Display summary statistics for each group
				st.divider()
				st.title("Test of " + nonmetric_column)
				st.write("Mean values of selected numeric Variables -  " + nonmetric_column)

				# Filter data based on user input
				# selected_data = data[selected_categories + selected_numerics]

				# chatOpeni:
				selected_data = data[[nonmetric_column] + selected_numerics]

				st.write("selected_data:", selected_data)

				# Group data by the selected category
				# grouped_data = selected_data.groupby(nonmetric_column)
				# Vorschlag chatOpenai
				grouped_data = data.groupby(nonmetric_column)

				st.write(grouped_data[selected_numerics].mean())

				# st.write("Summary Statistics:")
				# st.write(grouped_data[selected_numerics].describe())

				for numeric_column in selected_numerics:
					# f_statistic, p_value = f_oneway(*[grouped_data.get_group(name)[numeric_column] for name, group in grouped_data])
					# ChatOpenAI:

					# Extract values for the ANOVA test
					group_values = [group[1][numeric_column].dropna().values for group in grouped_data]

					# Perform ANOVA test
					f_statistic, p_value = f_oneway(*group_values)

					st.subheader(f"Variable: {numeric_column}")
					st.write(f"F-statistic: {f_statistic}")
					st.write(f"P-value: {p_value}")

					# Interpretation of results
					if p_value < 0.05:
						st.success("There is a significant difference between the groups.")

						# Calculate Tukey's HSD for pairwise comparisons
						# tukey_result = pairwise_tukeyhsd(selected_data[numeric_column], selected_data[nonmetric_column])
						# ChatOpenAi: Calculate Tukey's HSD for pairwise comparisons
						tukey_result = pairwise_tukeyhsd(data[numeric_column], data[nonmetric_column])

						# st.write("Tukey's HSD:")
						# st.write(tukey_result.summary())

						# Display confidence intervals for pairwise comparisons
						st.write("Confidence Intervals:")
						ci_df = pd.DataFrame(data=tukey_result.confint, columns=["Lower CI", "Upper CI"])
						ci_df.index.name = "Pairwise Comparison"
						st.write(ci_df)
						summary_data["Category"].append(nonmetric_column)
						summary_data["Variable"].append(numeric_column)
						summary_data["Significant Difference"].append("Yes")



					else:
						st.warning("There is no significant difference between the groups.")
						tukey_result = pairwise_tukeyhsd(selected_data[numeric_column], selected_data[nonmetric_column])

						# st.write("Tukey's HSD:")
						# st.write(tukey_result.summary())

						st.write("Confidence Intervals:")
						ci_df = pd.DataFrame(data=tukey_result.confint, columns=["Lower CI", "Upper CI"])
						ci_df.index.name = "Pairwise Comparison"
						st.write(ci_df)
						summary_data["Category"].append(nonmetric_column)
						summary_data["Variable"].append(numeric_column)
						summary_data["Significant Difference"].append("No")
					st.divider()

				# Display summary table at the end
				st.subheader("Summary of Significant Differences:")
				summary_table = pd.DataFrame(summary_data)
				st.write(summary_table)


########################### Heatmap ###############################################################################################################################################
if option =="Heatmap":

	st.title("Supersimple Heatmap")
	st.info(
		"Upload a Excel, choose two categorical variables and one numeric variable and  explore the differences between the mean values of the category combinations")

	# Load Excel file
	uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
	if uploaded_file is not None:
		df = pd.read_excel(uploaded_file)

		# Display DataFrame
		st.dataframe(df)

		# Select categorical and numeric variables
		categorical_columns = df.select_dtypes(include='object').columns
		numeric_columns = df.select_dtypes(exclude='object').columns

		categorical_var1 = st.sidebar.selectbox("Select Categorical Variable 1", categorical_columns)
		categorical_var2 = st.sidebar.selectbox("Select Categorical Variable 2", categorical_columns)
		numeric_var = st.sidebar.selectbox("Select Numeric Variable", numeric_columns)

		crossTableColumn1, crossTableColumn2 = st.columns(2)

		with crossTableColumn1:

			# Create crosstab
			crosstab = pd.crosstab(df[categorical_var1], df[categorical_var2], values=df[numeric_var], aggfunc='mean')

			# Display crosstab
			st.write("")
			st.write("Editable Crosstab:")

			crosstab = st.data_editor(crosstab, num_rows="dynamic")

		with crossTableColumn2:
			# Display table with number of cases
			st.write("")
			st.write("Number of Cases:")
			cases_table = pd.crosstab(df[categorical_var1], df[categorical_var2])
			st.dataframe(cases_table)

		# st.dataframe(crosstab)

		# Display heatmap using matplotlib and seaborn
		st.write("")
		st.subheader("Heatmap")
		st.info("with mean values for " + numeric_var)

		farbwahl = st.selectbox("Colorscheme", ("brg", "viridis", 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
												'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
												'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
												'turbo', 'nipy_spectral', 'gist_ncar'))

		plt.figure(figsize=(10, 6))
		sns.heatmap(crosstab, annot=True, cmap=farbwahl, fmt=".1f")
		# sns.heatmap(crosstab, annot=True, cmap="viridis", fmt=".1f")
		st.pyplot()

		# Display bar chart for mean values and number of cases of categorical_var1
		st.write("")
		st.write(f"Mean Values and Number of Cases for {categorical_var1}:")
		grouped_var1 = df.groupby(categorical_var1)
		mean_values_var1 = grouped_var1[numeric_var].mean()
		count_var1 = grouped_var1[numeric_var].count()

		fig, ax1 = plt.subplots(figsize=(10, 6))
		ax2 = ax1.twinx()

		ax1.bar(mean_values_var1.index, mean_values_var1, alpha=0.7, label=f"Mean {numeric_var}")
		ax2.plot(count_var1.index, count_var1, color='red', marker='o', label=f"Number of Cases", linestyle='dashed')

		ax1.set_xlabel(categorical_var1)
		ax1.set_ylabel(f"Mean {numeric_var}", color='blue')
		ax2.set_ylabel("Number of Cases", color='red')
		plt.title(f"Mean {numeric_var} and Number of Cases for {categorical_var1}")
		ax1.tick_params(axis='y', labelcolor='blue')
		ax2.tick_params(axis='y', labelcolor='red')

		# Add annotations on top of the bars
		for i, (mean_value, count) in enumerate(zip(mean_values_var1, count_var1)):
			ax1.text(i, mean_value, f"{mean_value:.1f}\n{count}", ha='center', va='bottom', color='black')

		st.pyplot()

		# Display bar chart for mean values and number of cases of categorical_var2
		st.write("")
		st.write(f"Mean Values and Number of Cases for {categorical_var2}:")
		grouped_var2 = df.groupby(categorical_var2)
		mean_values_var2 = grouped_var2[numeric_var].mean()
		count_var2 = grouped_var2[numeric_var].count()

		fig, ax1 = plt.subplots(figsize=(10, 6))
		ax2 = ax1.twinx()

		ax1.bar(mean_values_var2.index, mean_values_var2, alpha=0.7, label=f"Mean {numeric_var}")
		ax2.plot(count_var2.index, count_var2, color='red', marker='o', label=f"Number of Cases", linestyle='dashed')

		ax1.set_xlabel(categorical_var2)
		ax1.set_ylabel(f"Mean {numeric_var}", color='blue')
		ax2.set_ylabel("Number of Cases", color='red')
		plt.title(f"Mean {numeric_var} and Number of Cases for {categorical_var2}")
		ax1.tick_params(axis='y', labelcolor='blue')
		ax2.tick_params(axis='y', labelcolor='red')

		# Add annotations on top of the bars
		for i, (mean_value, count) in enumerate(zip(mean_values_var2, count_var2)):
			ax1.text(i, mean_value, f"{mean_value:.1f}\n{count}", ha='center', va='bottom', color='black')

		st.pyplot()

		st.divider()

		st.subheader("")
		# Perform ANOVA test
		st.subheader("ANOVA Tests:")

		# Handling missing values option
		missing_values_option = st.radio("Choose how to handle missing values:",
										 ["Do nothing", "Drop Missing Values", "Impute Missing Values"])

		if missing_values_option == "Do nothing":
			df_clean = df

		if missing_values_option == "Drop Missing Values":
			df_clean = df.dropna(subset=[numeric_var])
			st.write("")
			st.success("Missing values dropped.")
			st.write("")

		if missing_values_option == "Impute Missing Values":
			# Impute missing values with the mean
			mean_value = df[numeric_var].mean()
			df_clean = df.copy()
			df_clean[numeric_var].fillna(mean_value, inplace=True)
			st.write("")
			st.success(f"Missing values imputed with mean value: {mean_value:.2f}")
			st.write("")

		st.subheader("")
		st.write("Anova of " + categorical_var1)
		categories = df_clean[categorical_var1].unique()
		data_by_category = [df_clean[numeric_var][df_clean[categorical_var1] == category] for category in categories]

		f_statistic, p_value = f_oneway(*data_by_category)

		st.write(f"F-statistic: {f_statistic:.2f}")
		st.write(f"P-value: {p_value:.4f}")

		if not pd.isnull(f_statistic) and not pd.isnull(p_value):
			if p_value < 0.05:
				st.success("The p-value is less than 0.05, indicating that there is a significant difference in means.")
			else:
				st.info(
					"The p-value is greater than 0.05, indicating that there is no significant difference in means.")
		else:
			st.warning("Unable to calculate F-statistic and p-value. Please drop missing values / check your data.")

		st.subheader("")
		st.write("Anova of " + categorical_var2)
		categories = df_clean[categorical_var2].unique()
		data_by_category = [df_clean[numeric_var][df_clean[categorical_var2] == category] for category in categories]

		f_statistic, p_value = f_oneway(*data_by_category)

		st.write(f"F-statistic: {f_statistic:.2f}")
		st.write(f"P-value: {p_value:.4f}")

		if not pd.isnull(f_statistic) and not pd.isnull(p_value):
			if p_value < 0.05:
				st.success("The p-value is less than 0.05, indicating that there is a significant difference in means.")
			else:
				st.info(
					"The p-value is greater than 0.05, indicating that there is no significant difference in means.")
		else:
			st.warning("Unable to calculate F-statistic and p-value. Please drop missing values / check your data.")

		if missing_values_option != "Do nothing":

			st.write("")
			extraChartsExpander = st.expander("Results without missing values:")
			with extraChartsExpander:
				st.info("Results without missing values:")

				# Create crosstab
				crosstab = pd.crosstab(df_clean[categorical_var1], df_clean[categorical_var2], values=df_clean[numeric_var],
									   aggfunc='mean')

				# Display crosstab
				st.write("Crosstab:")
				st.dataframe(crosstab)

				# Display table with number of cases
				st.write("Number of Cases:")
				cases_table = pd.crosstab(df_clean[categorical_var1], df_clean[categorical_var2])
				st.dataframe(cases_table)

				# Display heatmap using matplotlib and seaborn
				st.write("Heatmap:")
				plt.figure(figsize=(10, 6))
				sns.heatmap(crosstab, annot=True, cmap=farbwahl, fmt=".1f")
				st.pyplot()

				# Display bar chart for mean values and number of cases of categorical_var1
				st.write(f"Mean Values and Number of Cases for {categorical_var1}:")
				grouped_var1 = df_clean.groupby(categorical_var1)
				mean_values_var1 = grouped_var1[numeric_var].mean()
				count_var1 = grouped_var1[numeric_var].count()

				fig, ax1 = plt.subplots(figsize=(10, 6))
				ax2 = ax1.twinx()

				ax1.bar(mean_values_var1.index, mean_values_var1, alpha=0.7, label=f"Mean {numeric_var}")
				ax2.plot(count_var1.index, count_var1, color='red', marker='o', label=f"Number of Cases",
						 linestyle='dashed')

				ax1.set_xlabel(categorical_var1)
				ax1.set_ylabel(f"Mean {numeric_var}", color='blue')
				ax2.set_ylabel("Number of Cases", color='red')
				plt.title(f"Mean {numeric_var} and Number of Cases for {categorical_var1}")
				ax1.tick_params(axis='y', labelcolor='blue')
				ax2.tick_params(axis='y', labelcolor='red')

				# Add annotations on top of the bars
				for i, (mean_value, count) in enumerate(zip(mean_values_var1, count_var1)):
					ax1.text(i, mean_value, f"{mean_value:.1f}\n{count}", ha='center', va='bottom', color='black')

				st.pyplot()

				# Display bar chart for mean values and number of cases of categorical_var2
				st.write(f"Mean Values and Number of Cases for {categorical_var2}:")
				grouped_var2 = df_clean.groupby(categorical_var2)
				mean_values_var2 = grouped_var2[numeric_var].mean()
				count_var2 = grouped_var2[numeric_var].count()

				fig, ax1 = plt.subplots(figsize=(10, 6))
				ax2 = ax1.twinx()

				ax1.bar(mean_values_var2.index, mean_values_var2, alpha=0.7, label=f"Mean {numeric_var}")
				ax2.plot(count_var2.index, count_var2, color='red', marker='o', label=f"Number of Cases",
						 linestyle='dashed')

				ax1.set_xlabel(categorical_var2)
				ax1.set_ylabel(f"Mean {numeric_var}", color='blue')
				ax2.set_ylabel("Number of Cases", color='red')
				plt.title(f"Mean {numeric_var} and Number of Cases for {categorical_var2}")
				ax1.tick_params(axis='y', labelcolor='blue')
				ax2.tick_params(axis='y', labelcolor='red')

				# Add annotations on top of the bars
				for i, (mean_value, count) in enumerate(zip(mean_values_var2, count_var2)):
					ax1.text(i, mean_value, f"{mean_value:.1f}\n{count}", ha='center', va='bottom', color='black')

				st.pyplot()


