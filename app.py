#2024.04.22.11 Kleine Formalitäten Welch-t test2 + exceldownload
#2024.03.20.12 Welch's T-Test 2 von Kategorien hinzugefügt
#2024.03.19.13 Welch's T-Test hinzugefügt
#2024.03.03.05.15 Streumasse hinzugefügt
#2024.03.03.05.24 Sample hinzugefügt
#2024.04.22 kleine Fomralitäten
#2025.04.29 einfache korrelationsberechnung hinzugefügt
#2025.06.256 added filtering by categories to welch's t-test1
import streamlit as st

#Streumasse
#für konfiidenzintervalle
from scipy import stats



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


#Für Sample
import math
from scipy.stats import norm


# für Excel-Export-Funktionen
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb


#Welch T-Tests
from scipy.stats import ttest_ind




st.set_page_config(
    page_title="Simple Statistictools",
    page_icon="🧊",
    #layout="wide",
    #initial_sidebar_state="expanded",
)

#---Option Menu -------------------

option = option_menu(
	menu_title="Simple Statistics",
	options=["Dispersion", "Anova","Heatmap","Sample", "Z-Test", "Welch T-Test 1", "Welch T-Test 2"],
	icons=["0-square", "2-circle","3-circle","bounding-box","2-circle","arrow-left-right","arrow-down-up"], #https://icons.getbootstrap.com/
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



########################### Streumasse / Dispersion ###############################################################################################################################################

if option =="Dispersion":

	st.set_option('deprecation.showPyplotGlobalUse', False)

	st.title('Scatter measures')
	st.info("Upload an Excel file containing, for example, survey data. The variable names must be in the first row, with the data for the variables in the subsequent rows. After uploading, select the numerical variables of interest and obtain a simple summary of the most common statistical scatter measures.")

	# Datei hochladen
	uploaded_file = st.file_uploader("Datei hochladen (Excel)", type=["xlsx", "xls"])

	if uploaded_file is not None:
		# Daten einlesen
		data = pd.read_excel(uploaded_file)

		# Nur numerische Spalten auswählen
		numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()

		# Auswahl der Variablen
		selected_columns = st.multiselect('Wähle numerische Variablen aus', numerical_columns)

		if st.checkbox("Confirm selection"):


			# Statistische Kennzahlen und Konfidenzintervalle berechnen

			statistics = {}

			missingsHandling = st.selectbox("Handling of missings?", ('Do nothing', 'Drop missings', 'Replace with Zero'))

			if st.checkbox("Start calculations!"):

				# zuerst einfache, schnelle übersichtstabelle
				st.write("Simple description of the selected variables:")
				st.write(data[selected_columns].describe())
				st.write("Correlations")
				st.write(data[selected_columns].corr())

				for column in selected_columns:

					if missingsHandling == "Do nothing":
						values = data[column]
					if missingsHandling == "Drop missings":
						values = data[column].dropna()  # Drop missing values within the loop
					if missingsHandling == "Replace with Zero":
						values = data[column].fillna(0)
					
					mean = values.mean()
					median = values.median()
					variance = values.var()
					std_dev = values.std()
					sem = stats.sem(values)
					ci_low, ci_high = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=sem)
					varKoeff = std_dev / mean #thomas testet Variationskoeffizient..

					statistics[column] = {
						'Min': values.min(),
						'Max': values.max(),
						'Spannweite': values.max() - values.min(),
						'Mittelwert': mean,
						'Median': median,
						'Fälle': len(values),
						'Varianz': variance,
						'Standardabweichung': std_dev,
						'Stichprobenfehler (ddof=1)': sem,
						'CI_low (95%)': ci_low, 
							'CI_high (95%)': ci_high,
						'Mean_CI_low': mean - ci_low,
						'Mean_CI_high': ci_high - mean,
						'Variationskoeffizient': varKoeff
					}

				# Ergebnisse je Variable anzeigen
				# st.subheader('Statistische Kennzahlen:')
				# for column, values in statistics.items():
				#    st.write(f'**{column}**:')
				# st.write(statistics)

				# Zusammenfassung in Tabelle anzeigen
				summary_df = pd.DataFrame(statistics).T
				st.subheader('Ergebnistabelle:')
				st.write(summary_df)



				codeBuchExpander = st.expander('Ergebnistabelle nach Excel exportierten')
				with codeBuchExpander:
					# speicherZeitpunkt = pd.to_datetime('today')
					st.write("")
					if len(summary_df) > 0:
						def to_excel(dfCodebuch):
							output = BytesIO()
							writer = pd.ExcelWriter(output, engine='xlsxwriter')
							summary_df.to_excel(writer, index=True, sheet_name='Sheet1')
							workbook = writer.book
							worksheet = writer.sheets['Sheet1']
							format1 = workbook.add_format({'num_format': '0.00'})
							worksheet.set_column('A:A', None, format1)
							writer.close()
							processed_data = output.getvalue()
							return processed_data


						df_xlsx = to_excel(summary_df)
						st.download_button(label='📥 Tabelle in Excel abspeichern?',
										data=df_xlsx,
										file_name='Streumasse' + '.xlsx')





				streuungScharts = st.button("Streu-Charts anzeigen")
				if streuungScharts:
					# Verteilung plotten
					st.subheader('Verteilung der Variablen:')
					for column in selected_columns:
						plt.figure(figsize=(8, 6))
						sns.histplot(data[column], kde=True)
						plt.title(f'Distribution of {column}')
						plt.xlabel(column)
						plt.ylabel('Frequency')
						st.pyplot()


				st.divider()









			# Button für Beschreibungen
			if st.button('Erklärungen anzeigen'):
				st.subheader('Erklärungen:')
				st.write(
					'**Median:** Der Median ist der mittlere Wert einer sortierten Datenreihe. Wenn die Daten in aufsteigender Reihenfolge sortiert sind, ist der Median der Wert in der Mitte.')
				st.write(
					'**Varianz:** Die Varianz ist ein Maß für die Streuung der Datenpunkte um den Mittelwert. Eine hohe Varianz deutet auf eine große Streuung hin, während eine niedrige Varianz auf eine geringere Streuung hinweist.')
				st.write(
					'**Standardabweichung:** Die Standardabweichung ist die Quadratwurzel der Varianz. Sie gibt an, wie weit die einzelnen Datenpunkte im Durchschnitt vom Mittelwert entfernt sind.')
				st.write(
					'**Stichprobenfehler:** Der Stichprobenfehler ist ein Maß für die Unsicherheit der Schätzung des Mittelwerts in einer Stichprobe im Vergleich zur Gesamtmenge. Er wird kleiner, wenn die Stichprobe größer ist.')
				st.write(
					'**CI (95%):** Das Konfidenzintervall (CI) gibt an, in welchem Bereich wir mit 95%iger Sicherheit den wahren Parameter erwarten können. In diesem Fall bezieht es sich auf den Mittelwert.')
				st.write(
					'**Der empirische Variationskoeffizient:** wird gebildet als Quotient aus empirischer Standardabweichung (ev ist hier aber nicht die empirische Standardabweichung verwendet worden)..und arithmetischem Mittel')













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
		"Upload an Excel-File, choose two categorical variables and one numeric variable and  explore the differences between the mean values of the category combinations")

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


		st.info("Table with mean values in the cells")
		st.dataframe(crosstab)

		if len(crosstab)>1:
			def to_excel(crosstab):
				output = BytesIO()
				writer = pd.ExcelWriter(output, engine='xlsxwriter')
				crosstab.to_excel(writer, index=True, sheet_name='Sheet1')
				workbook = writer.book
				worksheet = writer.sheets['Sheet1']
				format1 = workbook.add_format({'num_format': '0.00'})
				worksheet.set_column('A:A', None, format1)
				writer.close()
				processed_data = output.getvalue()
				return processed_data


			df_xlsx = to_excel(crosstab)
			st.download_button(label='📥 Download Crosstab as Excel?',
							   data=df_xlsx,
							   file_name='CrossTab' + '.xlsx')










		# Display heatmap using matplotlib and seaborn
		st.write("")
		st.subheader("Heatmap")
		st.write("Mean values for ")
		st.info(numeric_var)

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
		st.subheader("Anova of " + categorical_var1)
		categories = df_clean[categorical_var1].unique()
		data_by_category = [df_clean[numeric_var][df_clean[categorical_var1] == category] for category in categories]

		#st.write("data_by_category: ",data_by_category)

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
		st.subheader("Anova of " + categorical_var2)
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



################################### Sample Size #######################################
if option =="Sample":
	st.title('Vertrauensbereich von Stichproben')

	population_size = st.number_input('Bevölkerungsgröße/Grundgesamtheit eingeben:', min_value=1, step=1, value=7000000)
	sample_size = st.number_input('Stichprobenumfang - Anzahl Befragte:', min_value=1, step=1, value=500)
	percentage = st.slider('Prozentsatz des Messwerts eingeben: (z.B. 30% Nein)', 0.0, 100.0, 50.0)
	confidence_level = st.slider('Vertrauensniveau auswählen:', 0.1, 0.99, 0.95, 0.01)

	percentageWert = percentage / 100

	# st.write("percentageWert",percentageWert)

	if st.button('Berechnen'):
		standard_deviation = math.sqrt((percentageWert * (1 - percentageWert)) / sample_size)
		# st.write("standard_deviation", standard_deviation)

		margin_of_error = norm.ppf((1 + confidence_level) / 2) * (standard_deviation / math.sqrt(sample_size)) * 10
		# st.write("margin_of_error", margin_of_error)

		lower_bound = max(0, percentageWert - margin_of_error) * 100
		# st.write("lower_bound", lower_bound)

		upper_bound = min(100, percentageWert + margin_of_error) * 100
		# st.write("upper_bound", upper_bound)

		st.success(f'Der Vertrauensbereich beträgt {percentage}% +/- {round((upper_bound - lower_bound) / 2, 2)}%')
		st.info(f'Untere Grenze: {round(lower_bound, 2)}%, Obere Grenze: {round(upper_bound, 2)}%')







########################### Z-TEST ###############################################################################################################################################

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

if option =="Z-Test":

	st.title("Proportions Z-Test")
	st.subheader("Determine whether two population means are different")
	st.info("Upload one or two Excelfiles, and explore if the values of two columns with metric values differ significantly. This method requires equal sample sizes, use Welsch T-Test if sample sizes differ")

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
			if st.checkbox("Replace Missing Values with 0"):
    				# Replace missing values with 0 in the specified column
    				data1[column1].fillna(0, inplace=True)
			st.write(data1[column1])
			# st.write(data1[column1].mean())
			st.write(data1[column1].describe())
		# successes1 = int(data1[column1].mean() * len(data1))
		# st.write("successes: ",successes1)

		with dataColumns2:
			st.write("Column2 - from Survey 2")
			if st.checkbox("Replace Missing Values with 0 "):
    				# Replace missing values with 0 in the specified column
    				data2[column2].fillna(0, inplace=True)			
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

			thomasTestetZtestNomal = z_stat, p_value = ztest(data.column1, data.column2)
			st.write("thomasTestetZtestNomal:", thomasTestetZtestNomal)

			thomasPValue = twoSampleZTest[1]
			st.write("twoSampleZTest: ", twoSampleZTest)

			st.metric(label="P-Value", value=thomasPValue.round(2))

			# Interpretation
			alpha = 0.05

			if thomasPValue < alpha:
				st.success("The difference between the proportions is statistically significant.")
			if thomasPValue >= alpha:
				st.warning("The difference between the proportions is not statistically significant.")
			else:
				st.warning("Significance was not calculated, you can try Welch's T-Test")


	
			
			
			resultscol1, resultscols2 = st.columns(2)

			with resultscol1:
				st.write("Survey 1")
				m1 = data.column1.mean()
				s1 = data.column1.std()
				len1 = len(data1)
				se1 = np.std(data.column1, ddof=1) / np.sqrt(len(data.column1)) #alternative berechnung mit berücksichtigung der Fallzahl
				st.write("Mean1: ", m1)
				st.write("Std 1: ", s1)
				st.write("Std 1 - Alternativ: ", se1)
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
				se2 = np.std(data.column2, ddof=1) / np.sqrt(len(data.column2)) #alternative berechnung mit berücksichtigung der Fallzahl
				st.write("Mean2: ", m2)
				st.write("Std 2: ", s2)
				st.write("Std 2 - Alternativ: ", se2)
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


			ci1 = (confInterval1_left, confInterval1_right)
			ci2 = (confInterval2_left, confInterval2_right)

			# Plotting
			fig3, ax3 = plt.subplots()
			
			# Plot means
			ax3.bar([0.5, 1.5], [m1, m2], yerr=[[m1 - ci1[0], ci1[1] - m1], [m1 - ci2[0], ci2[1] - m2]], capsize=10)
			ax3.set_xticks([0.5, 1.5])
			ax3.set_xticklabels(['Sample 1', 'Sample 2'])
			ax3.set_ylabel('Mean Value')
			ax3.set_title('Comparison of Sample Means with Confidence Intervals')
			
			# Show plot
			st.pyplot(fig3)
	


























































############################### Welch T-Test 1 #############################################

if option =="Welch T-Test 1":

	st.title("Comparison of Sample means with Welch's T-Test")
	st.info("Compare mean values of variables from two different samples. Also works if samples are of different size.")

	with st.expander("Info about Welch's T-Test"):
		st.markdown("""
		
		1. Assumptions:

		The data in each sample are independent.
		The populations from which the samples are drawn are normally distributed, or the sample sizes are large enough for the Central Limit Theorem to apply.
		
		2. Usefulness:

		It's useful when the variances of the two populations being compared are unequal.
		It's robust to unequal sample sizes.
		
		3. Advantages:

		Robustness: Welch's t-test does not assume equal variances, making it robust when the assumption of equal variances is violated.
		Accuracy: It provides accurate results even when sample sizes are unequal, maintaining Type I error rates close to the nominal level.
		Flexibility: It can be applied to a wide range of scenarios where comparing the means of two independent samples is necessary.
		
		Comparison to other tests:

		Student's t-test: Welch's t-test is more robust when the assumption of equal variances is violated or when sample sizes are unequal.
			  
		Z-test: Welch's t-test is preferred when dealing with small sample sizes (typically n < 30) or when the population standard deviations are unknown.
			  
		In summary, Welch's t-test offers a flexible and robust method for comparing means of two independent samples, especially when the assumptions of other tests like the Student's t-test are not met. It's a valuable tool in statistical analysis, particularly in situations where data variability and sample size discrepancies exist.
						
					
					""")

	# Upload files
	uploaded_file1 = st.file_uploader("Upload first Excel file")
	uploaded_file2 = st.file_uploader("Upload second Excel file")

	if uploaded_file1 and uploaded_file2:
		data1 = pd.read_excel(uploaded_file1)
		data2 = pd.read_excel(uploaded_file2)

		st.write("### First DataFrame")
		st.write(data1.head())

		st.write("### Second DataFrame")
		st.write(data2.head())

		# Select variables
		numeric_variables1 = data1.select_dtypes(include='number').columns.tolist()
		numeric_variables2 = data2.select_dtypes(include='number').columns.tolist()

		cat_columns1 = data1.select_dtypes(include=['object', 'category']).columns.tolist()
		cat_columns2 = data2.select_dtypes(include=['object', 'category']).columns.tolist()


		selected_var1 = st.selectbox("Select variable from first DataFrame", numeric_variables1)
		selected_var2 = st.selectbox("Select variable from second DataFrame", numeric_variables2)

		st.write("")
		st.divider()
		st.info("Optional filtering by Category Variables:")

	    # Filter for DataFrame 1
	    cat_columns1 = data1.select_dtypes(include=['object', 'category']).columns.tolist()
	    if cat_columns1:
	        filter_col1 = st.selectbox("Filter first DataFrame by", ["None"] + cat_columns1, key="cat1")
	        if filter_col1 != "None":
	            filter_vals1 = data1[filter_col1].dropna().unique().tolist()
	            selected_vals1 = st.multiselect(f"Select values in '{filter_col1}'", filter_vals1, default=filter_vals1)
	            data1 = data1[data1[filter_col1].isin(selected_vals1)]
	
	    # Filter for DataFrame 2
	    cat_columns2 = data2.select_dtypes(include=['object', 'category']).columns.tolist()
	    if cat_columns2:
	        filter_col2 = st.selectbox("Filter second DataFrame by", ["None"] + cat_columns2, key="cat2")
	        if filter_col2 != "None":
	            filter_vals2 = data2[filter_col2].dropna().unique().tolist()
	            selected_vals2 = st.multiselect(f"Select values in '{filter_col2}'", filter_vals2, default=filter_vals2)
	            data2 = data2[data2[filter_col2].isin(selected_vals2)]




		st.divider()
		st.write("")

		if st.checkbox("Replace Missing Values with 0"):
			# Replace missing values with 0 in the specified column
			data1[selected_var1].fillna(0, inplace=True)
			data2[selected_var2].fillna(0, inplace=True)

		if st.button("Perform Welch's t-test"):
			t_statistic, p_value = ttest_ind(data1[selected_var1], data2[selected_var2], equal_var=False)

			st.write(f"Welch's t-test results:")
			st.write(f"T-statistic: {t_statistic}")
			st.write(f"P-value: {p_value}")

			mean1 = data1[selected_var1].mean()
			st.write("Mean 1: ",mean1)

			mean2 = data2[selected_var2].mean()
			st.write("Mean 2: ",mean2)

			if p_value < 0.05:
				st.success("The means of the selected variables are significantly different.")
			else:
				st.warning("The means of the selected variables are not significantly different.")







############################### Welch T-Test 2 #############################################

if option =="Welch T-Test 2":
	st.title("Comparison of Sample category means with Welch's T-Test")
	st.info("Compare mean values of categorical variables from two different samples. Also works if samples are of different size.")

	st.write("")

	# Information about the app functionality

	st.info("""
		This app allows you to compare survey data from two different datasets. Here's how it works:

		1. **Upload Data**: Upload two Excel files containing survey data.
		2. **Column Selection**: Select columns containing non-numerical data (e.g., sex, age groups, language) from both datasets.
		3. **Numerical Variable Selection**: Choose numerical variables that occur in both datasets.
		4. **Comparison**: The app performs statistical tests (Welch's t-test) to compare the mean values of the selected numerical variables for each category defined by the non-numerical columns.
		5. **Display Results**: View the results of the statistical tests in a table format, indicating the mean difference between the surveys and whether the difference is statistically significant.

	""")

	# File Upload
	st.header("Upload Data")
	uploaded_file1 = st.file_uploader("Upload first survey data (Excel file)", type="xlsx")
	uploaded_file2 = st.file_uploader("Upload second survey data (Excel file)", type="xlsx")

	if uploaded_file1 and uploaded_file2:
		# Load Data
		df1 = pd.read_excel(uploaded_file1)
		df2 = pd.read_excel(uploaded_file2)
		
		# Column Selection
		non_numerical_columns1 = st.multiselect("Select non-numerical columns, like categories (Men/Women) from first survey", df1.columns)

		#st.write(df1[non_numerical_columns1])

		non_numerical_columns2 = st.multiselect("Select non-numerical columns, like categories (Men/Women) from second survey", df2.columns)
		
		# Numerical Variable Selection
		numerical_variables = st.multiselect("Select numerical variables common in both surveys", df1.columns.intersection(df2.columns))
		
		casesColumns1, casesColumns2 = st.columns(2)

		casesColumns1.info("Cases in first survey: " + str(len(df1)))
		casesColumns2.info("Cases in second survey: " + str(len(df2)))


		# Perform Comparison
		st.write("")

		#showComparedData = st.checkbox("Show compared data after analysis")

		if st.button("Compare!"):
			comparisons = []
			for variable in numerical_variables:
				for column in non_numerical_columns1:
					for value in df1[column].unique():
						data1 = df1[df1[column] == value][variable]
						data2 = df2[df2[column] == value][variable]
						
						if len(data1) > 0 and len(data2) > 0:
							result = ttest_ind(data1, data2, equal_var=False)
							#st.write(f"Comparison for {variable} when {column} is {value}: p-value = {result.pvalue}")
							mean_diff = data2.mean() - data1.mean()
							significant = "Yes" if result.pvalue < 0.05 else "No"
							comparisons.append({
							"Variable": variable,
							"Category": column,
							"Value": value,
							"Cases1" : len(data1!=""),
							"Cases2" : len(data2!=""),
							"Mean1" : data1.mean(),
							"Mean2" : data2.mean(),
							"Mean Difference": mean_diff,
							"P-Value" : result.pvalue,
							"Significant Difference": significant
						})
				# Display Results
			


			if comparisons:
				st.header("Comparison Results")
				comparison_df = pd.DataFrame(comparisons)
				st.write(comparison_df)

				if len(comparison_df)>1:
					def to_excel(comparison_df):
						output = BytesIO()
						writer = pd.ExcelWriter(output, engine='xlsxwriter')
						comparison_df.to_excel(writer, index=True, sheet_name='Sheet1')
						workbook = writer.book
						worksheet = writer.sheets['Sheet1']
						format1 = workbook.add_format({'num_format': '0.00'})
						worksheet.set_column('A:A', None, format1)
						writer.close()
						processed_data = output.getvalue()
						return processed_data

				df_xlsx = to_excel(comparison_df)
				st.download_button(label='📥 Download Table with Results as Excel?',
								data=df_xlsx,
								file_name='Results Welch T-Test' + '.xlsx')

			else:
				st.write("No comparisons found.")






			#showDataColumns1, showDataColumns2 = st.columns(2)
			#if showComparedData:
			#	showDataColumns1.dataframe(data1)
			#	showDataColumns2.dataframe(data2)
