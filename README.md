# ANOVA test to determine if there are significant differences between groups defined by a categorical variable.

            This app is for researchers or analysts interested in understanding how ratings vary across different categories and identifying specific variables that show significant differences.
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
