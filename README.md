# Diabetes Dataset Analysis

## Overview
This project performs comprehensive analysis and visualization of the Diabetes dataset from scikit-learn. The dataset contains physiological measurements from 442 diabetes patients and a quantitative measure of disease progression one year after baseline.

## Dataset Information
- **Source**: scikit-learn toy datasets
- **Samples**: 442 patients
- **Features**: 10 baseline variables
- **Target**: Quantitative measure of disease progression

### Features:
1. age: Age in years
2. sex: Gender (male/female)
3. bmi: Body mass index
4. bp: Average blood pressure
5. s1: Total serum cholesterol
6. s2: Low-density lipoproteins
7. s3: High-density lipoproteins
8. s4: Total cholesterol / HDL ratio
9. s5: Possibly triglycerides level
10. s6: Blood sugar level

## Project Tasks

### Task 1: Load and Explore the Dataset
- Load the Diabetes dataset from scikit-learn
- Display the first few rows to inspect the data
- Check dataset structure, data types, and missing values
- Handle any missing values by filling with appropriate measures

### Task 2: Basic Data Analysis
- Compute basic statistics for numerical columns
- Analyze correlations between features and target variable
- Identify strongest positive and negative correlations with disease progression
- Discover interesting patterns in the data

### Task 3: Data Visualization
Create multiple visualizations to understand the data:

1. **Line Chart**: Disease progression trends across the first 50 samples
2. **Histogram**: Distribution of disease progression values
3. **Heatmap**: Correlation matrix between all features
4. **Scatter Plot**: Strongest correlated feature vs. disease progression
5. **Box Plot**: Disease progression by age groups
6. **Scatter Matrix**: Pairplot of top correlated features with target

## Technical Implementation

### Libraries Used
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Basic plotting and visualization
- seaborn: Advanced statistical visualizations
- scikit-learn: Dataset loading

### Key Functions
1. `load_dataset()`: Loads the Diabetes dataset from scikit-learn
2. `explore_dataset()`: Examines dataset structure and handles missing values
3. `basic_analysis()`: Performs statistical analysis and correlation studies
4. `create_visualizations()`: Generates all required plots and charts

## How to Run
1. Ensure all required libraries are installed
2. Execute the Python script
3. View generated visualizations (saved as PNG files)
4. Review console output for statistical insights

## Findings
The analysis reveals:
- Key factors most strongly correlated with diabetes progression
- Distribution patterns of the disease progression metric
- Relationships between different physiological measurements
- How disease progression varies across different age groups

## Files Generated
- `diabetes_visualizations.png`: Composite visualization with multiple plots
- `diabetes_pairplot.png`: Pairplot of top correlated features

## Customization
The code includes error handling and can be adapted for:
- Different visualization styles
- Additional statistical tests
- Alternative correlation analyses
- Expanded feature engineering

This analysis provides a solid foundation for understanding the Diabetes dataset and can serve as a template for similar healthcare data analysis projects.



# PYTHONPLPWeek7
