"""
Data Analysis and Visualization Script
This script loads the Diabetes dataset, performs basic analysis, and creates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_dataset():
    """
    Load the Diabetes dataset from sklearn.
    Returns a pandas DataFrame.
    """
    print("Loading Diabetes dataset from sklearn.")
    diabetes = load_diabetes()
    df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    df['target'] = df['target']  # Target represents disease progression
    
    return df

def explore_dataset(df):
    """Explore the dataset structure and handle missing values."""
    print("=== Dataset Overview ===")
    print(f"Dataset shape: {df.shape}")
    
    print("\n=== First 5 rows ===")
    print(df.head())
    
    print("\n=== Dataset Information ===")
    print(df.info())
    
    print("\n=== Missing Values ===")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    # Handle missing values if any
    if missing_values.any():
        print("\n=== Handling Missing Values ===")
        # For numerical columns, fill with mean
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ['int64', 'float64']:
                    fill_value = df[column].mean()
                    df[column].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in {column} with mean: {fill_value:.2f}")
    
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    return df

def basic_analysis(df):
    """Perform basic statistical analysis on the dataset."""
    print("=== Basic Statistics ===")
    print(df.describe())
    
    print("\n=== Correlation with Target ===")
    correlations = df.corr()['target'].sort_values(ascending=False)
    print(correlations)
    
    print("\n=== Interesting Findings ===")
    # Find the feature with strongest positive correlation
    positive_corr = correlations[correlations > 0].drop('target')
    if len(positive_corr) > 0:
        strongest_positive = positive_corr.idxmax()
        print(f"Feature with strongest positive correlation to target: {strongest_positive} ({positive_corr.max():.3f})")
    
    # Find the feature with strongest negative correlation
    negative_corr = correlations[correlations < 0]
    if len(negative_corr) > 0:
        strongest_negative = negative_corr.idxmin()
        print(f"Feature with strongest negative correlation to target: {strongest_negative} ({negative_corr.min():.3f})")
    
    return df

def create_visualizations(df):
    """Create various visualizations from the dataset."""
    print("\n=== Creating Visualizations ===")
    
    plt.figure(figsize=(15, 10))
    
    # 1. Line chart showing target values
    plt.subplot(2, 3, 1)
    plt.plot(df.index[:50], df['target'][:50], marker='o', color='red', label='Disease Progression')
    plt.title('Diabetes Disease Progression (First 50 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.legend()
    
    # 2. Histogram of target variable
    plt.subplot(2, 3, 2)
    df['target'].hist(bins=20, color='lightcoral', edgecolor='black')
    plt.title('Distribution of Disease Progression')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    
    # 3. Correlation heatmap
    plt.subplot(2, 3, 3)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    
    # 4. Scatter plot of strongest correlated feature vs target
    correlations = df.corr()['target'].drop('target')
    if len(correlations) > 0:
        strongest_feature = correlations.abs().idxmax()
        plt.subplot(2, 3, 4)
        plt.scatter(df[strongest_feature], df['target'], alpha=0.6, color='blue')
        plt.title(f'{strongest_feature} vs Disease Progression')
        plt.xlabel(strongest_feature)
        plt.ylabel('Target Value')
    
    # 5. Box plot of target by age groups (if age feature exists)
    if 'age' in df.columns:
        plt.subplot(2, 3, 5)
        # Create age groups
        df['age_group'] = pd.qcut(df['age'], q=4, labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
        sns.boxplot(x='age_group', y='target', data=df)
        plt.title('Disease Progression by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Target Value')
        plt.xticks(rotation=45)
    
    # 6. Scatter plot matrix for top 3 features
    plt.subplot(2, 3, 6)
    top_features = correlations.abs().nlargest(3).index.tolist()
    if len(top_features) >= 2:
        plt.scatter(df[top_features[0]], df[top_features[1]], 
                   c=df['target'], cmap='viridis', alpha=0.7)
        plt.colorbar(label='Disease Progression')
        plt.title(f'{top_features[0]} vs {top_features[1]}')
        plt.xlabel(top_features[0])
        plt.ylabel(top_features[1])
    
    plt.tight_layout()
    plt.savefig('diabetes_visualizations.png')
    plt.show()
    
    # Additional visualization: Pairplot of top correlated features
    try:
        top_corr_features = correlations.abs().nlargest(4).index.tolist() + ['target']
        if len(top_corr_features) > 1:
            sns.pairplot(df[top_corr_features], diag_kind='hist')
            plt.suptitle('Pairplot of Top Correlated Features with Target', y=1.02)
            plt.savefig('diabetes_pairplot.png')
            plt.show()
    except Exception as e:
        print(f"Could not create pairplot: {e}")

def main():
    """Main function to execute the data analysis pipeline."""
    print("Diabetes Data Analysis and Visualization Script")
    print("==============================================")
    
    # Load the dataset
    df = load_dataset()
    
    # Explore the dataset
    df = explore_dataset(df)
    
    # Perform basic analysis
    df = basic_analysis(df)
    
    # Create visualizations
    create_visualizations(df)
    
    print("\nAnalysis complete. Visualizations have been saved.")

if __name__ == "__main__":
    main()
