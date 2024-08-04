import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_histograms(df, num_cols):
    try:
        df[num_cols].hist(figsize=(14, 14))
        os.makedirs('charts', exist_ok=True)
        plt.savefig('charts/histograms.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting histograms: {e}")

def plot_categorical_distributions(df, cat_cols):
    try:
        os.makedirs('charts', exist_ok=True)
        for i in cat_cols:
            (pd.crosstab(df[i], df['Attrition'], normalize='index') * 100).plot(kind='bar', figsize=(8, 4), stacked=True)
            plt.ylabel('Percentage Attrition %')
            plt.savefig(f'charts/{i}_distribution.png', dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error plotting categorical distributions: {e}")

def plot_correlation_heatmap(df, num_cols):
    try:
        plt.figure(figsize=(15, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt='0.2f', cmap='YlGnBu')
        os.makedirs('charts', exist_ok=True)
        plt.savefig('charts/correlation_heatmap.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting correlation heatmap: {e}")
