import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def metrics_score(actual, predicted, model_name):
    try:
        report = classification_report(actual, predicted)
        print(report)
        
        cm = confusion_matrix(actual, predicted)
        plt.figure(figsize=(8, 5))
        sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Create charts folder if it doesn't exist
        os.makedirs('charts', exist_ok=True)
        plt.savefig(f'charts/{model_name}_confusion_matrix.png', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error in metrics_score: {e}")
