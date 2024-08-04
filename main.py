import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_logistic_regression, train_svm
from src.evaluate import metrics_score
from src.utils import plot_histograms, plot_categorical_distributions, plot_correlation_heatmap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info('Starting main function')
    try:
        file_path = 'src/dataset/HR_Employee_Attrition.xlsx'
        
        # Load and preprocess data
        logging.info('Loading data from HR_Employee_Attrition.xlsx')
        df = load_data(file_path)
        if df is None:
            logging.error('Failed to load data')
            return

        logging.info('Preprocessing data')
        X_scaled, Y, num_cols, cat_cols = preprocess_data(df)
        if X_scaled is None or Y is None:
            logging.error('Failed to preprocess data')
            return

        logging.info('Plotting histograms')
        plot_histograms(df, num_cols)

        logging.info('Plotting categorical distributions')
        plot_categorical_distributions(df, cat_cols)

        logging.info('Plotting correlation heatmap')
        plot_correlation_heatmap(df, num_cols)

        # Split the data
        logging.info('Splitting data into training and testing sets')
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
        
        # Train and evaluate Logistic Regression model
        logging.info('Training Logistic Regression model')
        lg_model = train_logistic_regression(x_train, y_train)
        if lg_model is not None:
            logging.info('Evaluating Logistic Regression model')
            y_pred_train = lg_model.predict(x_train)
            y_pred_test = lg_model.predict(x_test)
            logging.info('Logistic Regression Model:')
            metrics_score(y_train, y_pred_train, 'logistic_regression_train')
            metrics_score(y_test, y_pred_test, 'logistic_regression_test')
        
        # Train and evaluate SVM model with linear kernel
        logging.info('Training SVM model with linear kernel')
        svm_model_linear = train_svm(x_train, y_train, kernel='linear')
        if svm_model_linear is not None:
            logging.info('Evaluating SVM model (Linear Kernel)')
            y_pred_train_svm = svm_model_linear.predict(x_train)
            y_pred_test_svm = svm_model_linear.predict(x_test)
            logging.info('SVM Model (Linear Kernel):')
            metrics_score(y_train, y_pred_train_svm, 'svm_linear_train')
            metrics_score(y_test, y_pred_test_svm, 'svm_linear_test')
        
        # Train and evaluate SVM model with rbf kernel
        logging.info('Training SVM model with rbf kernel')
        svm_model_rbf = train_svm(x_train, y_train, kernel='rbf')
        if svm_model_rbf is not None:
            logging.info('Evaluating SVM model (RBF Kernel)')
            y_pred_train_svm_rbf = svm_model_rbf.predict(x_train)
            y_pred_test_svm_rbf = svm_model_rbf.predict(x_test)
            logging.info('SVM Model (RBF Kernel):')
            metrics_score(y_train, y_pred_train_svm_rbf, 'svm_rbf_train')
            metrics_score(y_test, y_pred_test_svm_rbf, 'svm_rbf_test')
        
        # Train and evaluate SVM model with polynomial kernel
        logging.info('Training SVM model with polynomial kernel')
        svm_model_poly = train_svm(x_train, y_train, kernel='poly', degree=3)
        if svm_model_poly is not None:
            logging.info('Evaluating SVM model (Polynomial Kernel)')
            y_pred_train_svm_poly = svm_model_poly.predict(x_train)
            y_pred_test_svm_poly = svm_model_poly.predict(x_test)
            logging.info('SVM Model (Polynomial Kernel):')
            metrics_score(y_train, y_pred_train_svm_poly, 'svm_poly_train')
            metrics_score(y_test, y_pred_test_svm_poly, 'svm_poly_test')

    except Exception as e:
        logging.error(f'Error occurred: {e}')

if __name__ == "__main__":
    main()
