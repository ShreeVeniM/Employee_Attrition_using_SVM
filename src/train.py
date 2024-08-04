from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(x_train, y_train):
    try:
        model = LogisticRegression()
        model.fit(x_train, y_train)
        return model
    except Exception as e:
        print(f"Error training Logistic Regression model: {e}")
        return None

def train_svm(x_train, y_train, kernel='linear', degree=3):
    try:
        model = SVC(kernel=kernel, degree=degree)
        model.fit(x_train, y_train)
        return model
    except Exception as e:
        print(f"Error training SVM model: {e}")
        return None
