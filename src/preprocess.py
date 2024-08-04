import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    try:
        df = df.drop(['EmployeeNumber','Over18','StandardHours'], axis=1)
        
        num_cols = ['DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'TotalWorkingYears',
                    'YearsAtCompany', 'NumCompaniesWorked', 'HourlyRate', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'TrainingTimesLastYear']

        cat_cols = ['Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 'EducationField', 'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance',
                    'StockOptionLevel', 'Gender', 'PerformanceRating', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus', 'RelationshipSatisfaction']

        dict_OverTime = {'Yes': 1, 'No': 0}
        dict_attrition = {'Yes': 1, 'No': 0}

        df['OverTime'] = df.OverTime.map(dict_OverTime)
        df['Attrition'] = df.Attrition.map(dict_attrition)
        
        to_get_dummies_for = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus']
        df = pd.get_dummies(data=df, columns=to_get_dummies_for, drop_first=True)

        Y = df.Attrition
        X = df.drop(columns=['Attrition'])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, Y, num_cols, cat_cols
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None, None, None