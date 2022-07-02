import os
import pandas as pd
from joblib import load


def binary_map(feature):
    return feature.map({'Yes': 1, 'No': 0})


def feature_transformations(data):

    replace_list = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']
    data[replace_list] = data[replace_list].replace(
        'No internet service', 'No')

    data.drop(['customerID', 'gender', 'PhoneService', 'MultipleLines', 'TotalCharges',
               'StreamingTV', 'StreamingMovies'], axis=1, inplace=True)

    # Encoding other binary category
    binary_list = ['Partner', 'Dependents', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'PaperlessBilling']
    data[binary_list] = data[binary_list].apply(binary_map)

    data['Contract'] = data['Contract'].replace(
        {'One year': 'Year', 'Two year': 'Year'})

    data = pd.get_dummies(data, drop_first=True)
    return data


def test_data(data):

    sc = load(os.path.join('models', 'scaler.pkl'))
    random_forest_classifier = load(os.path.join(
        'models', 'random_forest_classifier.pkl'))

    data[['tenure', 'MonthlyCharges']] = sc.transform(
        data[['tenure', 'MonthlyCharges']])

    selected_features = ['tenure', 'PaperlessBilling', 'MonthlyCharges',
                         'InternetService_Fiber optic', 'Contract_Year',
                         'PaymentMethod_Electronic check']

    selected_data = data[selected_features]
    y_pred = random_forest_classifier.predict(selected_data)

    return y_pred


def get_test_results():

    data = pd.read_csv(os.path.join('data', 'test_data.csv'))

    data_transformed = feature_transformations(data.copy())

    results = test_data(data_transformed)

    data_transformed['Churn'] = results
    data_transformed.to_csv(os.path.join('data', 'results.csv'), index=False)
    print("Results Saved")


if __name__ == "__main__":
    get_test_results()
