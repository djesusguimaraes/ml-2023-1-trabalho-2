import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

def clean_data(data):

    print(f'Original Shape: \t{data.shape}\n')

    data = data.dropna(axis=1, how='all')

    print(f'Not NaN Columns Shape: \t{data.shape}\n')

    data = data.dropna(thresh=10)

    print(f'Not NaN Shape: \t{data.shape}\n')

    string_columns = data.select_dtypes(include=['object']).columns

    print(f'Object Columns count: {len(string_columns)}\n')

    for column in string_columns:
        data[column] = data[column].astype('category').cat.codes

    data = data.fillna(data.median())
    
    print(f'Filled NaN with Median: \t{data.shape}\n')
    
    return data

def print_result(model, X_test, y_test):
    prds = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, prds).ravel()
    print(f'tn {tn}, fp {fp}, fn {fn}, tp {tp}', '\n\n',
        'Accuracy:', (accuracy_score(y_test, prds)), '\n\n',
        'Classification Report:\n', (classification_report(y_test, prds)))
    pass

def decision_tree(x, y):
    model_name = 'Decision Tree'
    print(f'\n{model_name}\n')
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=101)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    print_result(model, X_test, y_test)
    
    plot_boundary(model, X_train, y_train, model_name)

def knn(x, y):
    model_name = 'KNN'
    print(f'\n{model_name}\n')
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1012)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    print_result(model, X_test, y_test)
    
    plot_boundary(model, X_train, y_train, model_name)

def naive_bayes(x, y):
    model_name = 'Naive Bayes'
    print(f'\n{model_name}\n')
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=32)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    print_result(model, X_test, y_test)

    plot_boundary(model, X_train, y_train, model_name)

def plot_boundary(model, X_train, y_train, model_name):
    disp = DecisionBoundaryDisplay.from_estimator(
        model, X_train, response_method="predict",
        xlabel="Mean platelet volume", ylabel="Platelets",
        alpha=0.5)
    
    disp.ax_.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k")
    plt.title(f'{model_name}')
    plt.axis('tight')

if __name__ == "__main__":
    data = pd.read_excel('dataset.xlsx')
    
    data = clean_data(data)
    
    print(data['SARS-Cov-2 exam result'].value_counts(normalize=True))
    print('\n')
    
    x = data.drop(columns=['SARS-Cov-2 exam result'])
    x = data[['Mean corpuscular volume (MCV)', 'Patient age quantile']]
    y = data['SARS-Cov-2 exam result']
    
    scaler =StandardScaler().fit(x)
    x = scaler.transform(x)

    x_tree = x.copy()
    y_tree = y.copy()

    x_nb = x.copy()
    y_nb = y.copy()
    
    x_knn = x.copy()
    y_knn = y.copy()
    
    decision_tree(x_tree, y_tree)
    naive_bayes(x_nb, y_nb)
    knn(x_knn, y_knn)
    
    plt.show()
    