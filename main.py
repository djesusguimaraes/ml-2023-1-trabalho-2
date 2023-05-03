import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

def z_score_outliers(df, threshold=3):
    z_scores = np.abs((df - df.mean()) / df.std())

    outliers = df[(z_scores > threshold).any(axis=1)]
    
    return outliers
    
def clean_data(data):
    data_cp = data.copy()

    data_cp = data_cp[data_cp.columns[data_cp.isna().sum()/data_cp.shape[0] < 0.9]]

    for y in data_cp.columns:
        if data_cp[y].dtype == 'object': 
            lbl = LabelEncoder()
            lbl.fit(list(data_cp[y].values))
            data_cp[y] = lbl.transform(list(data_cp[y].values))
            
    outliers = z_score_outliers(data_cp)
    
    for outlier in outliers:
        data_cp.loc[outlier] = np.nan
    
    data_cp = data_cp.fillna(data_cp.median())
    
    return data_cp

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
    
    data_copy = clean_data(data)
    
    print(data_copy['SARS-Cov-2 exam result'].value_counts(normalize=True))
    print('\n')
    
    x = data_copy.drop(columns=['SARS-Cov-2 exam result'])
    x = data_copy[['Mean corpuscular volume (MCV)', 'Patient age quantile']]
    y = data_copy['SARS-Cov-2 exam result']
    
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
    