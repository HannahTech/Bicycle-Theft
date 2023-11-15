'''
Group 04 :
Abubakr, Eman 
Delamasa, Matthew 
Khajehpour, Hengameh 
Moshfegh, Peyman 
Soliman, Mohamed   
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

######################## Load & check the data #################################
#1.	Load the data into a pandas dataframe 
data_bicycle = pd.read_csv("/Users/heny/Desktop/Group4/Bicycle_Thefts.csv")

#number of rows and columns
print (data_bicycle.shape)
#Carryout some initial investigations
#Check the names of columns 
print(data_bicycle.columns.values)

# types of columns
print("Data types:\n",data_bicycle.dtypes)

#Check the missing values.
#showing the column name and the number of missing values per column.
print("Missing values:\n",data_bicycle.isnull().sum())

#Check the statistics of the numeric fields 
print(data_bicycle.describe())

#Checking the categorical values
data_bicycle["Status"].unique()
data_bicycle["Status"].value_counts()

data_bicycle["City"].unique()
data_bicycle["City"].value_counts()

data_bicycle["Division"].unique()
data_bicycle["Division"].value_counts()

data_bicycle["Bike_Type"].unique()
data_bicycle["Bike_Type"].value_counts()

data_bicycle["Occurrence_Month"].unique()
data_bicycle["Occurrence_Month"].value_counts()

data_bicycle["Occurrence_DayOfWeek"].unique()
data_bicycle["Occurrence_DayOfWeek"].value_counts()

data_bicycle["Bike_Colour"].unique()
data_bicycle["Bike_Colour"].value_counts()

data_bicycle["NeighbourhoodName"].unique()
data_bicycle["NeighbourhoodName"].value_counts()

data_bicycle["Location_Type"].unique()
data_bicycle["Location_Type"].value_counts()

data_bicycle["Premises_Type"].unique()
data_bicycle["Premises_Type"].value_counts()

data_bicycle["Primary_Offence"].unique()
data_bicycle["Primary_Offence"].value_counts()

data_bicycle["Occurrence_Date"].unique()
data_bicycle["Occurrence_Date"].value_counts()

data_bicycle["Report_Date"].unique()
data_bicycle["Report_Date"].value_counts()

data_bicycle["Report_Month"].unique()
data_bicycle["Report_Month"].value_counts()

data_bicycle["Report_DayOfWeek"].unique()
data_bicycle["Report_DayOfWeek"].value_counts()

data_bicycle["Bike_Make"].unique()
data_bicycle["Bike_Make"].value_counts()

data_bicycle["Longitude"].unique()
data_bicycle["Longitude"].value_counts()

data_bicycle["Latitude"].unique()
data_bicycle["Latitude"].value_counts()

################ cleaning the data ######################
#Drop the ID columns as they are very unique and the column which has a lot of missing values
data_bicycle=data_bicycle.drop(["OBJECTID","OBJECTID_1","event_unique_id","Hood_ID"], axis=1)

#Drop the column which has a lot of missing values
data_bicycle = data_bicycle.drop("Bike_Model", axis=1)

#Drop columns with unuseful data
data_bicycle=data_bicycle.drop(["Occurrence_Date","Occurrence_Year","Occurrence_Month",
                                "Occurrence_DayOfMonth","Report_Date","Report_Year","Report_Month",
                                "Report_DayOfWeek","Report_DayOfMonth","Report_DayOfYear","Report_Hour",
                                "NeighbourhoodName","Bike_Make","Bike_Colour","Location_Type","City"], axis=1)
##data_bicycle=data_bicycle.drop(["Longitude","Latitude"], axis=1)

df_missing = data_bicycle
df_missing['Status'].value_counts()

df_missing['Status'] = [1 if b=='RECOVERED' else 0 for b in df_missing.Status]

# Separate majority and minority classes
df_majority = df_missing[df_missing.Status==0]
df_minority = df_missing[df_missing.Status==1]

# Upsample minority class
from sklearn.utils import resample
df_minority_upsampled = resample(df_minority, replace=True, n_samples=28658, random_state=42)
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.Status.value_counts()

data_bicycle = df_upsampled



##Initial plots -histograms of each attribute 
data_bicycle.hist(bins=50, figsize=(20,15))

#Generate a heatmap showing the relationship between the all the columns
plt.figure(figsize=(20,20))
sns.heatmap(data_bicycle.corr(), annot=True, square=True, cmap='coolwarm')
plt.show()

#Since there is geographical information (latitude and longitude), it is a good idea to create a scatterplot of all districts to visualize the data
# set alpha (for better transparency)
#data_bicycle.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.1)

#Visualize more correlations

#data_bicycle.plot(kind="scatter", x="X", y="Longitude", alpha=0.1)
#data_bicycle.plot(kind="scatter", x="Y", y="Latitude", alpha=0.1)

#Visualizing Correlation map of Toronto
data_bicycle.plot(kind="scatter", x="X", y="Latitude", alpha=0.1)
data_bicycle.plot(kind="scatter", x="Status", y="Cost_of_Bike", alpha=0.1)
data_bicycle.plot(kind="scatter", x="Status", y="Bike_Speed", alpha=0.1)
data_bicycle.plot(kind="scatter", x="Bike_Speed", y="Cost_of_Bike", alpha=0.1)
data_bicycle.plot(kind="scatter", x="Status", y="Latitude", alpha=0.1)

#Majority of bikes stolen are 15-25 km/h speed bikes
data_bicycle.plot(kind="scatter", x="X", y="Latitude", alpha=0.4,
                  c="Bike_Speed", cmap=plt.get_cmap("jet"), colorbar=True, vmax=40)

# vmax of $5000 to indicate cases of grand theft
data_bicycle.plot(kind="scatter", x="X", y="Latitude", alpha=0.4,
                  c="Cost_of_Bike", cmap=plt.get_cmap("jet"), colorbar=True, vmax=2000)

data_bicycle=data_bicycle.drop(["Longitude","Latitude", "X","Y"], axis=1)

######################## Data Modelling #################################
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

#Deleting Unknown Values
#data_bicycle = data_bicycle[data_bicycle["Status"].str.contains("UNKNOWN") == False].copy()

#Creating preprocessing pipelines for numeric and categorical data
for col in ["Occurrence_DayOfYear","Occurrence_Hour","Bike_Speed"]:
    data_bicycle[col] = data_bicycle[col].astype('float64')
    
numeric_features =  ["Occurrence_DayOfYear","Occurrence_Hour","Bike_Speed","Cost_of_Bike"]

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median", missing_values=np.nan)),("scaler", StandardScaler())])   

#Pipelines for categorical features
for col in ["Primary_Offence","Occurrence_DayOfWeek","Division","Premises_Type","Bike_Type"]:
    data_bicycle[col] = data_bicycle[col].astype('category')

categorical_features = ["Primary_Offence","Occurrence_DayOfWeek","Division","Premises_Type","Bike_Type"]

categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),('onehot',OneHotEncoder(handle_unknown='ignore'))])

#Creating appropriate pipeline
#le = preprocessing.LabelEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

sample_incomplete_rows = data_bicycle[data_bicycle.isnull().any(axis=1)]
print (len(sample_incomplete_rows))

############## LogisticRegression is just a placeholder, we will use gridsearch for better algorithms later ##############
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(solver='lbfgs', max_iter=1000,class_weight="balanced"))]
)

# because of running speed dataset  size is reduced
#data_bicycle = data_bicycle.sample(frac =.01)
#data_bicycle.shape

X = data_bicycle.drop("Status", 1)
# y = pd.DataFrame(le.fit_transform(data_bicycle[["Status"]]))
y = data_bicycle["Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.to_csv('X_train.csv',index=False)
X_test.to_csv('X_test.csv',index=False)

y_train.to_csv('y_train.csv',index=False)
y_test.to_csv('y_test.csv',index=False)

# cm = confusion_matrix(y_train, ypredtrain)
######################## End of data modelling #################################
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn import metrics
import joblib


pipe_lr = Pipeline([("preprocessor", preprocessor),
                    ('scl', StandardScaler(with_mean=False)),
                    ('LR', LogisticRegression(random_state=42))])
pipe_dt = Pipeline([("preprocessor", preprocessor),
                    ('scl', StandardScaler(with_mean=False)),
                    ('DT',DecisionTreeClassifier(random_state=42))])
pipe_rf = Pipeline([("preprocessor", preprocessor),
                    ('scl', StandardScaler(with_mean=False)),
                    ('RF',RandomForestClassifier(random_state=42))])
pipe_knn = Pipeline([("preprocessor", preprocessor),
                     ('scl', StandardScaler(with_mean=False)),
                    ('KNN', KNeighborsClassifier())])
pipe_svm = Pipeline([("preprocessor", preprocessor),
                     ('scl', StandardScaler(with_mean=False)),
                     ('SVM', SVC(random_state=42))])
pipe_lsvc = Pipeline([("preprocessor", preprocessor),
                     ('scl', StandardScaler(with_mean=False)),
                     ('LSVC', LinearSVC(dual=False, random_state=42))])

# for the big data it's better to use linear svc instead of SVM
param_range = [1, 2]
param_range_fl = [1.0, 0.5, 0.1]

lr_param_grid = [{'LR__penalty': ['l1', 'l2'],
                   'LR__C': param_range_fl,
                   'LR__solver': ['liblinear']}]
dt_param_grid = [{'DT__criterion': ['gini', 'entropy'],
                   'DT__min_samples_leaf': param_range,
                   'DT__max_depth': param_range,
                   'DT__min_samples_split': param_range[1:]}]
rf_param_grid = [{'RF__min_samples_leaf': param_range,
                   'RF__max_depth': param_range,
                   'RF__min_samples_split': param_range[1:]}]
knn_param_grid = [{'KNN__n_neighbors': param_range,
                   'KNN__weights': ['uniform', 'distance'],
                   'KNN__metric': ['euclidean', 'manhattan']}]
svm_param_grid = [{'SVM__kernel': ['linear'], 
                    'SVM__C': param_range}]
lsvc_param_grid = [{'LSVC__tol': [0.01], 
                    'LSVC__C':np.arange(0.01,100,10)}]


lr_grid_search = GridSearchCV(estimator=pipe_lr,
        param_grid=lr_param_grid,
        scoring='accuracy',
        cv=10)
dt_grid_search = GridSearchCV(estimator=pipe_dt,
        param_grid=dt_param_grid,
        scoring='accuracy',
        cv=10)
rf_grid_search = GridSearchCV(estimator=pipe_rf,
        param_grid=rf_param_grid,
        scoring='accuracy',
        cv=10)
knn_grid_search = GridSearchCV(estimator=pipe_knn,
        param_grid=knn_param_grid,
        scoring='accuracy',
        cv=10)
svm_grid_search = GridSearchCV(estimator=pipe_svm,
        param_grid=svm_param_grid,
        scoring='accuracy',
        cv=5)
lsvc_grid_search = GridSearchCV(estimator=pipe_lsvc,
        param_grid=lsvc_param_grid,
        scoring='accuracy',
        cv=5, return_train_score=True)


# List of pipelines for ease of iteration
#grids = [lr_grid_search, dt_grid_search, rf_grid_search, knn_grid_search, lsvc_grid_search, svm_grid_search]
grids = [lr_grid_search, dt_grid_search, rf_grid_search, knn_grid_search, lsvc_grid_search]

# Dictionary of pipelines and classifier types for ease of reference
grid_dict ={0: 'Logistic Regression', 1: 'Decision Trees', 
             2: 'Random Forest', 3: 'K-Nearest Neighbors', 
             4: 'Linear SVC', 5: 'Support Vector Machines'}

# Fit the grid search objects
print('Performing model optimizations...')
best_acc = 0.0
best_clf = 0
best_model = ''
for i, model in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[i])	
    # Fit grid search	
    model.fit(X_train, y_train)
    # Best params
    print('%s best params: %s' % (grid_dict[i], model.best_params_))
    # Best training data accuracy
    print('%s best training accuracy: %.3f' % (grid_dict[i], model.best_score_))
    # Predict on test data with best params
    y_pred = model.predict(X_test)
    print('%s test Report: \n %s' % (grid_dict[i], classification_report(y_test, y_pred)))
    # Plot confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative','Positive']
    plt.title('%s confusion Matrix \n' % grid_dict[i])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames, rotation=90)
    s = [['TN','FP'], ['FN', 'TP']]
    for j in range(2):
        for k in range(2):
            plt.text(k,j, str(s[j][k])+" = "+str(cm[j][k]))
    plt.show()
    # Test data accuracy of model with best params
    print('\n%s test set accuracy score for best params: %.3f ' % (grid_dict[i], accuracy_score(y_test, y_pred)))
    # Plot ROC model
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    plt.plot(fpr,tpr)
    plt.title(grid_dict[i])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    #save dump files
    dump_file = '%s_model.pkl' % grid_dict[i]
    joblib.dump(model, dump_file, compress=1)
    print('\n Saved %s grid search pipeline to file: %s' % (grid_dict[i], dump_file))
    # Track best (highest test accuracy) model
    if accuracy_score(y_test, y_pred) > best_acc:
        best_acc = accuracy_score(y_test, y_pred)
        best_model = model
        best_clf = i
        
print('\n\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])

# Save best grid search pipeline to file
dump_file = 'best_model_pipeline_All_Dataset.pkl'
joblib.dump(best_model, dump_file, compress=1)
print('\nSaved %s grid search pipeline to file: %s' % (grid_dict[best_clf], dump_file))


######################## Part 05 #################################
from flask import Flask, request, jsonify
import traceback
import pandas as pd
import pickle
import joblib
import sys
from os import path
from sklearn import metrics
from flask_cors import CORS

project_folder = r'/Users/heny/Desktop/Group4'
models = {
         "Best_Model":"best_model_pipeline_All_Dataset.pkl"
         ,"Random_Forest": "Random forest_model.pkl"
         ,"K-nearest neighbors": "K-nearest neighbors_model.pkl"
         ,"Decision_Tree": "Decision Trees_model.pkl"
         ,"Logistic_Regression": "logistic regression_model.pkl"
         ,"Linear SVM": "linear svc_model.pkl"
         }

# data frames 

X_train_df = pd.read_csv(path.join(project_folder,"x_train.csv"))
y_train_df = pd.read_csv(path.join(project_folder,"y_train.csv"))
X_test_df = pd.read_csv(path.join(project_folder,"x_test.csv"))
y_test_df = pd.read_csv(path.join(project_folder,"y_test.csv"))

# Your API definition
app = Flask(__name__)
CORS(app)

@app.route("/predict/<model_name>", methods=['GET','POST']) #use decorator pattern for the route
def predict(model_name):
    if loaded_model:
        try:
            json_ = request.json
            print('JSON: \n', json_)
            query = pd.DataFrame(json_, columns=model_columns)
            prediction = list(loaded_model[model_name].predict(query))
            print(f'Returning prediction with {model_name} model:')
            print('prediction=', prediction)
            res = jsonify({"prediction": str(prediction)})
            res.headers.add('Access-Control-Allow-Origin', '*')
            return res
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model available.')
    
@app.route("/scores/<model_name>", methods=['GET','POST']) #use decorator pattern for the route
def scores(model_name):
    if loaded_model:
        try:
            y_pred = loaded_model[model_name].predict(X_test_df)
            print(f'Returning scores for {model_name}:')
            accuracy = metrics.accuracy_score(y_test_df, y_pred)
            precision = metrics.precision_score(y_test_df, y_pred)
            recall = metrics.recall_score(y_test_df, y_pred)
            f1 = metrics.f1_score(y_test_df, y_pred)
            print(f'accuracy={accuracy}  precision={precision}  recall={recall}  f1={f1}')
            res = jsonify({"accuracy": accuracy,
                            "precision": precision,
                            "recall":recall,
                            "f1": f1
                           })
            res.headers.add('Access-Control-Allow-Origin', '*')
            return res
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model available.')
        

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
        
    # load all models:
    loaded_model = {}
    for model_name in (models):
        loaded_model[model_name] = joblib.load(path.join(project_folder, models[model_name]))
        print(f'Model {model_name} loaded')
        
    model_columns = ['Primary_Offence',
           'Occurrence_DayOfWeek', 'Occurrence_DayOfYear', 'Occurrence_Hour',
           'Division','Premises_Type', 'Bike_Type',
           'Bike_Speed', 'Cost_of_Bike']
    
    
    app.run(port=port, debug=True)