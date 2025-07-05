import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import optuna
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset,DataQualityPreset,ClassificationPreset,TargetDriftPreset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris


## Step 1: Load and prepare data
iris=load_iris()
X=iris.data
y=iris.target

mlflow.set_tracking_uri("http://127.0.0.1:5000")

### Step 2: Split into reference and current data
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=11)

##Hyperparameter tuning using Optuna
def objective(trial):
    max_depth=trial.suggest_int("max_depth",2,60)
    n_estimators=trial.suggest_int("n_estimators",10,200)
    
    model=RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    score=cross_val_score(model,X_train,Y_train,cv=5,scoring='accuracy').mean()
    return score

mlflow.set_experiment("Iris project using ML")

with mlflow.start_run():
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
    study.optimize(objective,n_trials=25)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_score",study.best_value)

    best_model=RandomForestClassifier(**study.best_params,random_state=11,oob_score=True)
    best_model.fit(X_train,Y_train)
    y_pred=best_model.predict(X_test)
    accuracy=accuracy_score(y_pred,Y_test)

    mlflow.log_metric("accuracy",accuracy)

    mlflow.sklearn.log_model(best_model,"RandomForestmodel")

    #Data drift report
    reference_data=pd.DataFrame(X_train,columns=iris.feature_names)
    reference_data['target']=Y_train
    reference_data["prediction"] = best_model.predict(X_train)

    current_data=pd.DataFrame(X_test,columns=iris.feature_names)
    current_data['target']=Y_test
    current_data["prediction"] = best_model.predict(X_test)
    #current_data['prediction']=y_pred

    report=Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset(),
        ClassificationPreset()
    ])

    report.run(reference_data=reference_data,current_data=current_data)
    report.save_html("iris_drift_report.html")
    mlflow.log_artifact("iris_drift_report.html")
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(best_model,"RandomForestmodel")

    print(accuracy)
    print(study.best_params)
    print(study.best_trials)
    print(study.best_trial.value)


    
  

