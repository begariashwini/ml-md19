#STEP 1:import libraries
import pandas a pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection imort train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import(
f1_score,
accuracy_score,
classification_rteport,
confusion_matrix
)
#STEP 1:Load dataset
data = load_breast_cancer()
X = pd.DataFrame(
  data.data,
  columns=data.feature_names
)
y = pd.Series(data.target,name="target")
print("Dataset shape:",X.shape)
print("\nFirst 5 rows:")
print(X.head())
print("\nTarget values:")
print(y.value_counter())
#STEP 3:Split dataset
X_train,X_test,y_train,y_test = train_test_split(
  X,
  y,
  test_size=0.20,
  random-state=42,
stratify=y
)
print("\nTraining samples:",X+train.shape[0])
print("Testing samples:",X_test.shape[0])
#STEP 4:Define classification algorithms
models = {
  "Logistic Regression:"LogisticRegression(
    max_iter=5000
  ),
  "KNN":KNeighborsClssifier(),
"Decision Tree":DecisionTreeClassifier(
  random_state=42
),
"Random Forest":RandomForestClassifier(
  random_sate=42
),
"SVM":SVC()
}
#STEP 5:Define hyperparameter grids
param_grids={
  "Logistic Regression":{
    "model__c":[0.1,1,10],
    "model__solver":["liblinear"]
  },
  "KNN":{
    "model__n_neighbors":[3,5,7,9],
    "model__eights":["uniform","distance"]
  },
  "Decision Tree":{
    "model__max_depth":[None,3,5,10],
    "model__min_samples_split":[2,5,10]
  },
  "Random Forest":{
    "model__n_estimators":[100,200],
    "model__max_depth":[None,5,10],
    "model__min_samples_split":[2,5]
  },
  "SVM":{
    "model__c";[0.1,1,10],
"model__kernel":["linear","rbg"],
"model__gamma":["scale","auto"]
}
}
#STEP 6:Run GridSearchCV for every algorithm
results = []
best_models={}
for model_names,model in models.items():
  print("\n"+"="*60)
  print("Running:",model_name)
  print("="*60)
  print("Running:",model_name)
  print("="*60)
  pipeline = Pipeline([
  ("scaler",StandardScaler()),
  ("model",model)
  ])
  grid_search = GridSearchCV(
  estimator=pipeline,
  param_grid=param_grids[model_name],
  scoring="f1",
  cv=5,n_jobs=-1
  )
  grid_search.fit(X_train,y_train)
  best_models[model_name]=grid_search.best_estimator_results.append({
  "model":model_name,
  "Best F1 Score":grid_search.best_score_,
  "Best Parameters":grid_search.best_params_,
  })
  print("Best F1 Score:",
        round(grid_search.bnest_score_,4)
        print("Best Parameters:",
              grid_search.best_params_)
        #STEP 7:Compare all models
        results_df = pd.DataFrame(results)
  results_df = results_df.sort_values(
    by="Best F1 Score",
    ascending=False
  )
  print("\n"+"="*60)
  print("MODEL COMPARISON")
  print("="*60)
  print(
    resukts_df[
    ["Model","Best F1 Score"]
    ].to_string(index=False)
  )
  #STEP 8:Select the best model
best_model_name = results_dfiloc[0]["Model"]
best_model = best_models[best_model_name]
best_cv_f1 = results_df.iloc[0]["Best F1_score"]
print("\Best_model based on CV F1_score:")
print(best_model_name)
print("\nBest CV F1-score:")
print(round(best_cv_f1,4))
#STEP 9:Evaluate on unseen test data
y_pred = best_model.predict(X_test)
print("\n"+"="*60)
print("FINAL MODEL EVALUATION")
print("="*60)
print("FINAL MODEL EVALUATION")
print("="*60)
print("Accuracy:",accuracy_score(y_test,y_prted))
print("F1 Score:",f1_score(y_test.y_pred))
print("\nClassification Report:")
print(
  calssifiction_report(
    y_test,
    y_pred,
    target_names=data.target_names
  )
)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))
#STEP 10:Save the best model
joblib.dump(best_model,"besr_model.joblib")
print("\nBest model saved successfully!")
print("Model:",best_model_name)
  
