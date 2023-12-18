# Import necessary libraries
# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import matplotlib.pyplot as plt

# Load and explore the dataset
# %%
pima_df = pd.read_csv('diabetes.csv')
pima_df.head()
pima_df.describe()
pima_df.info()
shape = pima_df.shape
print(shape)

# Check for missing values
# %%
pima_df.isnull().sum()

# Extract features (X) and target variable (Y)
# %%
# print(pima_df.columns)
X = pima_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y = pima_df[['Outcome']]

# Split the dataset into training and testing sets
# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Train a Decision Tree Classifier
# %%
clf = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
clf.fit(X_train, y_train)

# Predictions and Evaluation
# %%
dtree_y_pred = clf.predict(X_test)
dtree_y_true = y_test

# Calculate and print accuracy, precision, recall, and F1-score
accuracy = accuracy_score(dtree_y_true, dtree_y_pred)
precision = precision_score(dtree_y_true, dtree_y_pred, average='weighted')
recall = recall_score(dtree_y_true, dtree_y_pred, average='weighted')
f1_score = f1_score(dtree_y_true, dtree_y_pred, average='weighted')
print("Accuracy: ", accuracy)
print("Precision Score: ", precision)
print("Recall Score: ", recall)
print("F1 Score: ", f1_score)

# Plot ROC curve
# %%
dtree_probs = clf.predict_proba(X_test)[:, 1]
dtree_auc = plot_roc(dtree_y_true, dtree_probs)

# Evaluate using Stratified K-Fold Cross Validation
# %%
max_acc = 0
max_k = 0

for k in range(2, 11):
    skfold = StratifiedKFold(n_splits=k, random_state=100, shuffle=True)
    results_skfold_acc = (cross_val_score(clf, X, Y.values.ravel(), cv=skfold)).mean() * 100.0
    
    if results_skfold_acc > max_acc:
        max_acc = results_skfold_acc
        max_k = k
    print("Accuracy: %.2f%%" % results_skfold_acc)

best_accuracy = max_acc
best_k_fold = max_k

print(best_accuracy, best_k_fold)
