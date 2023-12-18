import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Read the dataset and inspect the head to understand column names and existence
df_rf = pd.read_csv('diabetes.csv')
df_rf.head()

# Perform basic null checks to decide whether imputation or drop is required
df_rf.isnull().sum()

# Split the data into features (X) and target values (y)
y = df_rf['Outcome']
X = df_rf.drop('Outcome', axis=1)

# Perform a train-test split with a 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Create a RandomForestClassifier with prescribed hyperparameters
rf_clf = RandomForestClassifier()

# Train and fit the RandomForestClassifier
rf_clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = rf_clf.predict(X_test)

# Report accuracy, precision, recall, and f1_score
rf_y_true = y_test
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision Score:", precision)
print("Recall Score: ", recall)
print("F1 Score: ", f1_score)

# Plot ROC curve and report AUC score
rf_probs = rf_clf.predict_proba(X_test)[:, 1]
rf_auc = 0

def plot_roc(rf_y_true, rf_probs):
    rf_fpr, rf_tpr, _ = metrics.roc_curve(rf_y_true, rf_probs)
    rf_auc_val = metrics.auc(rf_fpr, rf_tpr)
    
    print('AUC=%0.2f' % rf_auc_val)
    
    plt.plot(rf_fpr, rf_tpr, label='AUC=%0.2f' % rf_auc_val, color='darkorange')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return rf_auc_val

rf_auc = plot_roc(rf_y_true, rf_probs)

# Report the best k value and corresponding accuracy using StratifiedKFold Cross Validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

max_acc, max_k = 0, 0

for k in range(2, 5):
    skfold = StratifiedKFold(n_splits=k, random_state=100, shuffle=True)
    results_skfold_acc = (cross_val_score(rf_clf, X, y, cv=skfold)).mean() * 100.0
    
    if results_skfold_acc > max_acc:
        max_acc = results_skfold_acc
        max_k = k

    print("Accuracy: %.2f%%" % (results_skfold_acc))

best_accuracy = max_acc
best_k_fold = max_k

print(best_accuracy, best_k_fold)
