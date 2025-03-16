import sklearn

from utils import load_train_test_datasets
from static import PREDICTOR_COLUMNS, TARGET_COLUMN, TARGET_CLASS_DICT
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# usage of Copilot
# how to calculate accuracy of train and test data
def run_decision_tree(train_x, train_y, test_x, test_y):
    # Run a classification by constructing a decision tree
    # Please set the random_state to 5963
    clf = DecisionTreeClassifier(random_state=5963)
    clf.fit(train_x, train_y)

    train_pred = clf.predict(train_x)
    test_pred = clf.predict(test_x)

    train_accuracy = metrics.accuracy_score(train_y, train_pred)
    test_accuracy = metrics.accuracy_score(test_y, test_pred)

    print(f'[Decision Tree] Use {PREDICTOR_COLUMNS} to predict: {TARGET_COLUMN}')

    print(f'Training Accuracy: {train_accuracy:}')
    print(f'Test Accuracy: {test_accuracy:}')
    ef
    run_random_forest(train_x, train_y, test_x, test_y):
    # Run a classification by constructing a random forest
    # Please set the random_state to 5963
    clf = RandomForestClassifier(random_state=5963)
    clf.fit(train_x, train_y)

    train_pred = clf.predict(train_x)
    test_pred = clf.predict(test_x)

    train_accuracy = metrics.accuracy_score(train_y, train_pred)
    test_accuracy = metrics.accuracy_score(test_y, test_pred)

    print(f'[Random Forest] Use {PREDICTOR_COLUMNS} to predict: {TARGET_COLUMN}')

    print(f'Training Accuracy: {train_accuracy:}')
    print(f'Test Accuracy: {test_accuracy:}')

if __name__ == '__main__':
    _train_x, _train_y, _test_x, _test_y = load_train_test_datasets()
    run_decision_tree(_train_x, _train_y, _test_x, _test_y)
    run_random_forest(_train_x, _train_y, _test_x, _test_y)
