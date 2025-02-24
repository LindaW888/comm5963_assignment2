import sklearn

from utils import load_train_test_datasets
from static import PREDICTOR_COLUMNS, TARGET_COLUMN, TARGET_CLASS_DICT


def run_decision_tree(train_x, train_y, test_x, test_y):
    # Use df_iris and run a classification by constructing a decision tree
    # Please set the random_state to 5963
    print(f'[Decision Tree] Use {PREDICTOR_COLUMNS} to predict: {TARGET_COLUMN}')

def run_random_forest(train_x, train_y, test_x, test_y):
    # Use df_iris and run a classification by constructing a random forest
    # Please set the random_state to 5963
    print(f'[Random Forest] Use {PREDICTOR_COLUMNS} to predict: {TARGET_COLUMN}')



if __name__ == '__main__':
    _train_x, _train_y, _test_x, _test_y = load_train_test_datasets()
    run_decision_tree(_train_x, _train_y, _test_x, _test_y)
    run_random_forest(_train_x, _train_y, _test_x, _test_y)