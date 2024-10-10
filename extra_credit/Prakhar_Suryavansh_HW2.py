import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc


'''
General Instructions:

1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
the library specified in question instruction available. Importing additional library will result in 
compilation error and you will lose marks.

2. You are expected to fill in the skeleton code precisely as per provided. On top of skeleton code given,
you may write whatever deemed necessary to complete the assignment. For example, you may define additional 
default arguments, class parameters, or dependency functions (you probably have to either for making the 
skeleton work or providing necessary stats in the report).

3. Some initial steps or definition are given, aiming to help you getting started. As long as you follow 
the input/output format, you are free to change them as you see fit.

4. Your code should be free of compilation errors.
'''


'''
Problem A-1: Data Preprocessing and EDA
'''
class DataLoader:
    '''
    This class will be used to load the data and perform initial data processing. Fill in functions.
    You are allowed to add any additional functions which you may need to check the data. This class 
    will be tested on the pre-built enviornment with only numpy and pandas available.
    '''

    def __init__(self, data_root: str, random_state: int):
        '''
        Inialize the DataLoader class with the data_root path.
        Load train/test data as pd.DataFrame, store as needed and initialize other variables.
        All dataset should save as pd.DataFrame.
        '''
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.data = pd.read_csv(f"{data_root}/data_train-1.csv")
        self.data_test = pd.read_csv(f"{data_root}/data_test-1.csv")

        self.data_train = None
        self.data_valid = None

    def data_split(self) -> None:
        '''
        You are asked to split the training data into train/valid datasets on the ratio of 80/20. 
        Add the split datasets to self.data_train, self.data_valid. Both of the split should still be pd.DataFrame.
        '''
        # Shuffle the data before splitting
        self.data = self.data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        shuffled_data = self.data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Split 80/20
        split_idx = int(len(shuffled_data) * 0.8)

        # Split the data
        self.data_train = shuffled_data.iloc[:split_idx]
        self.data_valid = shuffled_data.iloc[split_idx:]

    def data_prep(self) -> None:
        '''
        You are asked to drop any rows with missing values in the data.
        '''
        self.data.dropna(inplace=True)
        self.data_test.dropna(inplace=True)

        # Drop the 'Unnamed: 0' index column and 'Loan_ID' column as they are meaningless for our classification model
        self.data.drop(columns=['Unnamed: 0', 'Loan_ID'], inplace=True)
        self.data_test.drop(columns=['Unnamed: 0', 'Loan_ID'], inplace=True)


    def extract_features_and_label(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        '''
        This function will be called multiple times to extract features and labels from train/valid/test 
        data.
        
        Expected return:
            X_data: np.ndarray of shape (n_samples, n_features) - Extracted features
            y_data: np.ndarray of shape (n_samples,) - Extracted labels
        '''
        X_data = data.drop(columns=['Loan_Status'], errors='ignore').values
        y_data = data.get('Loan_Status')
        if y_data is not None:
            y_data = y_data.values
        return X_data, y_data
    
    def encode(self, data):
        # Identify non-numerical columns
        non_numerical_columns = data.select_dtypes(include=['object']).columns

        # Apply Label Encoding for each non-numerical column
        label_encoders = {}
        for column in non_numerical_columns:
            unique_values = sorted(data[column].unique())
            # Create a dictionary mapping each unique value to an integer
            encoding_map = {value: idx for idx, value in enumerate(unique_values)}            
            # Apply the mapping
            data[column] = data[column].map(encoding_map)

'''
Porblem A-2: Classification Tree Inplementation
'''
class ClassificationTree:
    '''
    You are asked to implement a simple classification tree from scratch. This class will be tested on the
    pre-built enviornment with only numpy and pandas available.

    You may add more variables and functions to this class as you see fit.
    '''
    class Node:
        '''
        A data structure to represent a node in the tree.
        '''
        def __init__(self, split=None, value=None, left=None, right=None, prediction=None):
            '''
            split: int - Feature index to split on
            value: float - The threshold value for the split
            left: Node - Left child node
            right: Node - Right child node
            prediction: (any) - Prediction value if the node is a leaf
            '''
            self.split = split
            self.value = value
            self.left = left
            self.right = right
            self.prediction = prediction 

        def is_leaf(self):
            return self.prediction is not None

    def __init__(self, random_state: int):
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.tree_root = None

    def split_crit(self, y: np.ndarray) -> float:
        '''
        Implement the impurity measure of your choice here. Return the impurity value.
        '''
        classes = np.unique(y)
        n_instances = len(y)
        gini = 1.0
        for cls in classes:
            p_cls = np.sum(y == cls) / n_instances
            gini -= p_cls ** 2
        return gini

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth=0, max_depth=3, min_samples_split=2) -> Node:
        '''
        Implement the tree building algorithm here. You can recursivly call this function to build the 
        tree. After building the tree, store the root node in self.tree_root.
        '''
        n_samples, n_features = X.shape
        
        # Stopping condition: If all labels are the same or max depth is reached
        if len(np.unique(y)) == 1 or depth >= max_depth or n_samples < min_samples_split:
            leaf_prediction = np.argmax(np.bincount(y))
            return self.Node(prediction=leaf_prediction)
        
        # Get the best feature and split
        split_idx, split_val = self.search_best_split(X, y)
        if split_idx is None:
            leaf_prediction = np.argmax(np.bincount(y))
            return self.Node(prediction=leaf_prediction)
        
        # Split the dataset
        left_idx = X[:, split_idx] < split_val
        right_idx = X[:, split_idx] >= split_val
        left_node = self.build_tree(X[left_idx], y[left_idx], depth + 1, max_depth, min_samples_split)
        right_node = self.build_tree(X[right_idx], y[right_idx], depth + 1, max_depth, min_samples_split)
        
        # Create a node with the split feature index and value
        node = self.Node(split=split_idx, value=split_val, left=left_node, right=right_node)
        
        # After building the tree, store the root node in self.tree_root (depth == 0)
        if depth == 0:
            self.tree_root = node
        
        return node

    def search_best_split(self, X: np.ndarray, y: np.ndarray):
        '''
        Implement the search for best split here.

        Expected return:
        - tuple(int, float): Best feature index and split value
        - None: If no split is found
        '''
        n_samples, n_features = X.shape
        best_split = {'index': None, 'value': None, 'gini': float('inf')}
        
        # Iterate over features and their values
        for feature_idx in range(n_features):
            for threshold in np.unique(X[:, feature_idx]):
                left_idx = X[:, feature_idx] < threshold
                right_idx = X[:, feature_idx] >= threshold
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                
                # Calculate Gini impurity for the split
                left_gini = self.split_crit(y[left_idx])
                right_gini = self.split_crit(y[right_idx])
                weighted_gini = (len(y[left_idx]) * left_gini + len(y[right_idx]) * right_gini) / len(y)
                
                if weighted_gini < best_split['gini']:
                    best_split['index'] = feature_idx
                    best_split['value'] = threshold
                    best_split['gini'] = weighted_gini
        
        if best_split['index'] is None:
            return None
        
        return best_split['index'], best_split['value']

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Implement the prediction algorithm here. Return the prediction vector as a numpy array for the 
        input vector X.
        '''
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.array([self.predict_single(x, self.tree_root) for x in X])
    
    def predict_single(self, x: np.ndarray, node: Node):
        '''
        Recursively traverse the tree to make a prediction for a single sample.
        '''
        if node.is_leaf():
            return node.prediction
        if x[node.split] < node.value:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)

def train_XGBoost() -> dict:
    '''
    See instruction for implementation details. This function will be tested on the pre-built enviornment
    with numpy, pandas, xgboost available.
    '''
    aucs_xgboost = {alpha: [] for alpha in alpha_vals}  # Initialize dictionary to store AUCs for each alpha

    np.random.seed(10)  # Set the seed once, to ensure reproducibility

    # Perform bootstrapping for n_bootstraps times
    for _ in range(n_bootstraps):
        # Create bootstrapped samples
        bootstrap_indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_train_bootstrap = X_train[bootstrap_indices]
        y_train_bootstrap = y_train[bootstrap_indices]

        # Calculate scale_pos_weight for imbalanced classes
        unique_classes, class_counts = np.unique(y_train_bootstrap, return_counts=True)
        num_positive = class_counts[1]
        num_negative = class_counts[0]
        scale_pos_weight = num_negative / num_positive

        # Create XGBoost classifier
        for alpha in alpha_vals:
            # Set XGBoost parameters, including L2 regularization (lambda)
            model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                reg_lambda=alpha,  # L2 regularization
                eta=0.01,
                max_depth=5,
                scale_pos_weight=scale_pos_weight,
                n_estimators=max_iter
            )

            # Train the model
            model.fit(X_train_bootstrap, y_train_bootstrap)

            # Evaluate the model on the validation set
            y_pred_proba = model.predict_proba(X_valid)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba)  # Using the helper function to calculate ROC curve
            auc_score = auc(fpr, tpr)  # Using the helper function to calculate AUC
            
            # Append the AUC score to the list for this alpha
            aucs_xgboost[alpha].append(auc_score)

    # Calculate the mean AUC for each alpha over all bootstraps
    mean_aucs_xgboost = {alpha: np.mean(aucs) for alpha, aucs in aucs_xgboost.items()}

    return mean_aucs_xgboost

# Function for hyperparameter tuning
def tune_xgboost_params():
    results = []

    # Loop through combinations of parameters in the grid
    for eta in param_grid['eta']:
        for max_depth in param_grid['max_depth']:
            for min_child_weight in param_grid['min_child_weight']:
                for gamma in param_grid['gamma']:
                    for n_estimators in param_grid['n_estimators']:
                        aucs = []                      
                        # Define XGBoost parameters
                        model = XGBClassifier(
                            objective='binary:logistic',
                            eval_metric='auc',
                            reg_lambda=best_alpha,  # L2 regularization
                            eta=eta,
                            max_depth=max_depth,
                            scale_pos_weight=scale_pos_weight,
                            min_child_weight=min_child_weight,
                            gamma=gamma,
                            n_estimators=n_estimators
                        )

                        # Train the model
                        model.fit(X_train, y_train)
                        
                        # Make predictions on the validation set
                        y_pred_proba = model.predict_proba(X_valid)[:, 1]
                        fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba)
                        auc_score = auc(fpr, tpr)
                        aucs.append(auc_score)
                        results.append({
                            'eta': eta,
                            'max_depth': max_depth,
                            'min_child_weight': min_child_weight,
                            'gamma': gamma,
                            'n_estimators': n_estimators,
                            'auc': auc_score
                        })
    
    return results

dataloader = DataLoader('../data', 42)
dataloader.data_prep()
dataloader.encode(dataloader.data)
dataloader.encode(dataloader.data_test)
dataloader.data_split()

X_train, y_train = dataloader.extract_features_and_label(dataloader.data_train)
X_valid, y_valid = dataloader.extract_features_and_label(dataloader.data_valid)
X_test, y_test = dataloader.extract_features_and_label(dataloader.data_test)

alpha_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
n_bootstraps = 100
max_iter = 100

mean_aucs_xgboost = train_XGBoost()
best_alpha = max(mean_aucs_xgboost, key=mean_aucs_xgboost.get)

# Uncomment the lines below to print the dictionary obtained from train_XGBoost method
'''
print("Alpha Values and Corresponding Mean AUCs:")
for alpha, mean_auc in mean_aucs_xgboost.items():
    print(f"{alpha}: {mean_auc:.5f}")
print("Best alpha value obtained: ", best_alpha)
'''

# Define grid for hyperparameter tuning
param_grid = {
    'eta': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.5, 0.7],
    'n_estimators': [20, 40, 100]
}

# Calculate scale_pos_weight for imbalanced classes
unique_classes, class_counts = np.unique(y_train, return_counts=True)
num_positive = class_counts[1]
num_negative = class_counts[0]
scale_pos_weight = num_negative / num_positive

# Run the hyperparameter tuning
results_tune = tune_xgboost_params()

# Sort results by AUC score
sorted_results_tune = sorted(results_tune, key=lambda x: x['auc'], reverse=True)

# Uncomment the lines below to print the top 5 configurations after hypertuning XGBoost model
'''
print("Top 5 configurations:")
for result in sorted_results_tune[:5]:
    print(f"eta: {result['eta']}, max_depth: {result['max_depth']}, min_child_weight: {result['min_child_weight']}, "
          f"gamma: {result['gamma']}, max_iter: {result['n_estimators']}, Mean AUC: {result['auc']:.5f}")
'''

best_result = sorted_results_tune[0]

X_data, y_data = dataloader.extract_features_and_label(dataloader.data)

# Calculate scale_pos_weight for imbalanced classes on complete dataset
unique_classes, class_counts = np.unique(y_data, return_counts=True)
num_positive = class_counts[1]
num_negative = class_counts[0]
scale_pos_weight = num_negative / num_positive

'''
Initialize the following variable with the best model you have found. This model will be used in testing 
in our pre-built environment.
'''
my_best_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    reg_lambda=best_alpha,  # L2 regularization
    eta=best_result['eta'],
    max_depth=best_result['max_depth'],
    scale_pos_weight=scale_pos_weight,
    min_child_weight=best_result['min_child_weight'],
    gamma=best_result['gamma'],
    n_estimators=best_result['n_estimators']
)

if __name__ == "__main__":
    print("Hello World!")