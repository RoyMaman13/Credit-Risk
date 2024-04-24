from scipy.stats import uniform, randint
import json
import threading
from sklearn.model_selection import cross_val_predict, cross_validate, StratifiedKFold, train_test_split, RandomizedSearchCV

def input_with_timeout():
    print("This cell run may take a while, Do you want to skip the cell run and load variables from a file? (yes/no): ", end='', flush=True)

    response = None

    # Function to get user input
    def get_input():
        nonlocal response
        response = input().strip().lower()

    # Start a separate thread to get user input
    thread = threading.Thread(target=get_input)
    thread.start()
    thread.join(120)

    # If the thread is still alive, it means timeout occurred
    if thread.is_alive():
        print("\nTimeout reached. Assuming skip_response as 'yes'.")
        response = 'yes'

    return response

def run_parameters_tuning(lgbm_model, cv, X, y):
    skip_response = input_with_timeout()

    if skip_response == 'yes':
        # Load variables from file
        with open("./lgbm_model_paramaters.json", 'r') as file:
            data = json.load(file)

        # Extract variables from loaded data
        lgbm_best_params = data.get('best_parameters')
        lgbm_best_score = data.get('best_score')

        # Check if the data was loaded correctly
        if lgbm_best_params is None or lgbm_best_score is None:
            print("Error: Failed to load parameters or score from file.")
        else:
            print("Loaded variables from file:")
            print("Best Parameters:", lgbm_best_params)
            print("Best Score:", lgbm_best_score)
            return lgbm_best_params 
    else:
        # Define the parameter distribution for RandomizedSearchCV
        param_dist = {
            'learning_rate': uniform(0.001, 0.3),
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 10),
            'min_child_samples': randint(1, 20),
            'subsample': uniform(0.6, 1),
            'colsample_bytree': uniform(0.6, 1),
            'reg_alpha': uniform(0, 0.1),
            'reg_lambda': uniform(0, 0.1),
            'feature_fraction': uniform(0, 1),
            'bagging_fraction': uniform(0, 1)
        }

        # Conduct the RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=lgbm_model, 
            param_distributions=param_dist, 
            scoring='roc_auc', 
            cv=cv, 
            n_iter=20, 
            random_state=42, 
            n_jobs=-1
        )
        random_result = random_search.fit(X, y)

        # Get the best parameters and score from the search
        lgbm_best_params = random_result.best_params_
        lgbm_best_score = random_result.best_score_

        print("Best Parameters:", lgbm_best_params)
        print("Best Score:", lgbm_best_score)
        return lgbm_best_params 
