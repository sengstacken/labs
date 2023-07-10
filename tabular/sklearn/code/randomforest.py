import argparse
import joblib
import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# inference function for model loading
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":

    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--max_depth", type=int, default=10)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--data", type=str, default=os.environ.get("SM_CHANNEL_DATA"))
    parser.add_argument("--seed", type=int, default=54321)

    args, _ = parser.parse_known_args()
    
    # read in all data 
    df = pd.read_csv(args.data+'/abalone.data',header=None, names=['sex', 'length', 'diameter', 'height', 'weight', 'shucked_weight',
                                       'viscera_weight', 'shell_weight', 'rings'])
    
    # split data into training and validation
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=args.seed, 
        shuffle=True, 
    )
   
    # train
    print("training model")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators, 
        min_samples_leaf=args.min_samples_leaf, 
        oob_score = True, 
        n_jobs=-1,
        verbose=2,
        random_state=args.seed,
    )

    model.fit(train_df.iloc[:,0:-1].values, train_df.iloc[:,-1].values)

    # print accuracy
    print("validating model")
    y_pred = model.predict(val_df.iloc[:,0:-1].values)
    rmse = mean_squared_error(val_df.iloc[:,-1].values, y_pred)
    mape = mean_absolute_percentage_error(val_df.iloc[:,-1].values, y_pred)

    print(f"RMSE is: {rmse}")
    print(f"MAPE is: {mape}")
    print(f"OOB Score: {model.oob_score_:.3}")

    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)