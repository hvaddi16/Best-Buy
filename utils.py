import pandas as pd
import numpy as np
from catboost import CatBoostRegressor as cbr
from datetime import datetime as dt
from lightgbm import LGBMRegressor as lbr
from xgboost import XGBRegressor as xgb
from sklearn.ensemble import RandomForestRegressor as rf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from statsmodels.tsa.seasonal import STL
import optuna

def loadTrainData():
    data=pd.read_excel("data/Hackathon Data.xlsx",engine='openpyxl')
    data = data[~data.Encoded_SKU_ID.isna()]
    data = data[[i for i in data.columns if "Unnamed" not in i]]
    print("Shape of Train Dataset: ", data.shape)
    display(data.head())
    return data

def loadValidationData():
    val=pd.read_excel("data/Validation_Data.xlsx",engine='openpyxl')
    val = val[~val.Encoded_SKU_ID.isna()]
    val = val[[i for i in val.columns if "Unnamed" not in i]]
    print("Shape of Validation Dataset: ", val.shape)
    display(val.head())
    return val

def externalDataSources():
    dow_jones_index = pd.read_csv('data/dow_jones.csv')
    gscpi = pd.read_csv('data/gscpi_data.csv',names = ["sales_date", "gscpi"])
    gscpi["sales_date"] = pd.to_datetime(gscpi["sales_date"])
    owid = pd.read_csv('data/owid-covid-data.csv')
    owid = owid[owid["location"]=="United States"][["date", "total_cases", "new_cases"]].rename(columns = {"date":"sales_date"})
    owid["sales_date"] = pd.to_datetime(owid["sales_date"])
    trend = pd.read_csv('data/norm_trend_all_skus.csv')
    trend.columns = [i.lower() for i in trend.columns]
    trend = trend[["encoded_sku_id", "sales_date", "norm_trend"]]
    trend["sales_date"] = pd.to_datetime(trend["sales_date"])
    return dow_jones_index, gscpi, owid, trend

def addDateCols(df):
    """
    This function takes in a pandas dataframe as an argument and adds new columns to it based on the 'sales_date' column.
    The new columns are: 
    1. 'day' - day of the month from 'sales_date' column
    2. 'month' - month of the year from 'sales_date' column
    3. 'year' - year from 'sales_date' column
    4. 'day_of_week' - day of the week from 'sales_date' column (Monday is 0 and Sunday is 6)
    Args:
    df : pandas dataframe containing a column 'sales_date' of datetime type
    Returns:
    df : pandas dataframe with new columns added
    """
    df['day']=df['sales_date'].dt.day
    df['month']=df['sales_date'].dt.month
    df['year']=df['sales_date'].dt.year
    df['day_of_week']=df['sales_date'].dt.dayofweek
    return df

def addCovidGSCPI(df, owid, gscpi, trend):
    """
    The addCovidGSCPI function takes a DataFrame df as an input and performs the following operations:

    It uses the merge function to join the DataFrame owid with df on the 'sales_date' column. 
    This is done using the 'left' merge method, which means that all the rows in df will be included in the merged DataFrame, 
    but any rows in owid that don't match on the 'sales_date' column will be filled with NaN.

    It then uses the merge function again to join the DataFrame gscpi with the merged DataFrame from step 1 on the 'sales_date' column. 
    This is also done using the 'left' merge method.

    Finally, it uses the merge function again to join the DataFrame trend with the merged DataFrame from step 2 on the 'encoded_sku_id' and 'sales_date' columns.
    This is also done using the 'left' merge method.
    The function returns the final merged DataFrame
    It is assumed that owid, gscpi and trend are dataframes that have already been created and are available in the scope of the function, 
    and that 'sales_date' and 'encoded_sku_id' are columns in df and trend respectively.
    Args:
    df : pandas dataframe containing a column 'sales_date' of datetime type
    Returns:
    df : pandas dataframe with new columns added
    """
    df = df.merge(owid, how='left', on='sales_date')
    df = df.merge(gscpi, how='left', on='sales_date')
    df = df.merge(trend, how='left', on = ["encoded_sku_id", "sales_date"])
    return df

def addMarket(df, dow_jones_index):
    """
    This function takes in a pandas dataframe as an argument and adds new columns to it based on the Dow Jones index.
    It first copies the Dow Jones index and converts the date column to datetime format. Then it renames the columns
    of the index dataframe to match the column names of the input dataframe. It then selects the distinct date values
    from the input dataframe and merges it with the index dataframe on the date column. It then fills any missing market
    values with forward fill method and calculates the return and log return of the market. Finally, it merges the 
    market dataframe with the input dataframe on the date column and returns the modified dataframe.
    
    Args:
    df : pandas dataframe containing a column 'sales_date' of datetime type
    
    Returns:
    df : pandas dataframe with new columns added 'market', 'i_return', 'i_return_log'
    """
    index = dow_jones_index.copy()
    index['date'] = pd.to_datetime(index['date'], format="%d-%m-%y")
    index.columns = ['sales_date', 'market']
    dates_df = df[["sales_date"]].drop_duplicates().sort_values("sales_date").reset_index(drop=True)
    index = dates_df.merge(index, how = "left", on="sales_date").sort_values("sales_date").reset_index(drop=True)
    index["market"] = index["market"].fillna(method = "ffill")
    index['i_return'] = index['market'].diff()
    index['i_return_log'] = index['market'].apply(np.log).diff()
    df = df.merge(index, how='left', on='sales_date')
    return df

def cleanNonObjColumns(df):
    """
    This function takes in a pandas dataframe as an argument and performs cleaning and data type conversion on the non-object columns.
    It first converts the column names to lowercase. Then it selects the columns 'encoded_sku_id' and 'daily_units' and converts their data type to int.
    Next, it selects the columns 'retail_price', 'promo_price', 'competitor_price' and replaces any '?' with NaN and converts their data type to float.
    Args:
    df : pandas dataframe
    Returns:
    df : pandas dataframe with cleaned and converted non-object columns
    """
    df.columns = [i.lower() for i in df.columns]
    int_cols = ["encoded_sku_id", "daily_units"]
    df[int_cols] = df[int_cols].astype(int)
    float_cols = ["retail_price", "promo_price", "competitor_price"]
    df[float_cols] = df[float_cols].replace("?", np.nan).astype(float)
    return df

def cleanObjColumns(df):
    """
    This function takes in a pandas dataframe as an argument and performs label encoding on the object columns.
    It first selects all the object columns in the dataframe and stores them in a list 'obj_cols'.
    It creates an empty dictionary 'label_encoders' to store the label encoders for each column.
    It then defines an inner function 'labelEncode' which takes in a column and applies label encoding on it.
    It stores the label encoder object in the 'label_encoders' dictionary with the column name as the key.
    It then applies this inner function on each of the object columns in the dataframe using the 'apply' method.
    It returns the modified dataframe.
    
    Args:
    df : pandas dataframe
    Returns:
    df : pandas dataframe with label encoded object columns
    """
    obj_cols = df.columns[df.dtypes == "object"].tolist()
    label_encoders = {}
    def labelEncode(x):
        le = LabelEncoder()
        le.fit(x)
        label_encoders[x.name] = le
        return le.transform(x)

    df[obj_cols] = df[obj_cols].apply( labelEncode )
    return df


def dateGroupFeats(x):
    """
    This function takes in a pandas dataframe as an argument and creates new columns based on the groupby of 
    specified columns and retail price. 
    It first selects the columns 'subclass_name', 'class_name', 'ml_name', 'category_name' and assigns it to the variable 'cols'.
    Then it iterates over the columns in 'cols' and creates a new column 'retail_price_position_'+c for each column c.
    The new column is created by applying the groupby operation on the input dataframe based on the column c and 
    dividing the retail price by the mean of retail price for each group.
    It returns the modified dataframe
    
    Args:
    x : pandas dataframe
    Returns:
    x : pandas dataframe with new columns 'retail_price_position_'+c for each column c in 'cols'
    """
    cols = ["subclass_name", "class_name", "ml_name", "category_name"]
    for c in cols:
        x['retail_price_position_'+c] = x.groupby(c)["retail_price"].apply(lambda x: x/x.mean())
    return x


def skuGroupFeats(x):
    """
    This function takes in a pandas dataframe as an argument and creates a new column 'units_lag1' based on the 'daily_units' column.
    It first sorts the dataframe by 'sales_date' and creates a new column 'units_lag1' which is the shifted version of the 'daily_units' column by 1.
    It returns the modified dataframe.
    
    Args:
    x : pandas dataframe
    Returns:
    x : pandas dataframe with added column 'units_lag1'
    """
    x=x.sort_values("sales_date")
    x["units_lag1"] = x['daily_units'].shift(1)
    return x

def get_norm_trend(x):
    """
    This function takes in a pandas dataframe as an argument and creates a new column 'trend' based on the 'daily_units' column.
    It uses the STL function from statsmodels package to decompose the time series data in the 'daily_units' column with a period of 365.
    It then takes the trend component and normalizes it by subtracting the mean and dividing by the standard deviation.
    It then adds the normalized trend as a new column 'trend' in the input dataframe.
    It returns the modified dataframe.
    
    Args:
    x : pandas dataframe
    Returns:
    x : pandas dataframe with added column 'trend'
    """
    res = STL(x["daily_units"], period=365).fit()
    trend = res.trend
    x["trend"] = (trend-trend.mean())/trend.std()
    return x


def featureEng(df):
    """
    This function takes in a pandas dataframe as an argument and performs feature engineering on it.
    It first converts the 'sales_date' column to int64, divides it by 1e9 and assigns it back to the same column.
    It then creates a new column 'competitor_price_ratio' which is the ratio of 'competitor_price' to 'retail_price' columns.
    It creates a new column 'discount' which is the difference of 1 and the ratio of 'promo_price' to 'retail_price' columns.
    It creates a new column 'black_friday' which is a binary column denoting whether the sale date is a black friday or not.
    It then creates a new column 'black_friday' which is a column that denotes the proximity of the sale date to the nearest black friday.
    The proximity is calculated by taking the minimum difference of the date and all the black friday dates and dividing it by 691200 (number of seconds in 8 days).
    It applies the 'dateGroupFeats' function on the dataframe grouped by 'sales_date' and then applies the 'skuGroupFeats' function on the dataframe grouped by 'encoded_sku_id'.
    It returns the modified dataframe.

    Args:
    df : pandas dataframe containing a column 'sales_date' of datetime type
    Returns:
    df : modified pandas dataframe 
    """
    df["sales_date"] = df.sales_date.astype('int64')/1e9
    df["competitor_price_ratio"] = df["competitor_price"]/df["retail_price"]
    df["discount"] = 1 - (df["promo_price"]/df["retail_price"])
    df["black_friday"] = ((df["month"]==11) & (df["day_of_week"]==4) & (df["day"]<=29) & (df["day"]>=23)).astype(int)
    bf_days = df[df["black_friday"]==1]["sales_date"].unique()
    df["black_friday"] = df["sales_date"].apply( lambda x: max(0,1 - (np.abs(bf_days-x).min()/691200)) )
    df = df.groupby("sales_date").apply(dateGroupFeats)
    df = df.groupby("encoded_sku_id").apply(skuGroupFeats).droplevel(0)
    return df

def fillMissingDates(x, s, e):
    """
    This function takes in a pandas dataframe, a start date and an end date as arguments.
    It sorts the input dataframe by 'sales_date' column and separates the rows with 'sales_date' greater than the end date.
    It then filters the input dataframe to only keep the rows with 'sales_date' less than or equal to the end date.
    It sets the 'sales_date' column as index and reindexes the dataframe with date range from the start date to end date.
    It fills any missing values in the dataframe using forward fill and backward fill.
    It renames the index as 'sales_date' and resets the index of the dataframe.
    It concatenates the filtered dataframe with the rows that were separated earlier.
    It returns the modified dataframe.
    
    Args:
    x : pandas dataframe
    s : start date as string in format 'YYYY-MM-DD'
    e : end date as string in format 'YYYY-MM-DD'
    Returns:
    x : pandas dataframe with missing dates filled and sorted by 'sales_date'
    """
    x = x.sort_values("sales_date")
    x1 = x[x["sales_date"]>e]
    x = x[x["sales_date"]<=e]
    x = x.set_index('sales_date').reindex(
        pd.date_range(s, e)
    ).fillna(method='ffill').fillna(method='bfill').rename_axis('sales_date').reset_index()
    x = pd.concat([x,x1])
    return x

def cleanData(df):
    """
    This function takes in a pandas dataframe as an input.
    It first calls the 'cleanNonObjColumns' function to clean up the non-object columns of the dataframe.
    It then replaces all negative values in the 'daily_units' column with 0.
    It then extracts the earliest and the 8th last date from the 'sales_date' column and assigns them as start and end dates.
    It groups the dataframe by 'encoded_sku_id' column and applies the 'fillMissingDates' function on each group.
    It then calls the 'addMarket' function, 'addCovidGSCPI' function, 'addDateCols' function and 'cleanObjColumns' function in that order.
    It then calls the 'featureEng' function on the dataframe.
    It sorts the dataframe by 'sales_date' and 'encoded_sku_id' columns and resets the index of the dataframe.
    It returns the cleaned and transformed dataframe.
    
    Args:
    df : pandas dataframe
    Returns:
    df : cleaned and transformed dataframe
    """
    dow_jones_index, gscpi, owid, trend = externalDataSources()
    df = cleanNonObjColumns(df)
    df["daily_units"] = np.where(df.daily_units<0, 0, df.daily_units)
    dates = sorted(df.sales_date.unique())
    start, end = dates[0],dates[-8]
    df = df.groupby('encoded_sku_id').apply(lambda x: fillMissingDates(x, start, end))
    df = addMarket(df, dow_jones_index)
    df = addCovidGSCPI(df, owid, gscpi, trend)
    df = addDateCols(df)
    df = cleanObjColumns(df)
    df = featureEng(df)
    df = df.sort_values(["sales_date", "encoded_sku_id"]).reset_index(drop=True)
    return df

def createTrTs(df, splits=10, test_size=7):
    """
    This function takes in a pandas dataframe and splits it into train and test sets.
    It first extracts all unique dates in the 'sales_date' column and sorts them.
    It then selects the last 'test_size' number of dates and assigns them as the test set.
    It splits the remaining dates into 'splits' number of train sets.
    It then creates a list of test sets by taking the first 'test_size' number of dates from each of the train sets except the first one and appending the last test set.
    It returns the train sets and test sets as two separate lists.
    
    Args:
    df : pandas dataframe
    splits : number of splits for train sets, default value is 10
    test_size : number of last dates to be selected as test set, default value is 7
    
    Returns:
    trains : list of train sets
    tests : list of test sets
    """
    dates = np.array(sorted(df.sales_date.unique()))
    test = dates[-test_size:]
    dates = dates[:-test_size]
    trains = np.array_split(dates,splits)
    tests = [i[:test_size] for i in trains[1:]] + [test]
    return trains, tests

def generateTimeSeriesSplits(trains, tests, df):
    """
    This function takes in a list of train sets, a list of test sets and the original dataframe.
    It iterates over the test sets and for each test set, it concatenates all the train sets up to that point.
    It then creates a train set and a test set by filtering the original dataframe based on the sales dates present in the train and test sets respectively.
    It sorts these dataframes by encoded_sku_id and sales_date and resets the index.
    It returns a tuple of the train and test sets for each iteration.
    
    Args:
    trains : list of train sets
    tests : list of test sets
    df : original dataframe
    
    Returns:
    tr : train set
    ts : test set
    """
    for i, ts in enumerate(tests):
        tr = np.concatenate(trains[:i+1])
        tr,ts = df[df.sales_date.isin(tr)].copy(), df[df.sales_date.isin(ts)].copy()
        yield tr.sort_values(["encoded_sku_id","sales_date"]).reset_index(drop=True),ts.sort_values(["encoded_sku_id","sales_date"]).reset_index(drop=True)

def OneDayForecast(trial):
    """
    This function is an objective function for an optimization algorithm. It is used to optimize the parameters of a lightgbm regressor (lbr) to minimize the mean squared error between the predicted values and actual values.

    The function takes a trial object as input, which is used to suggest the parameter values for the optimization. The parameters being optimized are:

    max_depth: the maximum depth of the decision tree
    reg_alpha: L1 regularization term on weights
    reg_lambda: L2 regularization term on weights
    num_leaves: the number of leaves in the tree
    
    The function then creates an instance of the lightgbm regressor with the suggested parameter values, and trains the model on the input data (tr) and 
    corresponding target values (daily_units). Then it makes prediction on the test data (ts) and calculates the mean squared error between the predicted values and actual values. 
    The function returns the square root of the mean squared error as the final output.

    Args:
    trial : pandas dataframe
    Returns:
    df : scalar, RMSE of the predictions on the test set

    """
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'reg_alpha': trial.suggest_uniform('lambda_l1', 0.001, 1),
        'reg_lambda': trial.suggest_uniform('lambda_l2', 0.001, 1),
        'num_leaves': trial.suggest_int('num_leaves', 2, 10),
    }

    reg = lbr(**params)
    xtr,ytr = tr.drop("daily_units", axis=1), tr["daily_units"]
    ts_save =  ts.copy()
    reg.fit(xtr,ytr)
    yts = reg.predict(ts.drop("daily_units",axis=1))
    return mean_squared_error(ts_save["daily_units"], yts, squared=False)

def SevenDaysForecast(trial):
    """
    The function objective(trial) is using Optuna library to perform a hyperparameter optimization for the LightGBM model. 
    The trial object is passed to the function and it uses the suggest methods of this object to sample the hyperparameters.

    The function sets the initial values for the parameters like max_depth, reg_alpha, reg_lambda, num_leaves 
    and then it creates a LightGBM model object with these parameters.

    The function then takes the training data and trains the model on it. 
    After that, it uses the predict method of the model to predict the units for the test data.

    The function then defines a nested function moving_fill(x) that takes in a dataframe and for each encoded_sku_id it applies 
    the predictions of the model on the test data and updates the predicted values for the next day based on the previous days predictions.

    Finally, the function returns the mean squared error between the predicted and actual values of the test data.

    Args:
    trial : pandas dataframe
    Returns:
    df : scalar, RMSE of the predictions on the test set

    """

    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'reg_alpha': trial.suggest_uniform('lambda_l1', 0.001, 1),
        'reg_lambda': trial.suggest_uniform('lambda_l2', 0.001, 1),
        'num_leaves': trial.suggest_int('num_leaves', 2, 10),
    }

    reg = lbr(**params)
    xtr,ytr = tr.drop("daily_units", axis=1), tr["daily_units"]
    ts_save =  ts.copy()

    reg.fit(xtr,ytr)

    def moving_fill(x):
        temp = tr[tr['encoded_sku_id']==x['encoded_sku_id'].iloc[0]].reset_index(drop=True)
        x = x.reset_index(drop=True)
        for i in range(x.shape[0]):
            prev = temp["daily_units"].iloc[-1] if temp.shape[0]>0 else np.nan
            x.loc[i,"units_lag1"] = x.loc[i-1, "daily_units"] if i>0 else prev
            x.loc[i,'daily_units'] = reg.predict(x.drop(['daily_units'],axis=1).iloc[i:i+1,:])
            # print('S')
        return x

    ts_pred = ts.groupby('encoded_sku_id', as_index=False).apply(moving_fill)
    return mean_squared_error(ts_save["daily_units"], ts_pred["daily_units"], squared=False)