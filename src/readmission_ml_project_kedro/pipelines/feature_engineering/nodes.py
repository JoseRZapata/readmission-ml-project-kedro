"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.2
"""
import logging
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
import numpy as np

# Assemble pipeline(s)
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


def pre_processing(x: pd.DataFrame,
                   y:pd.Series,
                   parameters: Dict[str, Any]) -> pd.DataFrame:
    """data processing only in the train data but not in the test data

    Args:
        data: Data train frame containing features.
    Returns:
        data: Processed data for training .
    """
    data = pd.concat([x,y],axis=1)

    pipe_functions = [
                ('filter_discharge_col',FunctionTransformer(filter_cols_values,
                                                    kw_args={'filter_col':'discharge_disposition_id',
                                                    'filter_values':[11, 13, 14, 19, 20]}
                                                )),
                ('remove_nan_rows_diag1',FunctionTransformer(filter_cols_values,
                                                    kw_args={'filter_col':'diag_1',
                                                    'filter_values':[np.nan]}
                                                ))

    ]
    pipeline_pre_processing = Pipeline(steps=pipe_functions)
    data_processed = pipeline_pre_processing.fit_transform(data)                                            

    x_out = data_processed[parameters['features']]
    y_out = data_processed[parameters['target_column']]

    logger.info(f"Shape = {x_out.shape} pre_processing")

    return x_out, y_out


def first_processing(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    create pipeline of General transformations to the data like creating new features.

    Args:
        data: train data after splitting
        parameters: list of the general transforms to apply to all the data

    Returns:
        pd.DataFrame: transformed data

    """
    logger.info(f"Shape = {data.shape} first_processing")


    pipe_functions = [
        ('fill_diag2_with_diag1',FunctionTransformer(fill_na_with_col,
                                                    kw_args={'fill_col':'diag_2', 
                                                    'fill_col_from':'diag_1'}
                                                    )),
        ('fill_medical_specialty', FunctionTransformer(fill_na_with_string,
                                                    kw_args={'fill_col':'medical_specialty',
                                                    'fill_string':'Unknown'}
                                                    )),
        ('medication_changes', FunctionTransformer(medication_changes)),
        ('medication_encoding', FunctionTransformer(medication_encoding)),
        ('diagnose_encoding', FunctionTransformer(diagnose_encoding)),
        ('process_medical_specialty', FunctionTransformer(process_medical_specialty))
    ]



    # get methods name for experimentation tracking
    methods = []
    for name, _ in pipe_functions:
        methods.append(name)

    mlflow.log_param('first-processing', methods)

    pipeline_train_data = Pipeline(steps=pipe_functions)
    return data, ('first_processing', pipeline_train_data)

def data_type_split(data: pd.DataFrame, parameters: Dict[str, Any]):

    if parameters['numerical_cols'] and parameters['categorical_cols']:
        numerical_cols = parameters['numerical_cols']
        categorical_cols = parameters['categorical_cols']
    else:
        numerical_cols = make_column_selector(dtype_include=np.numeric)(data)
        categorical_cols = make_column_selector(dtype_exclude=np.numeric)(data)
    mlflow.log_param('num_cols', numerical_cols)
    mlflow.log_param('cat_cols', categorical_cols)

    return numerical_cols, categorical_cols


def numerical_pipeline(numerical_cols, parameters: Dict[str, Any]):
    """
    dictionary o a list of numerical transformations in tuples
    """
    pipe_functions = [
        ('median_imputer', SimpleImputer(strategy='median'))               
    ]
    # get methods name for experimentation tracking
    methods = ['median_imputer']


    mlflow.log_param('numerical_transform', methods)
    numerical_pipe = Pipeline(steps=pipe_functions)
    return ('numerical', numerical_pipe, numerical_cols)


def categorical_pipeline(categorical_cols,
                         parameters: Dict[str, Any]) -> Tuple[str, Pipeline,list]:
    """
    dictionary o a list of categorical transformations in tuples
    """
    pipe_functions = [
        ('mode_imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))                
    ]
    # get methods name for experimentation tracking
    methods = []
    for name, _ in pipe_functions:
        methods.append(name)

    mlflow.log_param('categorical_transform', methods)

    categorical_pipe = Pipeline(steps=pipe_functions)
    return ('categorical', categorical_pipe, categorical_cols)


def last_processing(data: pd.DataFrame,
                    first: tuple,
                    numerical: Tuple,
                    categorical: Tuple ) -> Pipeline:
    pipe_transforms = Pipeline(steps= [
        first,
        ('columns', ColumnTransformer(
                        transformers=[
                            numerical,
                            categorical,
                        ],
                        remainder='drop')
         ),
        ('StandardScaler',StandardScaler()) 
        ])

    data_transformed = pipe_transforms.fit_transform(data)

    mlflow.log_param(f"shape train_transformed", data_transformed.shape)

    return pipe_transforms, data_transformed


def post_processing(x_in: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    General processing to transformed data, like remove duplicates
    important after transformation the data types are numpy ndarray

    Args:
        x_in: x data after transformations
        y_train: y_train

    Returns:

    """
    methods = ["remove duplicates"]

    mlflow.log_param('post-processing', methods)

    y = y_train['readmitted'].to_numpy().reshape(-1, 1)

    data = np.concatenate([x_in, y], axis=1)

    # remove duplicates
    data = np.unique(data, axis=0)
    y_out = data[:, -1]
    x_out = data[:, :-1]
    mlflow.log_param('shape post-processing', x_out.shape)
    return x_out, y_out



# --- help functions ---


def to_categorical(data: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    for x in categorical_cols:
        data[x] = data[x].astype('category')
    return data    


# function to filter data from a column if is in a list of values
def filter_cols_values(data: pd.DataFrame,
                       filter_col: str,
                       filter_values: list) -> pd.DataFrame:
    """Filter columns from data."""
    data = data[~data[filter_col].isin(filter_values)]
    return data



# function to fillna with a string in a column
def fill_na_with_string(data: pd.DataFrame,
                        fill_col: str,
                        fill_string: str) -> pd.DataFrame:
    """Fill missing values in data with a string."""
    data[fill_col] = data[fill_col].fillna(fill_string, axis=0)
    return data

# function to fill na values in one column using the values from another column
def fill_na_with_col(data: pd.DataFrame,
                     fill_col: str,
                     fill_col_from: str) -> pd.DataFrame:
        """Fill missing values in data with values from another column."""
        data[fill_col] = data[fill_col].fillna(data[fill_col_from], axis=0)
        return data


def encoding_columns(data: pd.DataFrame) -> pd.DataFrame:
    data["max_glu_serum"].replace({'>200':1,
                                   '>300':1,
                                   'Norm':0,
                                   'None':-99},
                                    inplace=True)

    data["A1Cresult"].replace({'>7':1,
                               '>8':1,
                               'Norm':0,
                               'None':-99},
                               inplace=True)
    data['change'].replace('Ch', 1, inplace=True)
    data['change'].replace('No', 0, inplace=True)
    data['diabetesMed'].replace('Yes', 1, inplace=True)
    data['diabetesMed'].replace('No', 0, inplace=True)
    return data

def medication_changes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Medication change for diabetics upon admission has been shown in this research: 
    [What are Predictors of Medication Change and Hospital Readmission in Diabetic Patients?](https://www.ischool.berkeley.edu/projects/2017/what-are-predictors-medication-change-and-hospital-readmission-diabetic-patients)
    to be associated with lower readmission rates.
    New variable is created  to count how many changes were made in total for each patient.
    Args:
        data (pd.DataFrame):
    Returns:
        pd.DataFrame: dataframe with new column
    """
    keys = ['metformin', 'repaglinide', 'nateglinide', 
            'chlorpropamide', 'glimepiride', 'glipizide', 
            'glyburide', 'pioglitazone', 'rosiglitazone', 
            'acarbose', 'miglitol', 'insulin', 
            'glyburide-metformin', 'tolazamide', 
            'metformin-pioglitazone','metformin-rosiglitazone',
            'glipizide-metformin', 'troglitazone', 'tolbutamide',
            'acetohexamide']
        
    for col in keys:
        colname = str(col) + 'temp'
        data[colname] = data[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)
        data['numchange'] = 0
    for col in keys:
        colname = str(col) + 'temp'
        data['numchange'] = data['numchange'] + data[colname]
        del data[colname]
    return data

def medication_encoding(data: pd.DataFrame) -> pd.DataFrame:
    keys = ['metformin', 'repaglinide', 'nateglinide', 
            'chlorpropamide', 'glimepiride', 'glipizide', 
            'glyburide', 'pioglitazone', 'rosiglitazone', 
            'acarbose', 'miglitol', 'insulin', 
            'glyburide-metformin', 'tolazamide', 
            'metformin-pioglitazone','metformin-rosiglitazone',
            'glipizide-metformin', 'troglitazone', 'tolbutamide',
            'acetohexamide']
    for col in keys:
        data[col].replace({'No': 0,'Steady': 1 , 'Up':1, 'Down': 1},
                            inplace=True)
    return data

def diagnose_encoding(data: pd.DataFrame) -> pd.DataFrame:

    diag_cols = ['diag_1','diag_2']
    df_copy = data[diag_cols].copy()
    for col in diag_cols:
        df_copy[col] = df_copy[col].str.replace('E','-')
        df_copy[col] = df_copy[col].str.replace('V','-')
        condition = df_copy[col].str.contains('250')
        df_copy.loc[condition,col] = '250'

    df_copy[diag_cols] = df_copy[diag_cols].astype(float)
    for col in diag_cols:
        df_copy['temp']=np.nan

        condition = (df_copy[col]>=390) & (df_copy[col]<=459) | (df_copy[col]==785)
        df_copy.loc[condition,'temp']='Circulatory'

        condition = (df_copy[col]>=460) & (df_copy[col]<=519) | (df_copy[col]==786)
        df_copy.loc[condition,'temp']='Respiratory'

        condition = (df_copy[col]>=520) & (df_copy[col]<=579) | (df_copy[col]==787)
        df_copy.loc[condition,'temp']='Digestive'

        condition = (df_copy[col]>=800) & (df_copy[col]<=999)
        df_copy.loc[condition,'temp']='Injury'

        condition = (df_copy[col]>=710) & (df_copy[col]<=739)
        df_copy.loc[condition,'temp']='Muscoloskeletal'

        condition = (df_copy[col]>=580) & (df_copy[col]<=629) | (df_copy[col]==788)
        df_copy.loc[condition,'temp']='Genitourinary'    

        condition = (df_copy[col]>=140) & (df_copy[col]<=239) | (df_copy[col]==780)
        df_copy.loc[condition,'temp']='Neoplasms'

        condition = (df_copy[col]>=240) & (df_copy[col]<=279) | (df_copy[col]==781)
        df_copy.loc[condition,'temp']='Neoplasms'

        condition = (df_copy[col]>=680) & (df_copy[col]<=709) | (df_copy[col]==782)
        df_copy.loc[condition,'temp']='Neoplasms'

        condition = (df_copy[col]>=790) & (df_copy[col]<=799) | (df_copy[col]==784)
        df_copy.loc[condition,'temp']='Neoplasms'

        condition = (df_copy[col]>=1) & (df_copy[col]<=139)
        df_copy.loc[condition,'temp']='Neoplasms'

        condition = (df_copy[col]>=290) & (df_copy[col]<=319)
        df_copy.loc[condition,'temp']='Neoplasms'

        condition = (df_copy[col]==250)
        df_copy.loc[condition,'temp']='Diabetes'

        df_copy['temp']=df_copy['temp'].fillna('Others')
        condition = df_copy['temp']=='0'
        df_copy.loc[condition,'temp']=np.nan
        df_copy[col]=df_copy['temp']
        df_copy.drop('temp',axis=1,inplace=True)
    data[diag_cols] = df_copy.copy()
    return data

def process_medical_specialty(data: pd.DataFrame) -> pd.DataFrame:
    """
    specialties with few values will be converted to value = `other`
    """
    med_specialty = ['Unknown', 'InternalMedicine', 'Family/GeneralPractice',
                     'Cardiology', 'Surgery-General', 'Orthopedics', 'Gastroenterology',
                     'Nephrology', 'Orthopedics-Reconstructive',
                     'Surgery-Cardiovascular/Thoracic', 'Pulmonology', 'Psychiatry',
                     'Emergency/Trauma', 'Surgery-Neuro', 'ObstetricsandGynecology',
                     'Urology', 'Surgery-Vascular', 'Radiologist']

    data.loc[~data['medical_specialty'].isin(med_specialty),'medical_specialty']='Other'

    return data 

