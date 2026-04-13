"""
Wrapper for Evelyn's preprocessing pipeline to integrate with existing experiment code.

This module provides a drop-in replacement for load_and_prepare_data() that uses
Evelyn's comprehensive preprocessing (winsorization, log transformation, normalization, etc.)
while maintaining the same interface as the original data_utils.py functions.

NEW: Supports fine-grained control over feature selection and preprocessing steps via config.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# Import column categorization module
from .column_categorizer import get_feature_columns

logger = logging.getLogger(__name__)


# ==============================================================================
# PREPROCESSING CLASS
# ==============================================================================

import datetime
import logging
import math
import numpy as np
import os
import pandas as pd
import re
import yaml

from category_encoders import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

class Preprocess:
    """
    Class which contains methods to preprocess data.

    Key Methods:
    - drop_null_labels(): drops rows where label is null
    - drop_single_value_cols(): drops columns that have only one value
    - drop_mostly_null_cols(): drops columns that have fewer than share_non_null non-null values
    - drop_repeat_sales(): drops all but the last instance of a property that has been sold multiple times
    - winsorize_continuous(): winsorizes continuous features at wins_pctile and 100-wins_pctile
    - winsorize_label(): winsorizes labels at wins_pctile and 100-wins_pctile
    - one_hot(): one-hot encodes categorical variables
    - normalize_continuous_cols(): normalizes continuous variables using sklearn StandardScaler()
    - gen_time_vars(): generates temporal features from sale_date
    """
    def __init__(self,
                 data: pd.DataFrame, # dataframe to preprocess
                 label: str=None, # string indicating model target or label
                 continuous_cols: list=None, # list of continuous features
                 binary_cols: list=None, # list of binary features
                 categorical_cols: list=None, # list of categorical features
                 meta_cols: list=None, # columns of metadata that methods should not modify
                 sale_date_col: str=None, # string indicating sale date column
                 geography: str=None,
                 share_non_null: float=0.25, # minimum share of non-null values required in each column
                 random_state: int=42, # for reproducibility
                 wins_pctile: int=1, # percentile at which data are winsorized (symmetric)
                 log_label: bool=True, # whether to apply log transformation to label
                 test_size: float=0.2, # desired size of test set
                 log_dir: str='logs' # logger filepath
                 ):
        
        """
        Initialize Preprocessor class and configure logging.
        """

        # set log filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"preprocess_log_{timestamp}.log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)

        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Avoid adding multiple handlers if re-instantiated
        if not self.logger.handlers:
            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("Logger initialized.")
        
        # private attributes
        self._data = data
        self._label = label or ''
        self._continuous_cols = continuous_cols or []
        self._binary_cols = binary_cols or []
        self._categorical_cols = categorical_cols or []
        self._meta_cols = meta_cols or []
        self._sale_date_col = sale_date_col
        self._time_cols = []
        if geography == 'bg':
            self._geo_col = ['block_group']
        else:
            self._geo_col = []
        
        # protected attributes
        self.__share_non_null = share_non_null
        self.__random_state = random_state
        self.__wins_pctile = wins_pctile
        self.__log_label = log_label
        self.__test_size = test_size

        self.logger.info("Preprocess class initialized.")

    ### setters and getters

    # input dataframe
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        self._data = new_data

    # label
    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, new_label):
        if isinstance(new_label, str):
            self._label = label
        else:
            self.logger.error("label must be a string")
            raise ValueError("label must be a string")
    
    # continuous features
    @property
    def continuous_cols(self):
        return self._continuous_cols
    
    @continuous_cols.setter
    def continuous_cols(self, new_continuous):
        if isinstance(new_continuous, list):
            self._continuous_cols = new_continuous
        else:
            self.logger.error("continuous_cols must be a list")
            raise ValueError("continuous_cols must be a list")

    # binary features
    @property
    def binary_cols(self):
        return self._binary_cols
    
    @binary_cols.setter
    def binary_cols(self, new_binary):
        if isinstance(new_binary, list):
            self._binary_cols = new_binary
        else:
            self.logger.error("binary_cols must be a list")
            raise ValueError("binary_cols must be a list")

    # categorical features
    @property
    def categorical_cols(self):
        return self._categorical_cols
    
    @categorical_cols.setter
    def categorical_cols(self, new_categorical):
        if isinstance(new_categorical, list):
            self._categorical_cols = new_categorical
        else:
            self.logger.error("categorical_cols must be a list")
            raise ValueError("categorical_cols must be a list")

    # meta features
    @property
    def meta_cols(self):
        return self._meta_cols

    @meta_cols.setter
    def meta_cols(self, new_meta):
        if isinstance(new_meta, list):
            self._meta_cols = new_meta
        else:
            self.logger.error("meta_cols must be a list")
            raise ValueError("meta_cols must be a list")
    
    # min non-null values
    @property
    def share_non_null(self):
        return self.__share_non_null
    
    @share_non_null.setter
    def share_non_null(self, new_n):
        if new_n >= 0 and new_n <= 1 and isinstance(new_n, float):
            self.__share_non_null = new_n
        else:
            self.logger.error("share_non_null must be a float between 0 and 1")
            raise ValueError("share_non_null must be a float between 0 and 1")

    # random state
    @property
    def random_state(self):
        return self.__random_state
    
    @random_state.setter
    def random_state(self, new_random_state):
        if new_random_state>=0 and isinstance(new_random_state, int):
            self.__random_state = new_random_state
        else:
            self.logger.error("random_state must be a non-negative integer")
            raise ValueError("random_state must be a non-negative integer")
        
    # winsorize percentile

    @property
    def wins_pctile(self):
        return self.__wins_pctile
    
    @wins_pctile.setter
    def wins_pctile(self, new_pctile):
        if new_pctile>=0 and isinstance(new_pctile, int):
            self.__wins_pctile = new_pctile
        else:
            self.logger.error("wins_pctile must be a non-negative integer")
            raise ValueError("wins_pctile must be a non-negative integer")

    def gen_time_vars(self, inplace: bool=True):
        """
        Generates time variables using sale_date_col.

        Variables include:
        - sale_year: year of sale (range is sample dependent)
        - sale_month: month of sale (1-12)
        - sale_day_of_month: day of month of sale, e.g. '31' if property sold on Jan 31. (1-31)
        - sale_day_of_year: day of year of sale, e.g. '65' if property sold on Mar 6 (1-366)
        - sale_day_of_week: day of week of sale (0-6)
        - sale_day: date diff between sale date and Jan 1 2000
        """

        copy = self._data.copy()
        sale_date = copy[self._sale_date_col].astype(str).str.zfill(8)

        # Commented out: Alternative approach for handling non-numeric characters
        # if sale_date.dtype == 'O':
        #     sale_date = sale_date.str.replace(r'\D+', '', regex=True).astype(int)

        # cast sale date as datetime object
        sale_date = pd.to_datetime(sale_date, format='%Y%m%d', errors='coerce')

        # gen sale year
        sale_year = [x.year for x in sale_date]

        # gen sale month
        sale_month = [x.month for x in sale_date]

        # gen sale day of month
        sale_day_of_month = [x.day for x in sale_date]

        # gen sale day of year
        sale_day_of_year = [x.day_of_year for x in sale_date]

        # gen sale day of week
        sale_day_of_week = [x.dayofweek for x in sale_date]

        # gen day of sale
        sale_day = [(x - pd.to_datetime('2000101', format='%Y%m%d')).days for x in sale_date]

        if inplace == True:
            self._data['sale_year'] = sale_year
            self._data['sale_month'] = sale_month
            self._data['sale_day_of_month'] = sale_day_of_month
            self._data['sale_day_of_year'] = sale_day_of_year
            self._data['sale_day_of_week'] = sale_day_of_week
            self._data['sale_day'] = sale_day
            
            self._time_cols = ['sale_year', 'sale_month', 'sale_day_of_month', 'sale_day_of_year', 'sale_day_of_week', 'sale_day']

        else:
            self.logger.warning("gen_time_vars() can only run if inplace==True")
            raise ValueError("gen_time_vars() can only run if inplace==True")
    
    def drop_null_labels(self, inplace: bool=True):

        """
        Drop observations where label is null. Observations are dropped at the row 
        level.

        If inplace==True, observations are dropped in place and self._data indices
        are reset after drop.

        Else, method returns copy of self._data called processed_data where 
        observations with missing labels have been dropped, and indices are 
        reset after drop.
        """

        before = len(self._data)
        if inplace == True:
            self._data.dropna(subset=[self._label], axis=0, inplace=True)
            self._data = self._data.reset_index(drop=True)
            self.logger.info(f"Dropped {before - len(self._data)} rows with null labels out of {before} total rows.")
        else:
            processed_data = self._data.dropna(subset=[self._label], axis=0, inplace=False)
            processed_data = processed_data.reset_index(drop=True)
            self.logger.info(f"Dropped {before - len(processed_data)} rows with null labels out of {before} total rows.")
            return processed_data

    def drop_lowest_ratios(self, inplace: bool=True):

        """
        Drop observations whose sales ratios (as measured using corelogic's SALE_AMOUNT divided by MARKET_TOTAL_VALUE)
        fall in the lowest percentile.
        """

        copy = self._data.copy()

        if 'MARKET_TOTAL_VALUE' not in copy.columns.tolist() or 'SALE_AMOUNT' not in copy.columns.tolist():
            self.logger.warning('MARKET_TOTAL_VALUE and/or SALE_AMOUNT not in data; drop_lowest_ratios() not applied. ')
            return copy

        # define sales ratios
        copy['ratio'] = copy['MARKET_TOTAL_VALUE']/copy['SALE_AMOUNT']

        if 'sale_year' in copy.columns:

            # define helper function percentile_rank
            def percentile_rank(group):
                return group.rank(pct=True) * 100  

            # apply percentile rank to data grouped by sale year
            copy['ratio_percentile'] = copy.groupby('sale_year')['ratio'].transform(percentile_rank).round(0)

            # drop observations in the lowest percentile of ratios by sale year
            copy = copy[copy.ratio_percentile >= 1]

            # drop ratio and ratio percentile column from data
            copy.drop(['ratio', 'ratio_percentile'], axis=1, inplace=True)
            
        else:
            
            # generate percentile bins of sales ratios
            copy['ratio_bins'] = pd.qcut(copy['ratio'], q=100, duplicates='drop', labels=False)

            # flag lowest percentile bin for dropping
            copy['drop'] = [1 if x < 1 else 0 for x in copy.ratio_bins]

            # drop da bin
            copy = copy[copy['drop'] != 1]

            # remove the cols we generated for this function
            copy.drop(['ratio', 'ratio_bins', 'drop'], axis=1, inplace=True)

        if inplace==True:
            self._data = copy
        else:
            return copy

    def drop_single_value_cols(self, inplace: bool=True):

        """
        Drop colums in input data that have only one value.
        This includes columns that are fully null and columns that take on only
        one non-null value, e.g. 0 or 1.

        If inplace==True, columns are dropped in place and self._data indices are reset after drop.
        Method does not return anything.
        
        Else, method returns copy of self._data with dropped columns.
        """

        # identify single-value columns
        drop_cols = [col for col in self._data.columns if self._data[col].nunique(dropna=True) <= 1 and col not in self._meta_cols]

        # drop single-value columns if found
        if drop_cols:
            self.logger.warning(f"Warning: {len(drop_cols)} columns have only one value. Dropping {drop_cols}")
        else:
            self.logger.info("No single-value columns to drop.")
            pass

        if inplace==True:
            self._data = self._data.drop(columns=drop_cols)
            self._data = self._data.reset_index(drop=True)
            self.logger.info("Updating cols attributes to reflect drops")
            self._continuous_cols = [x for x in self._continuous_cols if x in self._data.columns]
            self._binary_cols = [x for x in self._binary_cols if x in self._data.columns]
            self._categorical_cols = [x for x in self._categorical_cols if x in self._data.columns]
            self._time_cols = [x for x in self._time_cols if x in self._data.columns]
            
        else:
            processed_data = self._data.copy()
            processed_data = processed_data.drop(columns=drop_cols)
            processed_data = processed_data.reset_index(drop=True)
            return processed_data

    def drop_mostly_null_cols(self, inplace: bool=True):

        """
        Drop columns in input data that have fewer than share_non_null 
        non-null values. 

        If inplace==True, columns are dropped in place and self._data indices are reset after drop.
        Method does not return anything.

        Else, method returns copy of self._data with dropped columns.
        """

        # identify mostly null cols
        drop_cols = [col for col in self._data.columns if self._data[col].notnull().sum() <= int(np.floor(self.__share_non_null*self._data.shape[0])) and col not in self._meta_cols]

        if drop_cols:
            self.logger.warning(f'Warning: {len(drop_cols)} columns have fewer than {self.__share_non_null*100} percent non-null values. Dropping {drop_cols}')
        else:
            self.logger.info('No mostly null columns to drop.')
            pass

        if inplace==True:
            self._data.drop(columns=drop_cols, inplace=True)
            self._data.reset_index(drop=True, inplace=True)
            self.logger.info("Updating cols attributes to reflect drops")
            self._continuous_cols = [x for x in self._continuous_cols if x in self._data.columns]
            self._binary_cols = [x for x in self._binary_cols if x in self._data.columns]
            self._categorical_cols = [x for x in self._categorical_cols if x in self._data.columns]
            self._time_cols = [x for x in self._time_cols if x in self._data.columns]

        else:
            processed_data = self._data.copy()
            processed_data = processed_data.drop(columns=drop_cols)
            processed_data = processed_data.reset_index(drop=True)
            return processed_data
        
    def drop_repeat_sales(self, timebound: bool=True, inplace: bool=True):

        """
        Drop repeat sales of the same property. Only the most recent sale is kept. 

        If inplace==True, observations are dropped in place and self._data indices
        are reset after drop.

        Else, method returns copy of self._data called processed_data where 
        observations with missing labels have been dropped, and indices are 
        reset after drop.
        """

        before = len(self._data)

        if timebound and 'sale_year' in self._data.columns:
            if inplace==True:
                self._data.drop_duplicates(subset=['CLIP', 'sale_year', 'sale_month'], keep='last', inplace=True)
                self._data = self._data.reset_index(drop=True)
                self.logger.info(f"Dropped {before - len(self._data)} rows with duplicate sales out of {before} total rows.")
            else:
                processed_data = self._data.drop_duplicates(subset=['CLIP', 'sale_year', 'sale_month'], keep='last', inplace=False)
                processed_data = processed_data.reset_index(drop=True)
                self.logger.info(f"Dropped {before - len(processed_data)} rows with duplicate sales out of {before} total rows.")
                return processed_data

        else: 
            if inplace==True:
                self._data.drop_duplicates(subset=['CLIP'], keep='last', inplace=True)
                self._data = self._data.reset_index(drop=True)
                self.logger.info(f"Dropped {before - len(self._data)} rows with duplicate sales out of {before} total rows.")
            else:
                processed_data = self._data.drop_duplicates(subset=['CLIP'], keep='last', inplace=False)
                processed_data = processed_data.reset_index(drop=True)
                self.logger.info(f"Dropped {before - len(processed_data)} rows with duplicate sales out of {before} total rows.")
                return processed_data

    def one_hot(self, inplace: bool=True):

        """
        Method that one-hot encodes categorical features.

        If inplace==True:
        - generates dummies for all columns in self._categorical_cols
        that are present in self._data
        - drops categorical columns inplace in self._data, replacing
        them with dummy values
        - sets self._categorical_cols to []
        - adds dummy columns to self._binary_cols

        Else:
        - generates dummies for self._categorical_cols that are present in
        self._data
        - adds these dummies to a copy of self._data called processed_data
        - drops categorical columns in processed_data
        - returns processed_data
        
        """
        
        # subset to categorical columns that are still present in data
        # after previous methods have been applied.

        categorical_cols = [x for x in self._categorical_cols if x in self._data.columns]

        # generate copy of data
        copy = self._data.copy()

        # define helper function to help handle/normalize nonstandard dtypes within columns

        def normalize_category(x):
            if pd.isna(x):
                return 'missing'
            x_str = str(x).strip()

            if re.fullmatch(r'\d+\.0+', x_str):
                return str(int(float(x_str)))
                
            return x_str
            
        for col in categorical_cols:
            copy[col] = copy[col].apply(normalize_category)

        # define one-hot encoder
        encoder = OneHotEncoder(handle_unknown = 'infrequent_if_exist',
                                sparse_output = False,
                                min_frequency = 0.05 # category must include 5% of data to be encoded
                               )
        # one-hot encode categoricals
        encoded = encoder.fit_transform(copy[categorical_cols])
        
        # get clean column names of encoded variables
        encoded_columns = encoder.get_feature_names_out(categorical_cols)

        # turn encoded variables into dataframe with clean colnames
        encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=copy.index)

        encoded_copy = copy.drop(columns=categorical_cols).join(encoded_df)
    
        if inplace:
            self._data = encoded_copy
            self._binary_cols += encoded_columns.tolist()
            self._categorical_cols = []
        else:
            return encoded_copy

    def renumber_geo_col(self, inplace=True):
        
        """
        Renumbers geography column to consecutive integers starting at zero
        to help improve fit for tree-based models.

        inputs:
        - inplace: If True, geo column is modified inplace.
        """
        
        if len(self._geo_col) > 1:
            raise ValueError("more than one geography column specified; must specify unique column.")
            
        elif len(self._geo_col) > 0:
            # pull and sort unique values of geography column
            unique_vals = self._data[self._geo_col[0]].fillna(0).astype(int).unique().tolist()
            unique_vals.sort()

            # create an ordered list the same length as the unique values of geography column
            ordered_vals = list(range(len(unique_vals)))

            # generate a mapping between unordered and ordered values
            mapper = dict(zip(unique_vals, ordered_vals))
            
            if inplace == True:
                # replace existing geography column in data with ordered values using mapper
                self._data[self._geo_col[0]] = [mapper[int(x)] for x in self._data[self._geo_col[0]].fillna(0).astype(int)]
            else:
                return [mapper[str(x)] for x in self._data[self._geo_col[0]].astype(int)]
        else:
            return None
    
    def train_test_split(self, return_items: bool=False, by_year=True):

        """
        Generates train-test split either by year or using sklearn.train_test_split().

        Inputs:
        - self.__random_state: for repoducibility
        - self.__test_size: desired size of test set as fraction of self_data
        - self._data: full dataset including features, labels, and meta cols
        - self._features: features to include in X_train, X_test
        - self._label: label to include in y_train, y_test
        - self._meta_cols: meta features like property ids that should be 
            split using same indices as X and y but which are not used in 
            model development
        - return_items: if true, returns test-train split 
        - by_year: if True, train-test split sets test set as most-recent year of sales, and train set as previous years of sales.

        Returns:
        self.X_train, self.X_test, self.y_train, self.y_test, self.meta_train, self.meta_test
        as attributes of Preprocess() object.
        """
        if by_year and 'sale_year' in self._data.columns:
            
            copy = self._data.copy()

            # check whether there are sufficient observations in last year of sales to use as test set.
            # if so, use last year of sales as test set.
            if copy[copy.sale_year == copy.sale_year.max()].shape[0]/copy.shape[0] > 0.1:
                self.X_train = copy[copy.sale_year < copy.sale_year.max()][self._binary_cols + self._categorical_cols + self._continuous_cols + self._time_cols + self._geo_col]
                self.X_test = copy[copy.sale_year == copy.sale_year.max()][self._binary_cols + self._categorical_cols + self._continuous_cols + self._time_cols + self._geo_col]
                self.y_train = copy[copy.sale_year < copy.sale_year.max()][self._label]
                self.y_test = copy[copy.sale_year == copy.sale_year.max()][self._label]
                self.meta_train = copy[copy.sale_year < copy.sale_year.max()][self._meta_cols]
                self.meta_test = copy[copy.sale_year == copy.sale_year.max()][self._meta_cols]

            # if there are not enough sales in the last year of data, use the last two years as the test set.
            else:
                self.X_train = copy[copy.sale_year < (copy.sale_year.max()-1)][self._binary_cols + self._categorical_cols + self._continuous_cols + self._time_cols + self._geo_col]
                self.X_test = copy[copy.sale_year >= (copy.sale_year.max()-1)][self._binary_cols + self._categorical_cols + self._continuous_cols + self._time_cols + self._geo_col]
                self.y_train = copy[copy.sale_year < (copy.sale_year.max()-1)][self._label]
                self.y_test = copy[copy.sale_year >= (copy.sale_year.max()-1)][self._label]
                self.meta_train = copy[copy.sale_year < (copy.sale_year.max()-1)][self._meta_cols]
                self.meta_test = copy[copy.sale_year >= (copy.sale_year.max()-1)][self._meta_cols]

                self.logger.info("X_train, X_test, y_train, y_test, meta_train, meta_test now stored as attributes of preprocess() object. Split performed by year of sale.")
        else:
            
            if by_year and 'sale_year' not in self._data.columns:
                self.logger.warning("'sale_year' not in columns; cannot perform train-test split by year. Splitting by random shuffle instead.")
            
            self.X_train, self.X_test, self.y_train, self.y_test, self.meta_train, self.meta_test = train_test_split(self._data[self._binary_cols + self._categorical_cols + self._continuous_cols + self._time_cols + self._geo_col], 
                                                                                                self._data[self._label], 
                                                                                                self._data[self._meta_cols], 
                                                                                                test_size=self.__test_size, 
                                                                                                random_state=self.__random_state)
        
        self.logger.info("X_train, X_test, y_train, y_test, meta_train, meta_test now stored as attributes of preprocess() object. split performed using train_test_split()")

    def _drop_problematic_cols_from_splits(self):
        """
        Drops columns that become problematic after train-test split.
        """
        
        # First identify columns that are bad in either split
        train_problem_cols = self._find_bad_columns(self.X_train)
        test_problem_cols = self._find_bad_columns(self.X_test)
        
        # Union of problematic columns across splits
        all_problem_cols = list(set(train_problem_cols) | set(test_problem_cols))
        
        if all_problem_cols:
            self.logger.warning(
                f"Dropping {len(all_problem_cols)} columns that became problematic after splitting: "
                f"{all_problem_cols}"
            )
            self.X_train = self.X_train.drop(columns=all_problem_cols)
            self.X_test = self.X_test.drop(columns=all_problem_cols)
            self._update_column_attributes()
    
    def _find_bad_columns(self, df):
        """Identifies columns that should be dropped from a dataframe"""
        return [
            col for col in df.columns
            if (df[col].nunique(dropna=True) <= 1) or  # Single-value columns
               (df[col].notnull().sum() < int(np.floor(self.__share_non_null*df.shape[0])))  # Mostly null columns
        ]
    
    def _update_column_attributes(self):
        """Updates the column-type tracking attributes after dropping columns"""
        remaining_cols = set(self.X_train.columns)
        
        self._continuous_cols = [col for col in self._continuous_cols if col in remaining_cols]
        self._binary_cols = [col for col in self._binary_cols if col in remaining_cols]
        self._categorical_cols = [col for col in self._categorical_cols if col in remaining_cols]
        self._time_cols = [col for col in self._time_cols if col in remaining_cols]

    def winsorize_continuous(self, inplace: bool=True):

        """
        Method that winsorizes continuous features at wins_pctile and
        100-wins_pctile.

        Example: if wins_pctile = 1, then data is winsorized at 1st
        and 99th percentile. 

        If inplace=True, continuous features are modified in place.
        
        Otherwise, method returns a winsorized copy of the continuous
        features in the dataframe.
        """
        
        self.logger.info(f"Winsorizing continuous columns at {self.__wins_pctile} and {100-self.__wins_pctile} percentiles")
        valid_continuous = [col for col in self._continuous_cols if col in self.X_train.columns]
        
        if len(valid_continuous) > 0:
            lower = np.percentile(self.X_train[valid_continuous], self.__wins_pctile, axis=0)
            upper = np.percentile(self.X_train[valid_continuous], 100 - self.__wins_pctile, axis=0)
        else:
            self.logger.warning("No continuous columns found for winsorization — skipping.")
            return

        if inplace==True:
            self.X_train[valid_continuous] = np.clip(self.X_train[valid_continuous], lower, upper)
            self.X_test[valid_continuous] = np.clip(self.X_test[valid_continuous], lower, upper)

        else:
            processed_train = self.X_train[valid_continuous].copy()
            processed_train = np.clip(processed_train, lower, upper)

            processed_test = self.X_test[valid_continuous].copy()
            processed_test = np.clip(processed_test, lower, upper)
            
            return processed_train, processed_test
        
    def winsorize_label(self, inplace: bool=True):

        """
        Method that winsorizes label at wins_pctile and
        100-wins_pctile.

        Example: if wins_pctile = 1, then data is winsorized at 1st
        and 99th percentile. 

        If inplace=True, label is modified in place.
        
        Otherwise, method returns a winsorized copy of the label.
        """
        self.logger.info(f"Winsorizing label {self._label} at {self.__wins_pctile} and {100-self.__wins_pctile} percentiles")
        lower = np.percentile(self.y_train, self.__wins_pctile, axis=0)
        upper = np.percentile(self.y_train, 100-self.__wins_pctile, axis=0)

        if inplace==True:
            self.y_train = np.clip(self.y_train, lower, upper)
            self.y_test = np.clip(self.y_test, lower, upper)

        else:
            processed_train = self.y_train.copy()
            processed_train= np.clip(processed_train, lower, upper)
            processed_test = self.y_test.copy()
            processed_test= np.clip(processed_test, lower, upper)
            return processed_data

    def log_label(self, inplace: bool=True):
        """
        Applies log transformation to label if self.__log_label == True.

        If inplace, transformation occurs in place. Otherwise, transformation
        is applied to a copy of the label, and the copy is returned.
        """

        if self.__log_label == True:
            self.logger.info(f"Applying log transformation to {self._label}.")
            # validate that all values are non-negative first
            if (self.y_train <= 0).any() or (self.y_test <= 0).any():
                self.logger.error(f"Label {self._label} contains non-positive values; cannot apply log transformation")
                raise ValueError(f"Label {self._label} contains non-positive values; cannot apply log transformation")
            else:
                if inplace == True:
                    self.y_train = pd.Series([math.log(x) for x in self.y_train])
                    self.y_test = pd.Series([math.log(x) for x in self.y_test])
                    self.logger.info(f"log transformation of {self._label} successfully applied in place.")
                    return None
                else:
                    processed_train = self.y_train.copy()
                    processed_test = self.y_test.copy()
                    processed_train = pd.Series([math.log(x) for x in processed_train])
                    processed_test = pd.Series([math.log(x) for x in processed_test])
                    return processed_train, processed_test

        else:
            self.logger.info(f"log_label is False; log transformation not applied to label {self._label}.")
            return None

    def normalize_continuous_cols(self, inplace: bool=True, include_time_cols: bool=False):
        """
        Normalizes continuous features if present in data using sklearn's
        StandardScaler().

        inputs:
            inplace: boolean specifying whether normalization happens in place or
            on a copy of self._data
            include_time_cols: boolean specifying whether to include time columns

        If inplace==True, continuous columns of  self._data are modified in place. 
        Method does not return anything.

        If inplace==False, method is applied to a copy of self._data.
        Method returns normalized copy of continuous features in self._data.
        """
        if include_time_cols == True: 
            features_to_process = [x for x in self.X_train.columns if x in self._continuous_cols + self._time_cols]

        else: 
            features_to_process = [x for x in self.X_train.columns if x in self._continuous_cols]
            
        if not features_to_process:
            self.logger.info('Data has no continuous columns; normalize_continuous_cols() not applied.')
            return
            
        self.logger.info("Normalizing continuous columns")
        train_subset = self.X_train[features_to_process].copy()

        # define scaler
        scaler = StandardScaler()

        # fit scaler on train set
        scaler.fit(train_subset)

        if inplace==True:
            # scale X_train, X_test inplace
            self.X_train[features_to_process] = pd.DataFrame(
                scaler.transform(self.X_train[features_to_process]),
                columns=features_to_process,
                index=self.X_train.index
            )

            self.X_test[features_to_process] = pd.DataFrame(
                scaler.transform(self.X_test[features_to_process]),
                columns=features_to_process,
                index=self.X_test.index
            )
        else:
            return scaler.transform(self.X_train[features_to_process]), scaler.transform(self.X_test[features_to_process])
        
    def run(self,
            inplace: bool=True,
            one_hot: bool=True,
            gen_time_vars: bool=True,
            drop_lowest_ratios: bool=True,
            drop_repeat_sales: bool=True,
            target_encode: bool=False):

        """
        Run the complete preprocessing pipeline.

        Note: method as currently written only accommodates inplace modifications.
        Recommended usage: create a copy of whatever data you want to preprocess and
        set self._data as the copy.

        Inputs:
        - inplace: bool, determines whether modifications are made in place. current version only supports inplace modifications.
        - one_hot: bool. If True, wrapper calls one_hot() to encode categoricals. Default is False.
        - target_encode: bool. If True, wrapper calls target_encode to encode categorical variables.

        Returns:
        - self._data, modified inplace.
        """

        if inplace == False:
            self.logger.info("Wrapper run() only supports inplace modifications. Set inplace=True or run methods individually with inplace=False.")
            return

        self.logger.info("Running preprocessing pipeline.")
                
        self.drop_null_labels()

        if gen_time_vars:
            self.gen_time_vars()
        
        self.drop_single_value_cols()
        
        self.drop_mostly_null_cols()

        if drop_lowest_ratios:
            self.drop_lowest_ratios()
        
        if drop_repeat_sales:
            self.drop_repeat_sales()

        if one_hot:
            self.one_hot()

        self.renumber_geo_col()

        if len(self._data) == 0:
            self.logger.info("self._data is empty after preprocessing drops")
            raise ValueError("self._data is empty after preprocessing drops")


        self.train_test_split()

        if self.X_train.shape[0] == 0 or self.X_train.empty or self.X_train.shape[1] == 0 or self.X_train.isna().all().all():
            self.logger.info("Train set is completely empty after train-test split")
            raise ValueError("Train set is completely empty after train-test split")

        if self.X_test.shape[0] == 0 or self.X_test.empty or self.X_test.shape[1] == 0 or self.X_test.isna().all().all():
            self.logger.info("Test set is completely empty after train-test split")
            raise ValueError("Test set is completely empty after train-test split")

        if self.__wins_pctile > 0:
            self.winsorize_continuous()
            self.winsorize_label()

        self.log_label()
        
        if target_encode:
            self.target_encode()
            
        self._drop_problematic_cols_from_splits()

        print(self.X_train.info())
        print(self.X_test.info())

        # Note: Imputation is handled in load_and_prepare_data based on config
        # No need to call it here

        self.normalize_continuous_cols()

        def is_effectively_empty(X):
            if isinstance(X, (pd.DataFrame, pd.Series)):
                return X.shape[0] == 0 or X.shape[1] == 0 or X.isna().all().all()
            elif isinstance(X, np.ndarray):
                return X.shape[0] == 0 or (X.ndim > 1 and X.shape[1] == 0) or np.isnan(X).all()
            else:
                return X is None
        
        if is_effectively_empty(self.X_train):
            self.logger.info("Train set is completely empty after preprocessing")
            raise ValueError("Train set is completely empty after preprocessing")

        if is_effectively_empty(self.X_test):
            self.logger.info("Test set is completely empty after preprocessing")
            raise ValueError("Test set is completely empty after preprocessing")
        
        self.logger.info("Preprocessing complete, returning self.X_train, self.X_test, self.y_train, self.y_test, self.meta_train, self.meta_test, self._continuous_cols, self._binary_cols, and self._categorical_cols")

        return self.X_train, self.X_test, self.y_train, self.y_test, self.meta_train, self.meta_test, self._continuous_cols, self._binary_cols, self._categorical_cols


# ==============================================================================
# PIPELINE ORCHESTRATION FUNCTION
# ==============================================================================

def load_and_prepare_data(
    data_path: str,
    feature_config: dict,
    step_config: dict,
    cbg_column: str = 'block_group_id'
):
    """
    Load and preprocess data with fine-grained control over features and steps.

    This is the NEW recommended function that uses the modular column_definitions
    system for feature selection and conditional preprocessing step execution.

    Args:
        data_path: Path to CSV file
        feature_config: Feature selection flags:
            {
                'property_chars': bool,
                'census_bg': bool,
                'census_tract': bool,
                'assessed_value': bool,
                'geographic': bool,
                'temporal': bool,
            }
        step_config: Preprocessing step flags:
            {
                'drop_null_labels': bool,
                'drop_single_value_cols': bool,
                'drop_mostly_null_cols': bool,
                'share_non_null': float,
                'drop_lowest_ratios': bool,
                'drop_repeat_sales': bool,
                'generate_temporal_features': bool,
                'clean_bad_columns': bool,  # Control _drop_problematic_cols_from_splits
                'one_hot_encode': bool,
                'winsorize': bool,
                'winsorize_percentile': int,
                'log_transform_target': bool,
                'normalize_continuous': bool,
                'impute_method': str,  # "median", "mean", "none"
            }
        cbg_column: Name of CBG column (for interface compatibility)

    Returns:
        X_train, y_train, X_test, y_test, cbg_column

    Note:
        - y_train and y_test are LOG-TRANSFORMED if step_config['log_transform_target']=True
        - Make sure to pass log_transformed=True to compute_metrics() in that case
    """
    logger.info(f"Loading data from {data_path} with modular preprocessing")
    logger.info(f"Feature config: {feature_config}")
    logger.info(f"Step config keys: {list(step_config.keys())}")

    # Load raw data
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"Loaded data shape: {df.shape}")

    # Get feature columns using new column categorization system
    feature_cols = get_feature_columns(df, feature_config, target_column='SALE_AMOUNT')

    continuous_cols = feature_cols['continuous_cols']
    binary_cols = feature_cols['binary_cols']
    categorical_cols = feature_cols['categorical_cols']
    meta_cols = feature_cols['meta_cols']

    logger.info(f"Selected features: {len(continuous_cols)} continuous, "
                f"{len(binary_cols)} binary, {len(categorical_cols)} categorical")

    # Initialize Evelyn's Preprocess class
    preprocessor = Preprocess(
        data=df,
        label='SALE_AMOUNT',
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols,
        meta_cols=meta_cols,
        sale_date_col='sale_date',
        geography=None,
        share_non_null=step_config.get('share_non_null', 0.5),
        random_state=42,
        wins_pctile=step_config.get('winsorize_percentile', 1),
        log_label=step_config.get('log_transform_target', True),
        test_size=0.2,
        log_dir='logs'
    )

    # ========================================================================
    # CONDITIONAL PREPROCESSING PIPELINE
    # Apply each step based on step_config flags
    # ========================================================================

    logger.info("Running conditional preprocessing pipeline...")

    # Data cleaning
    if step_config.get('drop_null_labels', True):
        logger.debug("  - Dropping null labels")
        preprocessor.drop_null_labels()

    if step_config.get('generate_temporal_features', True):
        logger.debug("  - Generating temporal features")
        preprocessor.gen_time_vars()

    if step_config.get('drop_single_value_cols', True):
        logger.debug("  - Dropping single-value columns")
        preprocessor.drop_single_value_cols()

    if step_config.get('drop_mostly_null_cols', True):
        logger.debug("  - Dropping mostly-null columns")
        preprocessor.drop_mostly_null_cols()

    if step_config.get('drop_lowest_ratios', True):
        logger.debug("  - Dropping lowest ratio columns")
        preprocessor.drop_lowest_ratios()

    if step_config.get('drop_repeat_sales', True):
        logger.debug("  - Dropping repeat sales")
        preprocessor.drop_repeat_sales()

    # Feature engineering
    if step_config.get('one_hot_encode', True):
        logger.debug("  - One-hot encoding categoricals")
        preprocessor.one_hot()

    preprocessor.renumber_geo_col()

    # Train-test split (always required)
    logger.debug("  - Train-test split")
    preprocessor.train_test_split()

    # Fix string columns before winsorization
    for col in preprocessor._continuous_cols:
        if col in preprocessor.X_train.columns:
            if preprocessor.X_train[col].dtype == 'object':
                logger.warning(f"Column {col} is object type, converting to numeric")
                try:
                    preprocessor.X_train[col] = pd.to_numeric(
                        preprocessor.X_train[col], errors='coerce'
                    )
                    preprocessor.X_test[col] = pd.to_numeric(
                        preprocessor.X_test[col], errors='coerce'
                    )
                except:
                    logger.warning(f"Failed to convert {col}, removing")
                    preprocessor._continuous_cols.remove(col)

    # Outlier handling
    if step_config.get('winsorize', True):
        logger.debug("  - Winsorizing continuous features and labels")
        preprocessor.winsorize_continuous()
        preprocessor.winsorize_label()

    # Target transformation
    if step_config.get('log_transform_target', True):
        logger.debug("  - Log-transforming target")
        preprocessor.log_label()

    # Drop problematic columns after split (if enabled)
    if step_config.get('clean_bad_columns', True):
        logger.debug("  - Cleaning problematic columns from splits")
        preprocessor._drop_problematic_cols_from_splits()

    # Imputation
    impute_method = step_config.get('impute_method', 'median')
    if impute_method in ['median', 'mean']:
        logger.debug(f"  - Imputing with {impute_method}")
        for col in preprocessor.X_train.columns:
            if preprocessor.X_train[col].isnull().any():
                if pd.api.types.is_numeric_dtype(preprocessor.X_train[col]):
                    if impute_method == 'median':
                        fill_value = preprocessor.X_train[col].median()
                    else:  # mean
                        fill_value = preprocessor.X_train[col].mean()
                    preprocessor.X_train[col] = preprocessor.X_train[col].fillna(fill_value)
                    preprocessor.X_test[col] = preprocessor.X_test[col].fillna(fill_value)
                else:
                    fill_value = (
                        preprocessor.X_train[col].mode()[0]
                        if len(preprocessor.X_train[col].mode()) > 0
                        else 0
                    )
                    preprocessor.X_train[col] = preprocessor.X_train[col].fillna(fill_value)
                    preprocessor.X_test[col] = preprocessor.X_test[col].fillna(fill_value)
    elif impute_method == 'none':
        logger.debug("  - Skipping imputation")
    else:
        logger.warning(f"Unknown impute_method: {impute_method}, using median")
        # Use median as fallback
        for col in preprocessor.X_train.columns:
            if preprocessor.X_train[col].isnull().any():
                if pd.api.types.is_numeric_dtype(preprocessor.X_train[col]):
                    fill_value = preprocessor.X_train[col].median()
                    preprocessor.X_train[col] = preprocessor.X_train[col].fillna(fill_value)
                    preprocessor.X_test[col] = preprocessor.X_test[col].fillna(fill_value)

    # Normalization
    if step_config.get('normalize_continuous', True):
        logger.debug("  - Normalizing continuous features")
        preprocessor.normalize_continuous_cols()

    # Get processed data
    X_train = preprocessor.X_train
    y_train = preprocessor.y_train
    X_test = preprocessor.X_test
    y_test = preprocessor.y_test

    # Reset index for consistency with sample_train_set
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Fix string columns in final output
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            try:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                logger.warning(f"Converted column {col} from object to numeric")
            except:
                logger.warning(f"Could not convert column {col} to numeric, dropping")
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])

    logger.info(f"Preprocessing complete!")
    logger.info(f"Train pool shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Target log-transformed: {step_config.get('log_transform_target', True)}")

    return X_train, y_train, X_test, y_test, cbg_column
