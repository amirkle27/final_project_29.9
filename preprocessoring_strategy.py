from abc import ABC, abstractmethod
import pandas as pd
from errors import TargetColumnNotFoundError
from typing import Optional
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences

class PreprocessingStrategy(ABC):
    @abstractmethod
    def process(self, df:pd.DataFrame, target_col:str) -> tuple[pd.DataFrame, pd.Series]:
        pass

class LinearRegressionPreprocessor(PreprocessingStrategy):
    def process(self, df:pd.DataFrame, target_col:str) -> tuple[pd.DataFrame, pd.Series]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')
        if target_col not in df.columns:
            raise TargetColumnNotFoundError(target_col)
        y = df[target_col]
        X = df.drop(columns=[target_col])

        str_cols = [col for col in X.columns if X[col].dtype == 'object']

        if str_cols:
            X = pd.get_dummies(X, columns=str_cols)

        return X,y


class DecisionTreePreprocessor(PreprocessingStrategy):
    def process(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')

        if target_col not in df.columns:
            raise TargetColumnNotFoundError(target_col)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        str_cols = [col for col in X.columns if X[col].dtype == 'object']
        if str_cols:
            X = pd.get_dummies(X, columns=str_cols)

        return X, y

class LogisticRegressionPreprocessor(PreprocessingStrategy):
    def process(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')
        df = df.dropna()

        if target_col not in df.columns:
            raise TargetColumnNotFoundError(target_col)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        str_cols = [col for col in X.columns if X[col].dtype == 'object']
        if str_cols:
            X = pd.get_dummies(X, columns=str_cols)

        return X, y

class ClusteringPreprocessor(PreprocessingStrategy):
    def process(self, df: pd.DataFrame, target_col:Optional[str] = None ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')


        str_cols = [col for col in df.columns if df[col].dtype == 'object']

        if str_cols:
            df = pd.get_dummies(df, columns=str_cols)

        return df, None

class KNNPreprocessor(PreprocessingStrategy):
    def process(self, df: pd.DataFrame, target_col: Optional[str] = None) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')
        df = df.dropna()
        if target_col:
            if target_col not in df.columns:
                raise TargetColumnNotFoundError(target_col)

            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df.copy()

        str_cols = [col for col in X.columns if X[col].dtype == 'object']
        if str_cols:
            X = pd.get_dummies(X, columns=str_cols)

        return X, y

class ANNCNNPreprocessor(PreprocessingStrategy):
    def __init__(self):
        self.label_encoder = None

    def process(self, df: pd.DataFrame, target_col: Optional[str] = None) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')
        df = df.dropna()
        if target_col is not None:
            if target_col not in df.columns:
                raise TargetColumnNotFoundError(target_col)
            y = df[target_col]
            X = df.drop(columns=target_col)

            str_cols = [col for col in X.columns if X[col].dtype == 'object']
            if str_cols:
                X = pd.get_dummies(X, columns=str_cols)

            if y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                y = pd.Series(self.label_encoder.fit_transform(y))

            return X, y
        else:
            str_cols = [col for col in df.columns if df[col].dtype == 'object']
            if str_cols:
                df = pd.get_dummies(df, columns=str_cols)
            return df,None

class NLPPreprocessor(PreprocessingStrategy):
    def __init__(self, num_words: int = 10000, max_len: int = 100):
        self.tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        self.max_len = max_len
        self.fitted = False

    def process(self, df: pd.DataFrame, target_col: Optional[str] = None) -> tuple[np.ndarray, Optional[pd.Series]]:
        if target_col not in df.columns:
            raise TargetColumnNotFoundError(target_col)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        text_col = X.select_dtypes(include='object').columns[0]
        texts = X[text_col].astype(str).tolist()

        if not self.fitted:
            self.tokenizer.fit_on_texts(texts)
            self.fitted = True

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

        return padded, y


class PolynomialRegressionPreprocessor(PreprocessingStrategy):
    def __init__(self, degree: int = 2):
        self.degree = degree

    def process(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, PolynomialFeatures]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')
        df = df.dropna()
        if target_col not in df.columns:
            raise TargetColumnNotFoundError(target_col)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        str_cols = [col for col in X.columns if X[col].dtype == 'object']
        if str_cols:
            X = pd.get_dummies(X, columns=str_cols)

        poly = PolynomialFeatures(degree=self.degree, include_bias=False)

        return X, y, poly

