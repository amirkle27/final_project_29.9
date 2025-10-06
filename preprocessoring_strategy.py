"""
Preprocessing strategies used by the model factory.

Each strategy implements a `process()` method that transforms a pandas
DataFrame into features (X) and, when applicable, a target vector (y).
Most strategies:
- drop columns that look like IDs or indices,
- optionally drop missing rows,
- one-hot encode string/categorical columns with `pd.get_dummies`.

Some strategies include additional behavior:
- `PolynomialRegressionPreprocessor` also returns the fitted
  `PolynomialFeatures` transformer to reproduce the same basis at inference.
- `ANNCNNPreprocessor` label-encodes a string target.
- `NLPPreprocessor` tokenizes a single text column and returns padded sequences.

All strategies raise `TargetColumnNotFoundError` when `target_col`
is required but missing.
"""

from abc import ABC, abstractmethod
import pandas as pd
from errors import TargetColumnNotFoundError
from typing import Optional
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences

class PreprocessingStrategy(ABC):
     """Abstract base class for all preprocessing strategies.

    Subclasses must implement `process()` and return a tuple of
    `(X, y)` where `y` can be `None` for unsupervised settings.
    """
    @abstractmethod
    def process(self, df:pd.DataFrame, target_col:str) -> tuple[pd.DataFrame, pd.Series]:
                """Transform raw dataframe into features/target.

        Args:
            df: Input dataframe.
            target_col: Name of the target column (supervised use).

        Returns:
            A tuple `(X, y)` where `X` is the transformed feature dataframe
            and `y` is the target Series.

        Raises:
            TargetColumnNotFoundError: If `target_col` is required but missing.
        """

        pass

class LinearRegressionPreprocessor(PreprocessingStrategy):
     """Preprocessing for linear regression.

    - Drops columns whose names contain 'index' or 'id'.
    - Splits into `X`/`y`.
    - One-hot encodes object (string/categorical) columns in `X`.
    - Does **not** drop rows with NaNs (leave to estimator/pipeline).
    """
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
    """Preprocessing for decision trees / tree ensembles.

    Similar to the linear preprocessor, but does not drop NaNs or scale.
    """
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
    """Preprocessing for logistic classification.

    - Drops likely ID/index columns.
    - Drops rows with any missing values.
    - One-hot encodes object columns in `X`.
    """
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
    """Preprocessing for unsupervised clustering.

    - Drops likely ID/index columns.
    - One-hot encodes object columns.
    - Returns `(X, None)` because there is no target.
    """
    def process(self, df: pd.DataFrame, target_col:Optional[str] = None ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        drop_cols = [col for col in df.columns if "index" in col.lower() or "id" in col.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')


        str_cols = [col for col in df.columns if df[col].dtype == 'object']

        if str_cols:
            df = pd.get_dummies(df, columns=str_cols)

        return df, None

class KNNPreprocessor(PreprocessingStrategy):
    """Preprocessing for K-Nearest Neighbors (regression/classification).

    - Drops likely ID/index columns.
    - Drops rows with missing values (distance-based methods dislike NaNs).
    - If `target_col` is provided: split `(X, y)`; else return `(X, None)`.
    - One-hot encodes object columns.
    """
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
    """Preprocessing for dense/CNN tabular models.

    - Drops likely ID/index columns and rows with NaNs.
    - One-hot encodes object columns in `X`.
    - If the target is of dtype `object`, it is label-encoded to integers,
      and the fitted `LabelEncoder` is stored in `self.label_encoder`
      for downstream inverse transform if needed.

    Attributes:
        label_encoder: Optional `LabelEncoder` fitted on string targets.
    """
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
    """Preprocessing for simple text classification/regression.

    Assumes there is exactly one text column (first `object` column found).
    Uses a Keras `Tokenizer` to convert texts to integer sequences and pads
    them to a fixed length.

    Args:
        num_words: Max vocabulary size (most frequent words kept).
        max_len: Sequence length after padding/truncation.

    Attributes:
        tokenizer: Fitted Keras Tokenizer.
        max_len: Target sequence length.
        fitted: Whether the tokenizer has been fitted.
    """
    def __init__(self, num_words: int = 10000, max_len: int = 100):
        self.tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        self.max_len = max_len
        self.fitted = False

    def process(self, df: pd.DataFrame, target_col: Optional[str] = None) -> tuple[np.ndarray, Optional[pd.Series]]:
        """Tokenize and pad text, returning padded sequences and target.

        Args:
            df: Input dataframe containing one text column and a target.
            target_col: Name of the target column. Must exist.

        Returns:
            A tuple `(X_padded, y)`, where `X_padded` is a NumPy array
            of shape `(n_samples, max_len)`.

        Raises:
            TargetColumnNotFoundError: If `target_col` is missing.
            IndexError: If no object (text) column exists.
        """
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
    """Preprocessing for polynomial regression.

    - Drops likely ID/index columns and rows with NaNs.
    - One-hot encodes object columns in `X`.
    - Returns `(X, y, poly)` where `poly` is a `PolynomialFeatures` instance
      (not yet applied)—so the caller can fit/transform consistently.
    """
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


