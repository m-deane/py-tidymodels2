"""
Step for creating dummy variables from categorical columns

Wraps sklearn OneHotEncoder for creating indicator variables.
"""

from dataclasses import dataclass
from typing import List, Any
import pandas as pd


@dataclass
class StepDummy:
    """
    Create dummy variables from categorical columns.

    Converts categorical variables to numeric using one-hot encoding.

    Attributes:
        columns: Categorical columns to encode
        one_hot: Use one-hot encoding (True) or integer encoding (False)
    """

    columns: List[str]
    one_hot: bool = True

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepDummy":
        """
        Fit encoder to training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepDummy with fitted encoder
        """
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder

        # Filter to existing columns
        existing_cols = [col for col in self.columns if col in data.columns]

        if len(existing_cols) == 0:
            return PreparedStepDummy(
                columns=[],
                encoder=None,
                one_hot=self.one_hot,
                feature_names=[]
            )

        if self.one_hot:
            # One-hot encoding
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(data[existing_cols])

            # Get feature names
            feature_names = []
            for i, col in enumerate(existing_cols):
                categories = encoder.categories_[i]
                for cat in categories:
                    feature_names.append(f"{col}_{cat}")
        else:
            # Label encoding (integer encoding)
            encoder = {}
            for col in existing_cols:
                le = LabelEncoder()
                le.fit(data[col])
                encoder[col] = le
            feature_names = existing_cols

        return PreparedStepDummy(
            columns=existing_cols,
            encoder=encoder,
            one_hot=self.one_hot,
            feature_names=feature_names
        )


@dataclass
class PreparedStepDummy:
    """
    Fitted dummy encoding step.

    Attributes:
        columns: Columns to encode
        encoder: Fitted sklearn encoder(s)
        one_hot: Whether using one-hot encoding
        feature_names: Names of created features
    """

    columns: List[str]
    encoder: Any
    one_hot: bool
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted encoder to new data.

        Args:
            data: Data to transform

        Returns:
            DataFrame with dummy columns added and original columns removed
        """
        if self.encoder is None or len(self.columns) == 0:
            return data.copy()

        result = data.copy()

        if self.one_hot:
            # One-hot encoding
            encoded = self.encoder.transform(result[self.columns])
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.feature_names,
                index=result.index
            )

            # Remove original columns and add encoded ones
            result = result.drop(columns=self.columns)
            result = pd.concat([result, encoded_df], axis=1)
        else:
            # Label encoding
            for col in self.columns:
                result[col] = self.encoder[col].transform(result[col])

        return result
