from typing import Dict, Mapping, Sequence

import pandas as pd

from .constants import T_FeatureIndexDtype
from .feature import Feature


class HasFeaturesMixin:
    @property
    def _df_for_features(self) -> pd.DataFrame:
        self._data: pd.DataFrame
        return self._data

    @property
    def features(self) -> Mapping[T_FeatureIndexDtype, Feature]:
        return self._init_features()

    def _init_features(self) -> Mapping[T_FeatureIndexDtype, Feature]:
        features_dict: Dict[T_FeatureIndexDtype, Feature] = dict()
        for c in self._df_for_features.columns:
            features_dict[c] = Feature(name=c, series=self._df_for_features[c])
        return features_dict

    @property
    def feature_names(self) -> Sequence[T_FeatureIndexDtype]:
        return [k for k in self.features.keys()]

    @property
    def all_features_numeric(self) -> bool:
        return all(f.numeric_compatible for f in self.features.values())

    @property
    def all_features_categorical(self) -> bool:
        return all(f.categorical_compatible for f in self.features.values())

    @property
    def all_features_binary(self) -> bool:
        return all(f.binary_compatible for f in self.features.values())

    @property
    def n_features(self) -> int:
        return len(self.features)
