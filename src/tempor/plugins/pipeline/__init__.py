import abc
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Type

import omegaconf
import rich.pretty
from typing_extensions import Self

from tempor.data import dataset
from tempor.log import logger
from tempor.plugins import plugin_loader
from tempor.plugins.core._params import Params
from tempor.plugins.prediction.one_off.classification import BaseOneOffClassifier
from tempor.plugins.prediction.one_off.regression import BaseOneOffRegressor
from tempor.plugins.prediction.temporal.classification import BaseTemporalClassifier
from tempor.plugins.prediction.temporal.regression import BaseTemporalRegressor
from tempor.plugins.time_to_event import BaseTimeToEventAnalysis
from tempor.plugins.treatments.one_off import BaseOneOffTreatmentEffects
from tempor.plugins.treatments.temporal import BaseTemporalTreatmentEffects

from .generators import (
    _generate_constructor,
    _generate_fit,
    _generate_hyperparameter_space_for_step_impl,
    _generate_hyperparameter_space_impl,
    _generate_pipeline_seq_impl,
    _generate_predict,
    _generate_predict_counterfactuals,
    _generate_predict_proba,
    _generate_sample_hyperparameters_impl,
)

BASE_CLASS_CANDIDATES = (
    BaseOneOffClassifier,
    BaseOneOffRegressor,
    BaseTemporalClassifier,
    BaseTemporalRegressor,
    BaseTimeToEventAnalysis,
    BaseOneOffTreatmentEffects,
    BaseTemporalTreatmentEffects,
)


# TODO: Consider allowing transform-only pipelines.


class PipelineBase:
    stages: List
    """A list of method plugin instances corresponding to each step in the pipeline."""
    plugin_types: List[Type]
    """A list of types denoting the class of each step in the pipeline."""

    def __init__(self, plugin_params: Optional[Dict[str, Dict]] = None, **kwargs) -> None:  # pragma: no cover
        """Instantiate the pipeline, (optionally) providing initialization parameters for constituent step plugins.

        Note:
            The implementations of the methods on this class (``fit``, ``sample_hyperparameters``, etc.) are
            auto-generated by the :class:`PipelineMeta` metaclass.

        Args:
            plugin_params (Optional[Dict[str, Dict]], optional):
                A dictionary like ``{"plugin_1_name": {"init_param_1": value, ...}, ...}``. Defaults to None.
        """
        raise NotImplementedError("Not implemented")

    @staticmethod
    def pipeline_seq(*args: Any) -> str:  # pragma: no cover
        """Get a string representation of the pipeline, stating each stage plugin, e.g. like:
        ``'preprocessing.imputation.temporal.bfill->...->prediction.one_off.classification.nn_classifier'``

        Returns:
            str: String representation of the pipeline.
        """
        raise NotImplementedError("Not implemented")

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> Dict[str, List[Params]]:  # pragma: no cover
        """The pipeline version of the estimator static method of the same name. All the hyperparameters of the
        different stages will be returned.

        Returns:
            Dict[str, List[Params]]: A dictionary with each stage plugin names as keys and corresponding hyperparameter\
                space (``List[Params]``) as values.
        """
        raise NotImplementedError("Not implemented")

    @staticmethod
    def hyperparameter_space_for_step(name: str, *args: Any, **kwargs: Any) -> List[Params]:  # pragma: no cover
        """Return the hyperparameter space (``List[Params]``) for the step of the pipeline as specified by ``name``.

        Args:
            name (str): Name of the pipeline step (i.e. the name of the underlying plugin).

        Returns:
            List[Params]: the hyperparameter space for the step of the pipeline.
        """
        raise NotImplementedError("Not implemented")

    @classmethod
    def sample_hyperparameters(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        """The pipeline version of the estimator method of the same name. Returns a hyperparameter sample.

        Returns:
            Dict[str, Any]: a dictionary with hyperparameter names as keys and corresponding hyperparameter samples\
                as values.
        """
        raise NotImplementedError("Not implemented")

    def fit(self, data: dataset.BaseDataset, *args: Any, **kwargs: Any) -> Self:  # pragma: no cover
        """The pipeline version of the estimator ``fit`` method.

        By analogy to `sklearn`, under the hood, ``fit_transform`` will be called on all the pipeline steps except
        for the last one (the transformer steps of the pipeline), and `fit` will be called on the last step
        (the predictive step of the pipeline).

        Args:
            data (dataset.BaseDataset): Input dataset.

        Returns:
            Self: Returns the fitted pipeline itself.
        """
        raise NotImplementedError("Not implemented")

    def predict(self, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        """The pipeline version of the estimator ``predict`` method. Applicable if the final step of the pipeline has
        a ``predict`` method implemented.

        Args:
            data (dataset.PredictiveDataset): Input dataset.

        Returns:
            Any: the same return type as the final step of the pipeline.
        """
        raise NotImplementedError("Not implemented")

    def predict_proba(self, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        """The pipeline version of the estimator ``predict_proba`` method. Applicable if the final step of the pipeline
        has a ``predict_proba`` method implemented.

        Args:
            data (dataset.PredictiveDataset): Input dataset.

        Returns:
            Any: the same return type as the final step of the pipeline.
        """
        raise NotImplementedError("Not implemented")

    def predict_counterfactuals(
        self, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover
        """The pipeline version of the estimator ``predict_counterfactuals`` method. Applicable if the final step of
        the pipeline has a ``predict_counterfactuals`` method implemented.

        Args:
            data (dataset.PredictiveDataset): Input dataset.

        Returns:
            Any: the same return type as the final step of the pipeline.
        """
        raise NotImplementedError("Not implemented")

    @property
    def predictor_category(self) -> str:
        return self.plugin_types[-1].category

    @property
    def params(self) -> Dict[str, omegaconf.DictConfig]:
        out = dict()
        for p in self.stages:
            out[p.name] = p.params
        return out

    def __rich_repr__(self):
        yield "pipeline_seq", self.pipeline_seq()
        yield "predictor_category", self.predictor_category
        yield "params", {k: omegaconf.OmegaConf.to_container(v) for k, v in self.params.items()}

    def __repr__(self) -> str:
        return rich.pretty.pretty_repr(self)


def prepend_base(base: Type, bases: List[Type]) -> List[Type]:
    if base in bases:
        bases_final = bases
    else:
        bases_final = [base] + bases
    return bases_final


def raise_not_implemented(*args, **kwargs) -> NoReturn:
    raise NotImplementedError("The `{_fit/predict/...}` methods are not implemented for the pipelines")


class PipelineMeta(abc.ABCMeta):
    def __new__(
        cls: Any,
        __name: str,
        __bases: Tuple[type, ...],
        __namespace: Dict[str, Any],
        plugins: Tuple[Type, ...] = tuple(),
        **kwds: Any,
    ) -> Any:
        logger.debug(f"Creating pipeline defined by steps:\n{plugins}")

        # Constructor:
        __namespace["__init__"] = _generate_constructor()

        # Pipeline-specific:
        __namespace["pipeline_seq"] = _generate_pipeline_seq_impl(plugins)

        # sklearn style methods:
        __namespace["fit"] = _generate_fit()
        __namespace["predict"] = _generate_predict()
        __namespace["predict_proba"] = _generate_predict_proba()
        __namespace["predict_counterfactuals"] = _generate_predict_counterfactuals()

        # Hyperparameter methods:
        __namespace["hyperparameter_space"] = _generate_hyperparameter_space_impl(plugins)
        __namespace["hyperparameter_space_for_step"] = _generate_hyperparameter_space_for_step_impl(plugins)
        __namespace["sample_hyperparameters"] = _generate_sample_hyperparameters_impl(plugins)

        # Non-method attributes:
        __namespace["plugin_types"] = list(plugins)

        # Process base classes appropriately.
        bases = PipelineMeta.parse_bases(__bases, plugins)
        logger.debug(f"Pipeline base classes identified as:\n{bases}")

        # Avoid ABC error from the lack of _sk* method implementations.
        __namespace["_fit"] = raise_not_implemented
        __namespace["_predict"] = raise_not_implemented
        __namespace["_predict_proba"] = raise_not_implemented
        __namespace["_predict_counterfactuals"] = raise_not_implemented

        return super().__new__(cls, __name, bases, __namespace, **kwds)

    @staticmethod
    def parse_bases(bases: Tuple[type, ...], plugins: Tuple[Type, ...]):
        bases_final: List[Type] = list(bases)

        if len(plugins) > 0:
            predictive_step = plugins[-1]
            for base_class in BASE_CLASS_CANDIDATES:
                if issubclass(predictive_step, base_class):
                    bases_final = prepend_base(base_class, bases_final)

        bases_final = prepend_base(PipelineBase, bases_final)

        return tuple(bases_final)


def pipeline_classes(names: List[str]) -> Tuple[Type, ...]:
    """Return a list sequence of method plugin classes based on a sequence of fully-qualified ``names`` provided.

    Args:
        names (List[str]): A sequence of fully-qualified names of method plugins, corresponding to pipeline steps.

    Returns:
        Tuple[Type, ...]: The corresponding sequence of method plugin classes.
    """
    res = []

    for fqn in names:
        if "." not in fqn:
            raise RuntimeError(f"Invalid fqn: {fqn}")

        res.append(plugin_loader.get_class(fqn))

    return tuple(res)


def pipeline(plugins_str: List[str]) -> Type[PipelineBase]:
    """Use this method to create pipelines.

    Generates a pipeline (:class:`PipelineBase`) class with an implementation of the necessary methods
    (``fit``, ``sample_hyperparameters`` etc.), based on a sequence of steps defined by ``plugins_str``.

    All but the last steps must be data transformer plugins, and the last step must be a predictive method plugin.

    This method will return a pipeline class (``Type[PipelineBase]``), which should be instantiated. At time of
    instantiation, ``__init__`` input parameters for each step's method plugin can be provided. See
    :class:`PipelineBase` for details.

    Args:
        plugins_str (List[str]):
            A sequence of method plugins' fully-qualified names (e.g.
            ``"prediction.one_off.classification.nn_classifier"``).

    Returns:
        Type[PipelineBase]: The pipeline class (not instance) is returned.
    """
    plugins = pipeline_classes(plugins_str)

    class Pipeline(metaclass=PipelineMeta, plugins=plugins):
        pass

    return Pipeline  # type: ignore
