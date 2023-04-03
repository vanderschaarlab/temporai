from typing import Any, Optional, Tuple

import numpy as np
import sklearn
import sklearn.base
import sklearn.utils
import sklearn.utils.validation
from packaging.version import Version
from typing_extensions import Self

# --- Utilities. ---

if Version(sklearn.__version__) >= Version("1.1"):
    _has_input_name = True
else:
    _has_input_name = False


def check_y_survival(y_or_event: np.ndarray, *args, allow_all_censored=False) -> Tuple[np.ndarray, ...]:
    """Check that array correctly represents an outcome for survival analysis.

    Args:
        y_or_event (np.ndarray):
            Structured array with two fields, or boolean array. If a structured array, it must contain the binary
            event indicator as first field, and time of event or time of censoring as second field. Otherwise, it is
            assumed that a boolean array representing the event indicator is passed.
        *args:
            List of array-likes. Any number of array-like objects representing time information. Elements that are
            `None` are passed along in the return value.
        allow_all_censored (bool, optional):
            Whether to allow all events to be censored. Defaults to `False.`

    Returns:
        Tuple[np.ndarray, ...]:
            Tuple like ``(event, time)``.
                * event: ``np.ndarray, shape=(n_samples,), dtype=bool``. Binary event indicator.
                * time: ``np.ndarray, shape=(n_samples,), dtype=float``. Time of event or censoring.
                * ...: additional time elements, depending on number of ``args``.
    """
    if len(args) == 0:
        y = y_or_event

        if not isinstance(y, np.ndarray) or y.dtype.fields is None or len(y.dtype.fields) != 2:
            raise ValueError(
                "y must be a structured array with the first field"
                " being a binary class event indicator and the second field"
                " the time of the event/censoring"
            )

        event_field, time_field = y.dtype.names
        y_event = y[event_field]
        time_args: Tuple = (y[time_field],)
    else:
        y_event = np.asanyarray(y_or_event)
        time_args = args

    event = sklearn.utils.check_array(y_event, ensure_2d=False)
    if not np.issubdtype(event.dtype, np.bool_):
        raise ValueError("elements of event indicator must be boolean, but found {0}".format(event.dtype))

    if not (allow_all_censored or np.any(event)):
        raise ValueError("all samples are censored")

    return_val = [event]
    for i, yt in enumerate(time_args):
        if yt is None:
            return_val.append(yt)
            continue

        yt = sklearn.utils.check_array(yt, ensure_2d=False)
        if not np.issubdtype(yt.dtype, np.number):
            raise ValueError("time must be numeric, but found {} for argument {}".format(yt.dtype, i + 2))

        return_val.append(yt)

    return tuple(return_val)


def _check_estimate_1d(estimate, test_time):
    estimate = sklearn.utils.check_array(
        estimate,
        ensure_2d=False,
        **dict(input_name="estimate") if _has_input_name else dict(),
    )
    if estimate.ndim != 1:
        raise ValueError("Expected 1D array, got {:d}D array instead:\narray={}.\n".format(estimate.ndim, estimate))
    sklearn.utils.check_consistent_length(test_time, estimate)
    return estimate


def _check_inputs(event_indicator, event_time, estimate):
    sklearn.utils.check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = sklearn.utils.check_array(
        event_indicator,
        ensure_2d=False,
        **dict(input_name="event_indicator") if _has_input_name else dict(),
    )
    event_time = sklearn.utils.check_array(
        event_time,
        ensure_2d=False,
        **dict(input_name="event_time") if _has_input_name else dict(),
    )
    estimate = _check_estimate_1d(estimate, event_time)

    if not np.issubdtype(event_indicator.dtype, np.bool_):
        raise ValueError(
            "only boolean arrays are supported as class labels for survival analysis, got {0}".format(
                event_indicator.dtype
            )
        )

    if len(event_time) < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    return event_indicator, event_time, estimate


def _check_times(test_time, times):
    times = sklearn.utils.check_array(
        np.atleast_1d(times),
        ensure_2d=False,
        dtype=test_time.dtype,
        **dict(input_name="times") if _has_input_name else dict(),
    )
    times = np.unique(times)

    if times.max() >= test_time.max() or times.min() < test_time.min():
        raise ValueError(
            "all times must be within follow-up time of test data: [{}; {}[".format(test_time.min(), test_time.max())
        )

    return times


def _check_estimate_2d(estimate, test_time, time_points, estimator):
    estimate = sklearn.utils.check_array(
        estimate,
        ensure_2d=False,
        allow_nd=False,
        estimator=estimator,
        **dict(input_name="estimate") if _has_input_name else dict(),
    )
    time_points = _check_times(test_time, time_points)
    sklearn.utils.check_consistent_length(test_time, estimate)

    if estimate.ndim == 2 and estimate.shape[1] != time_points.shape[0]:
        raise ValueError(
            "expected estimate with {} columns, but got {}".format(time_points.shape[0], estimate.shape[1])
        )

    return estimate, time_points


def _compute_counts(
    event: np.ndarray, time: np.ndarray, order: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Count right censored and uncensored samples at each unique time point.

    Args:
        event (np.ndarray):
            Boolean event indicator.
        time (np.ndarray):
            Survival time or time of censoring.
        order (Optional[np.ndarray]):
            Indices to order time in ascending order.
            If None, order will be computed.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Tuple like ``(times, n_events, n_at_risk, n_censored)``
                * times (np.ndarray): Unique time points.
                * n_events (np.ndarray): Number of events at each time point.
                * n_at_risk (np.ndarray): Number of samples that have not been censored or have not had an event at
                each time point.
                * n_censored (np.ndarray): Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = np.argsort(time, kind="mergesort")

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty(n_samples, dtype=int)
    uniq_counts = np.empty(n_samples, dtype=int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = np.resize(uniq_times, j)
    n_events = np.resize(uniq_events, j)
    total_count = np.resize(uniq_counts, j)
    n_censored = total_count - n_events

    # Offset cumulative sum by one.
    total_count = np.r_[0, total_count]
    n_at_risk = n_samples - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored


def _compute_counts_truncated(
    event: np.ndarray, time_enter: np.ndarray, time_exit: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute counts for left truncated and right censored survival data.

    Args:
        event (np.ndarray):
            Boolean event indicator.
        time_start (np.ndarray):
            Time when a subject entered the study.
        time_exit (np.ndarray):
            Time when a subject left the study due to an
            event or censoring.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Tuple like ``(times, n_events, n_at_risk)``.
                * times (np.ndarray): Unique time points.
                * n_events (np.ndarray): Number of events at each time point.
                * n_at_risk (np.ndarray): Number of samples that are censored or have an event at each time point.
    """
    if (time_enter > time_exit).any():
        raise ValueError("exit time must be larger start time for all samples")

    n_samples = event.shape[0]

    uniq_times = np.sort(np.unique(np.r_[time_enter, time_exit]), kind="mergesort")
    total_counts = np.empty(len(uniq_times), dtype=int)
    event_counts = np.empty(len(uniq_times), dtype=int)

    order_enter = np.argsort(time_enter, kind="mergesort")
    order_exit = np.argsort(time_exit, kind="mergesort")
    s_time_enter = time_enter[order_enter]
    s_time_exit = time_exit[order_exit]

    t0 = uniq_times[0]
    # Everything larger is included:
    idx_enter = np.searchsorted(s_time_enter, t0, side="right")
    # Everything smaller is excluded.
    idx_exit = np.searchsorted(s_time_exit, t0, side="left")

    total_counts[0] = idx_enter
    # Except people die on the day they enter.
    event_counts[0] = 0

    for i in range(1, len(uniq_times)):
        ti = uniq_times[i]

        while idx_enter < n_samples and s_time_enter[idx_enter] <= ti:
            idx_enter += 1

        while idx_exit < n_samples and s_time_exit[idx_exit] < ti:
            idx_exit += 1

        risk_set = np.setdiff1d(order_enter[:idx_enter], order_exit[:idx_exit], assume_unique=True)
        total_counts[i] = len(risk_set)

        count_event = 0
        k = idx_exit
        while k < n_samples and s_time_exit[k] == ti:
            if event[order_exit[k]]:
                count_event += 1
            k += 1
        event_counts[i] = count_event

    return uniq_times, event_counts, total_counts


def kaplan_meier_estimator(
    event: Any,
    time_exit: Any,
    time_enter: Any = None,
    time_min: Optional[float] = None,
    reverse: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimator of survival function. See [1] for further description.

    Args:
        event:
            Array-like, ``shape = (n_samples,)``. Contains binary event indicators.
        time_exit:
            Array-like, ``shape = (n_samples,)``. Contains event/censoring times.
        time_enter:
            Array-like, ``shape = (n_samples,)``, optional. Contains time when each individual entered the study for
            left truncated survival data. Defaults to `None`.
        time_min (float, optional):
            Compute estimator conditional on survival at least up to the specified time. Defaults to `None`.
        reverse (bool, optional).
            Whether to estimate the censoring distribution. When there are ties between times at which events are
            observed, then events come first and are subtracted from the denominator. Only available for
            right-censored data, i.e. ``time_enter`` must be None. Defaults to `False`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Tuple like ``(time, prob_survival)``.
                * time: ``np.ndarray, shape = (n_times,)`` Unique times.
                * prob_survival:
                    ``np.ndarray, shape = (n_times,)``. Survival probability at each unique time point.
                    If ``time_enter`` is provided, estimates are conditional probabilities.

    References:
        [1] Kaplan, E. L. and Meier, P., "Nonparametric estimation from incomplete observations", Journal of The
        American Statistical Association, vol. 53, pp. 457-481, 1958.
    """
    event, time_enter, time_exit = check_y_survival(  # pylint: disable=unbalanced-tuple-unpacking
        event, time_enter, time_exit, allow_all_censored=True
    )
    sklearn.utils.check_consistent_length(event, time_enter, time_exit)

    if time_enter is None:
        uniq_times, n_events, n_at_risk, n_censored = _compute_counts(event, time_exit)

        if reverse:
            n_at_risk -= n_events
            n_events = n_censored
    else:
        if reverse:
            raise ValueError("The censoring distribution cannot be estimated from left truncated data")

        uniq_times, n_events, n_at_risk = _compute_counts_truncated(event, time_enter, time_exit)

    # account for 0/0 = nan
    ratio = np.divide(
        n_events,
        n_at_risk,
        out=np.zeros(uniq_times.shape[0], dtype=float),
        where=n_events != 0,
    )
    values = 1.0 - ratio

    if time_min is not None:
        mask = uniq_times >= time_min
        uniq_times = np.compress(mask, uniq_times)
        values = np.compress(mask, values)

    y = np.cumprod(values)
    return uniq_times, y


class SurvivalFunctionEstimator(sklearn.base.BaseEstimator):
    """Kaplan-Meier estimate of the survival function."""

    def __init__(self):
        pass

    def fit(self, y: np.ndarray) -> Self:
        """Estimate survival distribution from training data.

        Args:
            y (np.ndarray):
                Structured array, ``shape = (n_samples,)``. A structured array containing the binary event indicator
                as first field, and time of event or time of censoring as second field.

        Returns:
            Self.
        """
        event, time = check_y_survival(y, allow_all_censored=True)  # pylint: disable=unbalanced-tuple-unpacking

        unique_time, prob = kaplan_meier_estimator(event, time)
        self.unique_time_ = np.r_[-np.infty, unique_time]  # pylint: disable=attribute-defined-outside-init
        self.prob_ = np.r_[1.0, prob]  # pylint: disable=attribute-defined-outside-init

        return self

    def predict_proba(self, time: np.ndarray) -> np.ndarray:
        r"""Return probability of an event after given time point.  :math:`\hat{S}(t) = P(T > t)`.

        Args:
            time (np.ndarray):
                Array, ``shape = (n_samples,)``. Time to estimate probability at.

        Returns:
            np.ndarray: ``shape = (n_samples,)``. Probability of an event.
        """
        sklearn.utils.validation.check_is_fitted(self, "unique_time_")
        time = sklearn.utils.check_array(
            time,
            ensure_2d=False,
            estimator=self,
            **dict(input_name="estimate") if _has_input_name else dict(),
        )

        # K-M is undefined if estimate at last time point is non-zero.
        extends = time > self.unique_time_[-1]
        if self.prob_[-1] > 0 and extends.any():
            raise ValueError(
                "time must be smaller than largest " "observed time point: {}".format(self.unique_time_[-1])
            )

        # Beyond last time point is zero probability.
        Shat = np.empty(time.shape, dtype=float)
        Shat[extends] = 0.0

        valid = ~extends
        time = time[valid]
        idx = np.searchsorted(self.unique_time_, time)
        # For non-exact matches, we need to shift the index to left.
        eps = np.finfo(self.unique_time_.dtype).eps
        exact = np.absolute(self.unique_time_[idx] - time) < eps
        idx[~exact] -= 1
        Shat[valid] = self.prob_[idx]

        return Shat


class CensoringDistributionEstimator(SurvivalFunctionEstimator):
    """Kaplan-Meier estimator for the censoring distribution."""

    def fit(self, y: np.ndarray):
        """Estimate censoring distribution from training data.

        Args:
            y (np.ndarray):
                Structured array, ``shape = (n_samples,)``. A structured array containing the binary event indicator
                as first field, and time of event or time of censoring as second field.

        Returns:
            Self.
        """
        event, time = check_y_survival(y)  # pylint: disable=unbalanced-tuple-unpacking
        if event.all():
            self.unique_time_ = np.unique(time)  # pylint: disable=attribute-defined-outside-init
            self.prob_ = np.ones(self.unique_time_.shape[0])  # pylint: disable=attribute-defined-outside-init
        else:
            unique_time, prob = kaplan_meier_estimator(event, time, reverse=True)
            self.unique_time_ = np.r_[-np.infty, unique_time]  # pylint: disable=attribute-defined-outside-init
            self.prob_ = np.r_[1.0, prob]  # pylint: disable=attribute-defined-outside-init

        return self

    def predict_ipcw(self, y: np.ndarray) -> np.ndarray:
        r"""Return inverse probability of censoring weights at given time points.
        :math:`\omega_i = \delta_i / \hat{G}(y_i)`

        Args:
            y (np.ndarray):
                Structured array, ``shape = (n_samples,)``. A structured array containing the binary event indicator
                as first field, and time of event or time of censoring as second field.

        Returns:
            np.ndarray:
                IPCW, ``array, shape = (n_samples,)``. Inverse probability of censoring weights.
        """
        event, time = check_y_survival(y)  # pylint: disable=unbalanced-tuple-unpacking
        Ghat = self.predict_proba(time[event])

        if (Ghat == 0.0).any():
            raise ValueError("censoring survival function is zero at one or more time points")

        weights = np.zeros(time.shape[0])
        weights[event] = 1.0 / Ghat

        return weights


def _get_comparable(event_indicator, event_time, order):
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        start = i + 1
        end = start
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = np.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
        i = end

    return comparable, tied_time


def _estimate_concordance_index(
    event_indicator, event_time, estimate, weights, tied_tol=1e-8
) -> Tuple[float, int, int, int, int]:
    order = np.argsort(event_time)

    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    if len(comparable) == 0:
        raise ValueError("Data has no comparable pairs, cannot estimate concordance index.")

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask]]

        if not event_i:
            raise RuntimeError("got censored sample at index %d, but expected uncensored" % order[ind])

        ties = np.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def create_structured_array(
    event: Any,
    time: Any,
    name_event: Optional[str] = None,
    name_time: Optional[str] = None,
) -> np.ndarray:
    """Create structured array.

    Args:
        event (Any):
            Array-like. Event indicator. A boolean array or array with values 0/1.
        time (Any):
            Array-like. Observed time.
        name_event (Optional[str]):
            Name of event, optional. Defaults to "event".
        name_time (Optional[str]):
            Name of observed time, optional. Defaults to "time".

    Returns:
        np.ndarray:
            Structured array with two fields.
    """
    name_event = name_event or "event"
    name_time = name_time or "time"
    if name_time == name_event:
        raise ValueError("name_time must be different from name_event")

    time = np.asanyarray(time, dtype=float)
    y = np.empty(time.shape[0], dtype=[(name_event, bool), (name_time, float)])
    y[name_time] = time

    event = np.asanyarray(event)
    sklearn.utils.check_consistent_length(time, event)

    if np.issubdtype(event.dtype, np.bool_):
        y[name_event] = event
    else:
        events = np.unique(event)
        events.sort()
        if len(events) != 2:
            raise ValueError("event indicator must be binary")

        if np.all(events == np.array([0, 1], dtype=events.dtype)):
            y[name_event] = event.astype(bool)
        else:
            raise ValueError("non-boolean event indicator must contain 0 and 1 only")

    return y


# --- Metrics. ---


def brier_score(
    survival_train: np.ndarray, survival_test: np.ndarray, estimate: Any, times: Any
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Estimate the time-dependent Brier score for right censored data.

    The time-dependent Brier score is the mean squared error at time point :math:`t`:

    .. math::
        \mathrm{BS}^c(t) = \frac{1}{n} \sum_{i=1}^n I(y_i \leq t \land \delta_i = 1)
        \frac{(0 - \hat{\pi}(t | \mathbf{x}_i))^2}{\hat{G}(y_i)} + I(y_i > t)
        \frac{(1 - \hat{\pi}(t | \mathbf{x}_i))^2}{\hat{G}(t)} ,

    where :math:`\hat{\pi}(t | \mathbf{x})` is the predicted probability of remaining event-free up to time point
    :math:`t` for a feature vector :math:`\mathbf{x}`, and :math:`1/\hat{G}(t)` is a inverse probability of
    censoring weight, estimated by the Kaplan-Meier estimator.

    See [1] for details.

    Args:
        survival_train (np.ndarray):
            Structured array, ``shape = (n_train_samples,)``.
            Survival times for training data to estimate the censoring distribution from. A structured array containing
            the binary event indicator as first field, and time of event or time of censoring as second field.
        survival_test (np.ndarray):
            Structured array, ``shape = (n_samples,)``.
            Survival times of test data. A structured array containing the binary event indicator as first field, and
            time of event or time of censoring as second field.
        estimate (Any):
            Array-like, ``shape = (n_samples, n_times)``.
            Estimated risk of experiencing an event for test data at `times`. The i-th column must contain the
            estimated probability of remaining event-free up to the i-th time point.
        times (Any):
            Array-like, ``shape = (n_times,)``.
            The time points for which to estimate the Brier score. Values must be within the range of follow-up times
            of the test data `survival_test`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Tuple like ``(times, bier_scores)``.
                * times: ``np.ndarray, shape=(n_times,)``. Unique time points at which the brier scores was estimated.
                * brier_scores: ``np.ndarray , shape=(n_times,)``. Values of the brier score.

    References:
        [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher, "Assessment and comparison of prognostic
        classification schemes for survival data," Statistics in Medicine, vol. 18, no. 17-18, pp. 2529-2545, 1999.
    """
    test_event, test_time = check_y_survival(survival_test)  # pylint: disable=unbalanced-tuple-unpacking
    estimate, times = _check_estimate_2d(estimate, test_time, times, estimator="brier_score")
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)

    # Fit IPCW estimator:
    cens = CensoringDistributionEstimator().fit(survival_train)

    # Calculate inverse probability of censoring weight at current time point t.
    prob_cens_t = cens.predict_proba(times)
    prob_cens_t[prob_cens_t == 0] = np.inf
    # Calculate inverse probability of censoring weights at observed time point.
    prob_cens_y = cens.predict_proba(test_time)
    prob_cens_y[prob_cens_y == 0] = np.inf

    # Calculating the brier scores at each time point:
    brier_scores = np.empty(times.shape[0], dtype=float)
    for i, t in enumerate(times):
        est = estimate[:, i]
        is_case = (test_time <= t) & test_event
        is_control = test_time > t

        brier_scores[i] = np.mean(
            np.square(est) * is_case.astype(int) / prob_cens_y
            + np.square(1.0 - est) * is_control.astype(int) / prob_cens_t[i]
        )

    return times, brier_scores


def concordance_index_ipcw(
    survival_train: np.ndarray,
    survival_test: np.ndarray,
    estimate: Any,
    tau: Optional[float] = None,
    tied_tol: float = 1e-8,
) -> Tuple[float, int, int, int, int]:
    r"""Concordance index for right-censored data based on inverse probability of censoring weights.

    This is an alternative to the estimator in ``concordance_index_censored`` that does not depend on the distribution
    of censoring times in the test data. Therefore, the estimate is unbiased and consistent for a population
    concordance measure that is free of censoring.

    It is based on inverse probability of censoring weights, thus requires access to survival times from the
    training data to estimate the censoring distribution. Note that this requires that survival times ``survival_test``
    lie within the range of survival times `survival_train`. This can be achieved by specifying the truncation time
    ``tau``. The resulting ``cindex`` tells how well the given prediction model works in predicting events that occur in
    the time range from 0 to ``tau``.

    The estimator uses the Kaplan-Meier estimator to estimate the censoring survivor function. Therefore,
    it is restricted to situations where the random censoring assumption holds and censoring is independent of the
    features.

    See [1] for further description.

    Args:
        survival_train (np.ndarray):
            Structured array, ``shape = (n_train_samples,)``.
            Survival times for training data to estimate the censoring distribution from a structured array containing
            the binary event indicator as first field, and time of event or time of censoring as second field.
        survival_test (np.ndarray):
            Structured array, ``shape = (n_samples,)``.
            Survival times of test data. A structured array containing the binary event indicator as first field,
            and time of event or time of censoring as second field.
        estimate:
            Array-like, ``shape = (n_samples,)``. Estimated risk of experiencing an event of test data.
        tau (float, optional):
            Truncation time. The survival function for the underlying censoring time distribution :math:`D` needs to be
            positive at ``tau``, i.e., ``tau`` should be chosen such that the probability of being censored after time
            ``tau`` is non-zero: :math:`P(D > \tau) > 0`. If `None`, no truncation is performed.
        tied_tol (float, optional).
            The tolerance value for considering ties. If the absolute difference between risk scores is smaller or
            equal than ``tied_tol``, risk scores are considered tied. Defaults to ``1e-8``.

    Returns:
        Tuple[float, int, int, int, int]:
            Tuple like ``(cindex, concordant, discordant, tied_risk, tied_time)``.
                * cindex (float): Concordance index.
                * concordant (int): Number of concordant pairs.
                * discordant (int): Number of discordant pairs.
                * tied_risk (int): Number of pairs having tied estimated risks.
                * tied_time (int): Number of comparable pairs sharing the same time.

    References:
        [1] Uno, H., Cai, T., Pencina, M. J., D'Agostino, R. B., & Wei, L. J. (2011). "On the C-statistics for
        evaluating overall adequacy of risk prediction procedures with censored survival data". Statistics in Medicine,
        30(10), 1105-1117.
    """
    test_event, test_time = check_y_survival(survival_test)  # pylint: disable=unbalanced-tuple-unpacking

    if tau is not None:
        mask = test_time < tau
        survival_test = survival_test[mask]

    estimate = _check_estimate_1d(estimate, test_time)

    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw_test = cens.predict_ipcw(survival_test)
    if tau is None:
        ipcw = ipcw_test
    else:
        ipcw = np.empty(estimate.shape[0], dtype=ipcw_test.dtype)
        ipcw[mask] = ipcw_test  # pyright: ignore
        ipcw[~mask] = 0  # pyright: ignore

    w = np.square(ipcw)

    return _estimate_concordance_index(test_event, test_time, estimate, w, tied_tol)
