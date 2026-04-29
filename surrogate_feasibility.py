"""
Surrogate feasibility model for the MILP–simulation feedback loop.

The surrogate approximates the boundary between feasible and infeasible
regions in the MILP decision-variable space.  It is trained on observed
(solution, feasible/infeasible) pairs produced by the DES simulation and
its linear decision boundary is exposed as a set of coefficients that can
be injected directly into a Gurobi model as a single linear constraint.

Only a logistic-regression surrogate is implemented because its decision
boundary is a hyperplane — directly expressible as a linear Gurobi
constraint without any auxiliary variables or big-M tricks.

Typical usage
-------------
    surrogate = FeasibilitySurrogate(feature_names=['B_1','B_2','B_map','N_map'])

    # After each simulation run:
    surrogate.add_observation(features={'B_1': 270, 'B_2': 130, ...},
                              feasible=False)
    surrogate.fit()

    # If trained, get coefficients to add to the MILP:
    if surrogate.is_trained:
        coeff_info = surrogate.get_constraint_coefficients()
        # coeff_info['coefficients'] : {feature_name: weight}
        # coeff_info['rhs']          : right-hand-side value
        # Constraint form: sum(w_i * x_i) >= rhs
"""

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class FeasibilitySurrogate:
    """
    Logistic-regression surrogate for simulation feasibility.

    The surrogate is fitted to observed (decision-variable vector,
    feasibility label) pairs and exposes its linear decision boundary as
    a Gurobi-compatible linear constraint of the form::

        sum_i  w_i * x_i  >=  rhs

    where ``x_i`` are the original (un-scaled) decision-variable values.

    Parameters
    ----------
    feature_names : list[str]
        Ordered list of decision-variable names used as features.
        Must match the keys used when calling :meth:`add_observation`
        and must correspond to variables that exist in the Gurobi model
        when the returned coefficients are applied.
    min_samples_per_class : int
        Minimum number of observations from *each* class (feasible /
        infeasible) required before the surrogate can be fitted.
        Defaults to 2.
    C : float
        Inverse of the regularization strength passed to
        ``LogisticRegression``.  Larger values mean less regularization.
        Defaults to 1.0.
    """

    def __init__(self, feature_names, min_samples_per_class=2, C=1.0):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn and numpy are required for FeasibilitySurrogate.  "
                "Install them with:  pip install scikit-learn numpy"
            )
        self.feature_names = list(feature_names)
        self.min_samples_per_class = min_samples_per_class
        self.C = C

        self._X = []          # list of raw feature vectors
        self._y = []          # list of int labels (1=feasible, 0=infeasible)
        self._is_trained = False
        self._model = None
        self._scaler = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self):
        """``True`` after at least one successful call to :meth:`fit`."""
        return self._is_trained

    @property
    def num_observations(self):
        """Total number of data points recorded so far."""
        return len(self._y)

    @property
    def num_feasible(self):
        """Number of feasible observations recorded."""
        return sum(self._y)

    @property
    def num_infeasible(self):
        """Number of infeasible observations recorded."""
        return self.num_observations - self.num_feasible

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def add_observation(self, features: dict, feasible: bool) -> None:
        """
        Record one (solution, feasibility) data point.

        Parameters
        ----------
        features : dict
            Mapping ``{feature_name: numeric_value}`` for every name in
            ``self.feature_names``.
        feasible : bool
            Whether the DES simulation confirmed this solution as
            feasible (all bus SOC ≥ 20 %, all MAP SOC ≥ 10 %).
        """
        vec = [float(features[name]) for name in self.feature_names]
        self._X.append(vec)
        self._y.append(int(feasible))

    def fit(self) -> bool:
        """
        Fit (or re-fit) the logistic regression on all observations so far.

        The model is only fitted when both classes (feasible and
        infeasible) have at least ``min_samples_per_class`` examples,
        so that the classifier can learn a meaningful boundary.

        Returns
        -------
        bool
            ``True`` if training succeeded; ``False`` if there were
            insufficient data.
        """
        n_pos = sum(self._y)
        n_neg = len(self._y) - n_pos

        if n_pos < self.min_samples_per_class or n_neg < self.min_samples_per_class:
            return False

        X = np.array(self._X, dtype=float)
        y = np.array(self._y, dtype=int)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = LogisticRegression(
            C=self.C,
            max_iter=2000,
            solver='lbfgs',
            random_state=42,
        )
        self._model.fit(X_scaled, y)
        self._is_trained = True
        return True

    def get_constraint_coefficients(self, safety_margin_std: float = 0.0) -> dict:
        """
        Return the linear feasibility constraint in the original feature space.

        The logistic-regression decision boundary in the standardised
        feature space is::

            w_s^T  x_s  +  b_s  =  0     (positive side → predicted feasible)

        Substituting  ``x_s = (x - mu) / sigma``  and rearranging gives
        the equivalent constraint in original (un-scaled) space::

            sum_i  (w_s_i / sigma_i) * x_i  >=  -b_s + sum_i (w_s_i * mu_i / sigma_i)

        An optional ``safety_margin_std`` (in standardised-space units)
        shifts the right-hand side up, making the constraint more
        conservative (i.e.  further into the predicted-feasible half-space).

        Parameters
        ----------
        safety_margin_std : float
            Extra margin added to the RHS in standardised space.  A value
            of 0.1 means the MILP must be at least 0.1 standard deviations
            inside the predicted-feasible region.

        Returns
        -------
        dict with keys:

            ``'coefficients'``
                ``{feature_name: weight_in_original_space}``
            ``'rhs'``
                Scalar right-hand-side value in original space.
        """
        if not self._is_trained:
            raise RuntimeError("Surrogate has not been fitted yet.  "
                               "Call fit() first.")

        w_s = self._model.coef_[0]          # weights in standardised space
        b_s = float(self._model.intercept_[0])

        mu = self._scaler.mean_             # feature means
        sigma = self._scaler.scale_         # feature std devs

        # Transform back to original space:
        #   w_orig_i = w_s_i / sigma_i
        #   rhs      = -b_s + sum_i(w_s_i * mu_i / sigma_i) + safety_margin_std
        w_orig = w_s / sigma
        rhs = -b_s + float(np.dot(w_s, mu / sigma)) + safety_margin_std

        coefficients = {
            name: float(w_orig[i])
            for i, name in enumerate(self.feature_names)
        }
        return {'coefficients': coefficients, 'rhs': rhs}

    def predict_feasible(self, features: dict) -> tuple:
        """
        Predict whether a candidate solution is feasible.

        Parameters
        ----------
        features : dict
            Mapping ``{feature_name: value}`` as passed to
            :meth:`add_observation`.

        Returns
        -------
        (feasible_predicted : bool, probability : float)
            ``feasible_predicted`` is ``True`` when the predicted
            probability of feasibility is ≥ 0.5.  Before the surrogate
            is trained it optimistically returns ``(True, 1.0)``.
        """
        if not self._is_trained:
            return True, 1.0

        vec = np.array([[float(features[name]) for name in self.feature_names]])
        vec_s = self._scaler.transform(vec)
        prob = float(self._model.predict_proba(vec_s)[0, 1])
        return prob >= 0.5, prob

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the surrogate state."""
        lines = [
            f"FeasibilitySurrogate  ({len(self.feature_names)} features)",
            f"  Observations : {self.num_observations} total "
            f"({self.num_feasible} feasible, {self.num_infeasible} infeasible)",
            f"  Trained      : {self._is_trained}",
        ]
        if self._is_trained:
            coeffs = self.get_constraint_coefficients()
            lines.append("  Decision boundary (surrogate constraint):")
            for name, w in coeffs['coefficients'].items():
                lines.append(f"    {w:+.6f}  ×  {name}")
            lines.append(f"    >=  {coeffs['rhs']:.6f}")
            lines.append("  (positive coefficient → larger value improves feasibility)")
        return "\n".join(lines)
