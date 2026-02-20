"""
Optuna Runner
Hyperparameter optimization using Optuna
"""

from typing import Dict, Any, Callable, Optional
import optuna
from optuna import Trial

from platform_sdk.common.logging import setup_logging

logger = setup_logging(__name__)


class OptunaRunner:
    """Optuna hyperparameter optimization runner"""

    def __init__(
        self,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        direction: str = "maximize"
    ):
        """
        Initialize Optuna runner
        
        Args:
            study_name: Study name
            storage: Storage URL (e.g., postgresql://...)
            direction: Optimization direction (maximize or minimize)
        """
        self.study_name = study_name or "default_study"
        self.storage = storage
        self.direction = direction
        self.optuna_available = self._check_optuna_available()

    def _check_optuna_available(self) -> bool:
        """Check if Optuna is available"""
        try:
            import optuna
            return True
        except ImportError:
            logger.warning("Optuna not installed. Install with: pip install optuna")
            return False

    def create_study(
        self,
        study_name: Optional[str] = None,
        direction: Optional[str] = None
    ) -> optuna.Study:
        """
        Create or load Optuna study
        
        Args:
            study_name: Study name (overrides instance default)
            direction: Direction (overrides instance default)
        
        Returns:
            Optuna Study object
        """
        if not self.optuna_available:
            raise ImportError("Optuna not available")

        study_name = study_name or self.study_name
        direction = direction or self.direction

        if self.storage:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage,
                direction=direction,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(
                study_name=study_name,
                direction=direction
            )

        return study

    def optimize(
        self,
        objective_func: Callable[[Trial], float],
        n_trials: int = 100,
        timeout: Optional[float] = None,
        study: Optional[optuna.Study] = None
    ) -> optuna.Study:
        """
        Run optimization
        
        Args:
            objective_func: Objective function that takes Trial and returns score
            n_trials: Number of trials
            timeout: Timeout in seconds
            study: Optional existing study
        
        Returns:
            Optimized study
        """
        if not self.optuna_available:
            raise ImportError("Optuna not available")

        if study is None:
            study = self.create_study()

        study.optimize(
            objective_func,
            n_trials=n_trials,
            timeout=timeout
        )

        logger.info(f"Optimization completed. Best value: {study.best_value}")
        logger.info(f"Best params: {study.best_params}")

        return study

    def suggest_hyperparameters(
        self,
        trial: Trial,
        param_space: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters from parameter space
        
        Args:
            trial: Optuna Trial
            param_space: Dictionary defining parameter space
                Example:
                {
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
                    "max_depth": {"type": "int", "low": 3, "high": 10},
                    "n_estimators": {"type": "int", "low": 100, "high": 1000}
                }
        
        Returns:
            Dictionary of suggested parameters
        """
        params = {}

        for param_name, param_config in param_space.items():
            param_type = param_config.get("type", "float")

            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    low=param_config["low"],
                    high=param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    low=param_config["low"],
                    high=param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    choices=param_config["choices"]
                )

        return params
