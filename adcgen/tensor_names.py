from .misc import Singleton, Inputerror
from pathlib import Path
from dataclasses import dataclass, fields
import json


_config_file = "tensor_names.json"


# NOTE: Currently it is only possible to modify the values through
# 'tensor_names.json'. It is not possible to adjust them by modifying the
# class attributes. This is less flexible but ensures that throughout
# a run of the program consistent names are used. Especially,
# due to the caching weird behaviour might be possible
# if names are changed in the middle of a run.
@dataclass(slots=True, frozen=True)
class TensorNames(metaclass=Singleton):
    """
    Singleton class that that is used to define the names of tensors
    used throughout the code. The names can be changed by modifying
    'tensor_names.json'. By default, the following names are used, where
    the attributes storing the names are given in brackets:
    - antisymmetric ERI in physicist notation (eri): V
    - Coulomb integrals in chemist notation (coulomb): v
    - The fock matrix (fock): f
    - The arbitrary N-particle operator matrix (operator): d
    - Ground state amplitudes (gs_amplitude): t
      Additionally, an integer representing the perturbation theoretical order
      and/or 'cc' to represent complex conjugate amplitudes are appended
      to the name.
    - ADC amplitudes belonging to the bra (left) state (left_adc_amplitude): X
    - ADC amplitudes belonging to the ket (right) state
      (right_adc_amplitude): X
    """
    eri: str = "V"
    coulomb: str = "v"
    fock: str = "f"
    operator: str = "d"
    gs_amplitude: str = "t"
    left_adc_amplitude: str = "X"
    right_adc_amplitude: str = "Y"

    @staticmethod
    def _from_config() -> 'TensorNames':
        """
        Construct the TensorNames instance with values from the config file
        'tensor_names.json'.
        """
        config_file = Path(__file__).parent.resolve() / _config_file
        tensor_names: dict[str, str] = json.load(open(config_file, "r"))
        return TensorNames(**tensor_names)

    def rename_tensors(self, expr):
        """
        Renames all tensors in the expression form their default names to the
        currently configured names.

        Parameters
        ----------
        expr: Expr
            The expression using the default tensor names.
        """
        from .expr_container import Expr

        if not isinstance(expr, Expr):
            raise Inputerror("Expr needs to be provided as Expr instance.")

        for field in fields(self):
            new = getattr(self, field.name)
            if field.default == new:  # nothing to do
                continue
            expr.rename_tensor(field.default, new)
        return expr


# init the TensorNames instance and overwrite the defaults with
# values from the config file
tensor_names = TensorNames._from_config()
