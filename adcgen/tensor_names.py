from .misc import Singleton, Inputerror

from sympy import Symbol

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
    - The ground state density matrix (gs_density): p
      Additionally, an integer representing the perturbation theoretical order
      is appended to the name.
    - ADC amplitudes belonging to the bra (left) state (left_adc_amplitude): X
    - ADC amplitudes belonging to the ket (right) state
      (right_adc_amplitude): X
    - Orbital energies (orb_energy): e
    - Symbolic orbital energy denominators [(e_i - e_a)^-1 -> D^{i}_{a}]
      (sym_orb_denom): D
    """
    eri: str = "V"
    coulomb: str = "v"
    fock: str = "f"
    operator: str = "d"
    gs_amplitude: str = "t"
    gs_density: str = "p"
    left_adc_amplitude: str = "X"
    right_adc_amplitude: str = "Y"
    orb_energy: str = "e"
    sym_orb_denom: str = "D"

    @staticmethod
    def _from_config() -> 'TensorNames':
        """
        Construct the TensorNames instance with values from the config file
        'tensor_names.json'.
        """
        config_file = Path(__file__).parent.resolve() / _config_file
        tensor_names: dict[str, str] = json.load(open(config_file, "r"))
        return TensorNames(**tensor_names)

    @staticmethod
    def defaults() -> dict[str, str]:
        """Returns the default values of all fields."""
        return {field.name: field.default for field in fields(TensorNames)}

    def map_default_name(self, name: str) -> str:
        """
        Takes a tensor name, checks whether it corresponds to any of the
        default names and returns the currently used name.
        """
        # split the name in base and extension for t-amplitudes and
        # ground state densities
        if (split_name := _split_default_t_amplitude(name)) is not None:
            _, ext = split_name
            return self.gs_amplitude + ext
        elif (split_name := _split_default_gs_density(name)) is not None:
            _, ext = split_name
            return self.gs_density + ext

        for field in fields(self):
            if field.default == name:
                return getattr(self, field.name)
        return name  # found not matching default name -> return input

    def rename_tensors(self, expr):
        """
        Renames all tensors in the expression form their default names to the
        currently configured names. Note that only the name of the tensors
        is changed, while their type (Amplitude, AntiSymmetricTensor, ...)
        remains the same.

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
            # find the necessary substitutions
            if field.name == "gs_amplitude":  # special case for t_amplitudes
                subs = []
                for sym in expr.sympy.atoms(Symbol):
                    split_name = _split_default_t_amplitude(sym.name)
                    if split_name is None:
                        continue
                    subs.append((sym.name, new + split_name[1]))
            elif field.name == "gs_density":  # and for gs densities
                subs = []
                for sym in expr.sympy.atoms(Symbol):
                    split_name = _split_default_gs_density(sym.name)
                    if split_name is None:
                        continue
                    subs.append((sym.name, new + split_name[1]))
            else:
                subs = [(field.default, new)]

            for old, new in subs:
                expr.rename_tensor(old, new)
        return expr


# init the TensorNames instance and overwrite the defaults with
# values from the config file
tensor_names = TensorNames._from_config()


def is_t_amplitude(name: str) -> bool:
    """
    Checks whether the tensor name belongs to a ground state amplitude.
    Possible patterns for names are (assuming the default gs_amplitude name t):
    - t
    - tcc (complex conjugate amplitude)
    - tn (where n is any positive integer)
    - tncc
    """
    base, order = split_t_amplitude_name(name)
    order = order.replace("c", "")
    if order:
        return base == tensor_names.gs_amplitude and order.isnumeric()
    else:
        return base == tensor_names.gs_amplitude


def split_t_amplitude_name(name: str) -> tuple[str, str]:
    """
    Split the name of a ground state amplitude in base and extension, e.g.,
    't3cc' -> ('t', '3cc').
    """
    n = len(tensor_names.gs_amplitude)
    return name[:n], name[n:]


def is_adc_amplitude(name: str) -> bool:
    """Checks whether the tensor name belongs to a ADC amplitude."""
    return (name == tensor_names.left_adc_amplitude or
            name == tensor_names.right_adc_amplitude)


def is_gs_density(name: str) -> bool:
    """
    Checks whether the tensor name belongs to the ground state density matrix
    """
    base, order = split_gs_density_name(name)
    if order:
        return base == tensor_names.gs_density and order.isnumeric()
    else:
        return base == tensor_names.gs_density


def split_gs_density_name(name: str) -> tuple[str, str]:
    """Splits the name of a ground state density matrix in base and order."""
    n = len(tensor_names.gs_density)
    return name[:n], name[n:]


def _split_default_t_amplitude(name: str) -> tuple[str, str] | None:
    """
    Checks whether the name belongs to a t amplitude with the default name
    and return the base and extension of the name
    """
    default = tensor_names.defaults()["gs_amplitude"]
    base, ext = name[:len(default)], name[len(default):]
    if base != default:
        return None
    order = ext.replace("c", "")
    if order and not order.isnumeric():
        return None
    return base, ext


def _split_default_gs_density(name: str) -> tuple[str, str] | None:
    """
    Checks whether the name belongs to a ground stat density with the default
    name and returns the base and extension of the name.
    """
    default = tensor_names.defaults()["gs_density"]
    base, order = name[:len(default)], name[len(default):]
    if base != default or (order and not order.isnumeric()):
        return None
    return base, order
