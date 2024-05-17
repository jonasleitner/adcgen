from sympy import S, Dummy
from .misc import Inputerror, Singleton


# base names for all used indices
idx_base = {'occ': 'ijklmno', 'virt': 'abcdefgh', 'general': 'pqrstuvw'}


class Index(Dummy):
    """
    Represents an Index. Wrapper implementation around the 'sympy.Dummy'
    class, which means Index("x") != Index("x").
    Important assumptions:
    - below_fermi: The index represents an occupied orbital.
    - above_fermi: The index represents a virtual orbital.
    - alpha: The index represents an alpha (spatial) orbital.
    - beta: The index represents a beta (spatial) orbital.
    """

    @property
    def spin(self) -> str:
        """
        The spin of the index. An empty string is returned if no spin is
        defined.
        """
        if self.assumptions0.get("alpha"):
            return "a"
        elif self.assumptions0.get("beta"):
            return "b"
        else:
            return ""

    @property
    def space(self) -> str:
        """The space to which the index belongs (occupied/virtual/general)."""
        if self.assumptions0.get("below_fermi"):
            return "occ"
        elif self.assumptions0.get("above_fermi"):
            return "virt"
        else:
            return "general"

    def _latex(self, printer) -> str:
        ret = self.name
        if (spin := self.spin):
            spin = "alpha" if spin == "a" else "beta"
            ret += "_{\\" + spin + "}"
        return ret


class Indices(metaclass=Singleton):
    """
    Manages the indices used thoughout the package and ensures that
    only a single class instance exists for each index.
    This is necessary because the 'equality' operator is essentially replaced
    by the 'is' operator for the indices: Index("x") != Index("x").
    """
    def __init__(self):
        # dict that holds all symbols that have been created previously.
        self._symbols = {'occ': {}, 'virt': {}, 'general': {},
                         'occ_a': {}, 'occ_b': {},
                         'virt_a': {}, 'virt_b': {},
                         'general_a': {}, 'general_b': {}}
        # dict that holds the generic indices. Automatically filled by
        # generated index strings.
        self._generic_indices = {'occ': [], 'occ_a': [], 'occ_b': [],
                                 'virt': [], 'virt_a': [], 'virt_b': [],
                                 'general': [], 'general_a': [],
                                 'general_b': []}
        # o/v indices that are exclusively available for direct request via
        # get_indices, i.e. they can't be generic indices.
        self._occ = ('i', 'j', 'k', 'l', 'm', 'n', 'o',
                     'i1', 'j1', 'k1', 'l1', 'm1', 'n1', 'o1',
                     'i2', 'j2', 'k2', 'l2', 'm2', 'n2', 'o2')
        self._virt = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                      'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
                      'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2')
        self._general = ('p', 'q', 'r', 's', 't', 'u', 'v', 'w')

    def _gen_generic_idx(self, space: str, spin: str):
        """
        Generated the next 'generation' of generic indices, i.e. extends
        _generic_indices list by incrementing the integer attached to the index
        base. The first generic indices will increment the highest integer
        found in the _occ/_virt tuples by one.
        """
        # first call -> counter has not been initialized yet.
        key = space if spin is None else f"{space}_{spin}"
        if not hasattr(self, f'counter_{key}'):
            idx_list = sorted(
                getattr(self, f'_{space}'), key=lambda idx:
                (int(idx[1:]) if idx[1:].isdigit() else 0, idx[0])
            )
            n = int(idx_list[-1][1:]) if idx_list[-1][1:].isdigit() else 0
            setattr(self, f'counter_{key}', n + 1)

        # generate the new index strings
        counter = getattr(self, f'counter_{key}')
        used = self._symbols[key]
        new_idx = [idx + str(counter) for idx in idx_base[space]
                   if idx + str(counter) not in used]

        # extend the generic list and increment the counter
        self._generic_indices[key].extend(new_idx)
        setattr(self, f'counter_{key}', counter + 1)

    def get_indices(self, indices: str, spins: str = None) -> dict:
        """
        Obtain the Indices for the provided string.

        Parameters
        ----------
        indices : str
            The names of the indices as a single string that is split
            automatically as "ij21kl3" -> i, j21, k, l3.
        spins : str, optional
            The spins of the indices as a single string, e.g., "aaba"
            for obtaining four indices with spin: alpha, alpha, beta, alpha.

        Returns
        -------
        dict
            key: the space and spin of the indices as string.
            value: list containing the indices in the order the indices are
                   provided in the indices input.
        """
        if not isinstance(indices, str):
            raise Inputerror(f"Indices {indices} need to be of type string.")
        indices = split_idx_string(indices)
        if spins is not None and len(indices) != len(spins):
            raise Inputerror(f"Indices {indices} and spin {spins} do not "
                             "match.")

        ret = {}
        for i, idx in enumerate(indices):
            space = index_space(idx)
            if spins is None:
                spin = None
                key = space
            else:
                spin = spins[i]
                key = space + "_" + spin
            if key not in ret:
                ret[key] = []
            # check whether the symbol is already available
            s = self._symbols[key].get(idx, None)
            if s is not None:
                ret[key].append(s)
                continue
            # did not find -> create a new symbol + add it to self._symbols
            # and if it is a generic index -> remove it from the current
            # generic index list.
            s = self._new_symbol(idx, space, spin)
            ret[key].append(s)
            self._symbols[key][idx] = s
            try:
                self._generic_indices[key].remove(idx)
            except ValueError:
                pass
        return ret

    def get_generic_indices(self, **kwargs) -> dict:
        """
        Request indices with unique names that have not been used in the
        current run of the program. Easy way to ensure that contracted indices
        do not appear anywhere else in a term.
        Indices can be requested with the keywords
        n_occ, n_virt, n_general (or n_o, n_v, n_g)
        to obtain indices without spin. Spin can be defined by appending
        '_a' or '_b' to the keyword.

        Returns
        -------
        dict
            key: the space and spin of the indices as string.
            value: list containing the indices in the order they are
                   provided in the indices input.
        """

        ret = {}
        for key, n in kwargs.items():
            if n == 0:
                continue
            # n_space(_spin)   where spin is optional
            key = key.split("_")[1:]
            if not key:
                raise Inputerror(f"{key} is not a valid input for requesting "
                                 "generic indices. Use e.g. 'n_occ=2'.")
            space = key[0]  # expand o/v/g
            if space == 'o':
                space = 'occ'
            elif space == 'v':
                space = 'virt'
            elif space == 'g':
                space = 'general'

            if len(key) == 1:
                spin = None
                spins = None
                key = space
            elif len(key) == 2:
                spin = key[1]
                spins = "".join(spin for _ in range(n))
                key = f"{space}_{spin}"
            else:
                raise ValueError(f"Invalid input argument {key}. Use e.g. "
                                 "'n_occ=2'.")
            while n > len(self._generic_indices[key]):
                self._gen_generic_idx(space, spin)
            idx = "".join(self._generic_indices[key][:n])
            ret.update(self.get_indices(idx, spins))
        return ret

    def _new_symbol(self, idx: str, ov: str, spin: str | None) -> Index:
        """Creates a new symbol with the defined name, space and spin."""
        assumptions = {}
        if ov == "occ":
            assumptions["below_fermi"] = True
        elif ov == "virt":
            assumptions["above_fermi"] = True
        elif ov != "general":
            raise ValueError(f"Invalid space {ov}")
        if spin is not None:
            if spin == "a":
                assumptions["alpha"] = True
            elif spin == "b":
                assumptions["beta"] = True
            else:
                raise ValueError(f"Invalid spin {spin}.")
        return Index(idx, **assumptions)

    def substitute_with_generic(self, expr):
        """
        Substitute all contracted indices with new, generic (unused) indices.
        """
        from . import expr_container as e

        def substitute_contracted(term: e.Term) -> e.Expr:
            # count how many indices need to be replaced and get new indices
            old = {}
            for s in term.contracted:
                key = s.space
                if s.spin:
                    key += "_" + s.spin
                if key not in old:
                    old[key] = []
                old[key].append(s)

            if not old:
                return term

            kwargs = {"n_" + key: len(idx) for key, idx in old.items()}
            new = self.get_generic_indices(**kwargs)

            sub = {}
            for idx_type, old_idx in old.items():
                new_idx = new[idx_type]
                if len(old_idx) != len(new_idx):
                    raise RuntimeError(f"{len(old_idx)} {idx_type} indices "
                                       "needed but generated only "
                                       f"{len(new_idx)} indices.")
                sub.update({s: new_s for s, new_s in zip(old_idx, new_idx)})

            new_term = term.subs(order_substitutions(sub))

            # ensure substitutions are valid
            if new_term.sympy is S.Zero and term.sympy is not S.Zero:
                raise ValueError(f"Substitutions {sub} are not valid for "
                                 f"{term}.")
            return new_term

        expr = expr.expand()
        if not isinstance(expr, e.Expr):
            expr = e.Expr(expr)
        substituted = e.Expr(0, **expr.assumptions)
        for term in expr.terms:
            substituted += substitute_contracted(term)
        return substituted


def index_space(idx: str) -> str:
    """Returns the space an index belongs to (occ/virt/general)."""
    for sp, idx_string in idx_base.items():
        if idx[0] in idx_string:
            return sp
    raise Inputerror(f"Could not assign the index {idx} to a space.")


def sort_idx_canonical(idx: Index):
    """Use as sort key to to bring indices in canonical order."""
    if isinstance(idx, Index):
        # also add the hash here for wicks, where multiple i are around
        return (idx.space[0],
                idx.spin,
                int(idx.name[1:]) if idx.name[1:] else 0,
                idx.name[0],
                hash(idx))
    else:  # necessary for subs to work correctly with simultaneous=True
        return ('', 0, str(idx), hash(idx))


def split_idx_string(str_tosplit: str) -> list:
    """
    Splits an index string of the form 'ij12a3b' in a list ['i','j12','a3','b']
    """
    splitted = []
    temp = []
    for i, idx in enumerate(str_tosplit):
        temp.append(idx)
        try:
            if str_tosplit[i+1].isdigit():
                continue
            else:
                splitted.append("".join(temp))
                temp.clear()
        except IndexError:
            splitted.append("".join(temp))
    return splitted


def n_ov_from_space(space_str: str):
    """
    Number of required occupied and virtual indices required for the given
    exictation space, e.g., "pph" -> 2 virtual and 1 occupied index.
    """
    return {'n_occ': space_str.count('h'), 'n_virt': space_str.count('p')}


def repeated_indices(idx_a: str, idx_b: str) -> bool:
    """Checks whether both index strings share an index."""
    split_a = split_idx_string(idx_a)
    split_b = split_idx_string(idx_b)
    return any(i in split_b for i in split_a)


def get_lowest_avail_indices(n: int, used: list[str], space: str) -> list[str]:
    """
    Return the names of the n lowest available indices belonging to the desired
    space.

    Parameters
    ----------
    n : int
        The number of available indices.
    used : list[str]
        The names of the indices that are already in use.
    space : str
        The space (occ/virt/general) to which the indices belong.
    """
    # generate idx pool to pick the lowest indices from
    base = idx_base[space]
    idx = list(base)
    required = len(used) + n  # the number of indices present in the term
    suffix = 1
    while len(idx) < required:
        idx.extend(s + str(suffix) for s in base)
        suffix += 1
    # remove the already used indices (that are not available anymore)
    # and return the first n elements of the resulting list
    return [s for s in idx if s not in used][:n]


def extract_names(syms: list[Index] | dict):
    """
    Extracts the names of the provided indices and returns them in a list.
    """
    from itertools import chain
    if isinstance(syms, dict):
        syms = chain.from_iterable(syms.values())
    return [s.name for s in syms]


def get_symbols(idx: str | Index | list[str] | list[Index],
                spins: str = None) -> list[Index]:
    """
    Uses the Indices class to initialize 'Index' instances with the
    provided names and spin.

    Parameters
    ----------
    idx : str | Index | list[str] | list[Index]
        The names of the indices to generate. If they are already instances
        of the 'Index' class we do nothing.
    spins : str, optional
        The spin of the indices.
    """

    if not idx:  # empty string/list
        return []
    elif isinstance(idx, Index):  # a single symbol is not iterable
        return [idx]
    elif all(isinstance(i, Index) for i in idx):
        return idx
    elif all(isinstance(i, str) for i in idx):
        symbols = Indices().get_indices("".join(idx), spins)
        for val in symbols.values():
            val.reverse()
        # idx and spins have to be compatible at this point
        idx = split_idx_string(idx)
        if spins is None:
            return [
                symbols[index_space(i)].pop() for i in idx
            ]
        else:
            return [
                symbols[f"{index_space(i)}_{spin}"].pop()
                for i, spin in zip(idx, spins)
            ]
    else:
        raise Inputerror("Indices need to be provided as string or a list "
                         f"of {Index} objects.")


def order_substitutions(subsdict: dict[Index, Index]) -> list:
    """
    Order substitutions such that only a minial amount of intermediate
    indices is required when the substitutions are executed one after another
    and the usage of the 'simultaneous=True' option of sympys 'subs' method.
    Adapted from the 'substitute_dummies' function defined in
    'sympy.physics.seconquant'.
    """
    from .sympy_objects import Index

    subs = []
    final_subs = []
    for o, n in subsdict.items():
        if o is n:  # indices are identical -> nothing to do
            continue
        # the new index is substituted by another index
        if (other_n := subsdict.get(n, None)) is not None:
            if other_n in subsdict:
                # i -> j / j -> i
                # temporary variable is needed
                p = Index('p')
                subs.append((o, p))
                final_subs.append((p, n))
            else:
                # i -> j / j -> k
                # in this case it is sufficient to do the i -> j substitution
                # after the j -> k substitution, but before temporary variables
                # are resubstituted again.
                final_subs.insert(0, (o, n))
        else:
            subs.append((o, n))
    subs.extend(final_subs)
    return subs


def minimize_tensor_indices(tensor_indices: tuple[Index],
                            target_idx: dict[tuple, list[str]]):
    """
    Minimizes the indices on a tensor using the lowest available indices that
    are no target indices.

    Parameters
    ----------
    tensor_indices : tuple[Index]
        List containing the indices of the tensor.
    target_idx : dict[tuple, list[str]]
        The names of target indices sorted by their space and spin with
        key = (space, spin).

    Returns
    -------
    tuple
        The minimized indices and the list of index permutations that was
        applied to reach this minimized state.
    """
    from .symmetry import Permutation, PermutationProduct

    for target in target_idx.values():
        if not all(isinstance(s, str) for s in target):
            raise TypeError("Target indices need to be provided as string.")

    tensor_indices: list[Index] = list(tensor_indices)
    n_unique_indices: int = len(set(tensor_indices))
    minimal_indices: dict[str, list] = {}
    permutations = []  # list for collecting the applied permutations
    minimized = set()
    for s in tensor_indices:
        if s in minimized:
            continue
        space = s.space
        # target indices of the corresponding space
        space_target = target_idx.get(space, [])
        # index is a target idx -> keep as is
        if s.name in space_target:
            minimized.add(s)
            continue
        # generate minimal indices for the corresponding space
        if space not in minimal_indices:
            minimal_indices[space] = get_symbols(
                get_lowest_avail_indices(n_unique_indices, space_target, space)
            )
        # get the lowest available index for the corresponding space
        min_s = minimal_indices[space].pop(0)
        minimized.add(min_s)
        if s is min_s:  # s is already the lowest available index
            continue
        # found a lower index
        # -> permute tensor indices and append permutation to permutations
        #    list
        perm = {s: min_s, min_s: s}
        for i, other_s in enumerate(tensor_indices):
            tensor_indices[i] = perm.get(other_s, other_s)
        permutations.append(Permutation(s, min_s))
    return tuple(tensor_indices), PermutationProduct(permutations)
