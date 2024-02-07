from sympy import S, Dummy
from .misc import Inputerror, Singleton


# base names for all used indices
idx_base = {'occ': 'ijklmno', 'virt': 'abcdefgh', 'general': 'pqrstuvw'}


class Index(Dummy):
    """Class to represent Indices. Inherits it's behaviour from the sympy
       'Dummy' class, i.e.,
        Index("x") != Index("x").
        Additional functionality:
         - assigning a spin to the variable via the spin keyword ('a'/'b')
    """
    def __new__(cls, name: str = None, dummy_index=None, spin: str = None,
                **assumptions):
        if spin is None:
            return super().__new__(cls, name, dummy_index, **assumptions)
        elif spin == "a":
            return super().__new__(cls, name, dummy_index, alpha=True,
                                   **assumptions)
        elif spin == "b":
            return super().__new__(cls, name, dummy_index, beta=True,
                                   **assumptions)
        else:
            raise Inputerror(f"Invalid spin {spin}. Valid values are 'a' and "
                             "'b'.")

    @property
    def spin(self) -> None | str:
        if self.assumptions0.get("alpha"):
            return "a"
        elif self.assumptions0.get("beta"):
            return "b"

    @property
    def space(self) -> str:
        if self.assumptions0.get("below_fermi"):
            return "occ"
        elif self.assumptions0.get("above_fermi"):
            return "virt"
        else:
            return "general"


class Indices(metaclass=Singleton):
    """Book keeping class that manages the indices in an expression.
       This ensures that for every index only a single instance exists,
       which allows to correctly identify equal indices.
       """
    def __init__(self):
        # dict that holds all symbols that have been created previously.
        self._symbols = {'occ': {}, 'virt': {}, 'general': {}}
        # dict that holds the generic indices. Automatically filled by
        # generated index strings.
        self.generic_indices = {'occ': [], 'virt': [], 'general': []}
        # o/v indices that are exclusively available for direct request via
        # get_indices, i.e. they can't be generic indices.
        self._occ = ('i', 'j', 'k', 'l', 'm', 'n', 'o',
                     'i1', 'j1', 'k1', 'l1', 'm1', 'n1', 'o1',
                     'i2', 'j2', 'k2', 'l2', 'm2', 'n2', 'o2')
        self._virt = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                      'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
                      'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2')
        self._general = ('p', 'q', 'r', 's', 't', 'u', 'v', 'w')

    def __gen_generic_idx(self, ov):
        """Generated the next 'generation' of generic indices, i.e. extends
           generic indice list incrementing the integer attached to the index
           base. The first generic indices will increment the integer found
           at the last element of self.o/self.v by one.
           """
        # first call -> counter has not been initialized yet.
        if not hasattr(self, f'counter_{ov}'):
            idx_list = sorted(
                getattr(self, f'_{ov}'), key=lambda idx:
                (int(idx[1:]) if idx[1:].isdigit() else 0, idx[0])
            )
            specific = int(idx_list[-1][1:]) \
                if idx_list[-1][1:].isdigit() else 0
            setattr(self, 'counter_' + ov, specific + 1)

        # generate the new index strings
        counter = getattr(self, f'counter_{ov}')
        used = self._symbols[ov]
        new_idx = [idx + str(counter) for idx in idx_base[ov]
                   if idx + str(counter) not in used]

        # extend the generic list and increment the counter
        self.generic_indices[ov].extend(new_idx)
        setattr(self, 'counter_' + ov, counter + 1)

    def get_indices(self, indices):
        """Obtain the symbols for the provided index string."""
        if not isinstance(indices, str):
            raise Inputerror(f"Indices {indices} need to be of type string.")

        ret = {}
        for idx in split_idx_string(indices):
            ov = index_space(idx)
            if ov not in ret:
                ret[ov] = []
            # check whether the symbol is already available
            if idx in self._symbols[ov]:
                ret[ov].append(self._symbols[ov][idx])
                continue
            # did not find -> create a new symbol + add it to self._symbols
            # and if it is a generic index -> remove it from the current
            # generic index list.
            s = self.__new_symbol(idx, ov)
            ret[ov].append(s)
            self._symbols[ov][idx] = s
            try:
                self.generic_indices[ov].remove(idx)
            except ValueError:
                pass
        return ret

    def get_generic_indices(self, **kwargs):
        """Method to request a number of genrated indices that has not been
           used yet. Indices obtained via this function may be reobtained using
           get_indices. Request indices with: n_occ=3, n_virt=2"""
        valid = {'n_occ': 'occ', 'n_o': 'occ', 'n_virt': 'virt', 'n_v': 'virt',
                 'n_general': 'general', 'n_g': 'general'}
        if not all(var in valid and isinstance(n, int)
                   for var, n in kwargs.items()):
            raise Inputerror(f"{kwargs} is not a valid input. Use e.g. "
                             "n_occ=2")
        ret = {}
        for n_ov, n in kwargs.items():
            if n == 0:
                continue
            ov = valid[n_ov]
            # generate new index strings until enough are available
            while n > len(self.generic_indices[ov]):
                self.__gen_generic_idx(ov)
            idx = "".join(self.generic_indices[ov][:n])
            ret[ov] = self.get_indices(idx)[ov]
        return ret

    def __new_symbol(self, idx, ov):
        """Creates the new symbol from the index string."""
        from .sympy_objects import Index

        if ov == 'occ':
            return Index(idx, below_fermi=True)
        elif ov == 'virt':
            return Index(idx, above_fermi=True)
        elif ov == 'general':
            return Index(idx)

    def substitute_with_generic(self, expr):
        """Substitute all contracted indices with new, generic indices."""
        from . import expr_container as e

        def substitute_contracted(term: e.Term) -> e.Expr:
            # count how many indices need to be replaced and get new indices
            old = {}
            for s in term.contracted:
                ov = s.space
                if ov not in old:
                    old[ov] = []
                old[ov].append(s)

            if not old:
                return term

            n_ov = {'occ': 'n_o', 'virt': 'n_v', 'general': 'n_g'}
            kwargs = {n_ov[ov]: len(sym_list) for ov, sym_list in old.items()}
            new = self.get_generic_indices(**kwargs)

            # match the old: new pairs and substitute the term
            sub = {}
            for ov, sym_list in old.items():
                if len(sym_list) != len(new[ov]):
                    raise RuntimeError(f"{len(sym_list)} {ov} indices needed "
                                       f"but got {len(new[ov])} new indices.")
                sub.update({s: new_s for s, new_s in zip(sym_list, new[ov])})
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
    """Returns the space an index belongs to (occ/virt/geneal)."""
    for sp, idx_string in idx_base.items():
        if idx[0] in idx_string:
            return sp
    raise Inputerror(f"Could not assign the index {idx} to a space.")


def idx_sort_key(s):
    return (int(s.name[1:]) if s.name[1:] else 0, s.name[0])


def split_idx_string(str_tosplit):
    """Splits an index string of the form ij12a3b in a list [i,j12,a3,b]."""
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


def n_ov_from_space(space_str):
    return {'n_occ': space_str.count('h'), 'n_virt': space_str.count('p')}


def repeated_indices(idx_a: str, idx_b: str) -> bool:
    """Checks whether both index strings share an index."""
    split_a = split_idx_string(idx_a)
    split_b = split_idx_string(idx_b)
    return any(i in split_b for i in split_a)


def get_lowest_avail_indices(n: int, used: list[str], space: str) -> list[str]:
    """Returns a list containing the n lowest indices that belong to the
       desired space and are not present in the provided list of used indices.
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


def extract_names(syms):
    """Extracts the names of the provided symbols and returns them in a list.
    """
    from itertools import chain
    if isinstance(syms, dict):
        syms = chain.from_iterable(syms.values())
    return [s.name for s in syms]


def get_symbols(idx: str | list[str] | list[Index]) -> list[Index]:
    """Ensure that all provided indices are sympy symbols. If a string of
       indices is provided the corresponding sympy symbols are
       created automatically."""

    if not idx:
        return []
    elif isinstance(idx, Index):  # a single symbol is not iterable
        return [idx]
    elif all(isinstance(i, Index) for i in idx):
        return idx
    elif all(isinstance(i, str) for i in idx):
        idx_cls = Indices()
        idx = split_idx_string(idx)
        return [
            idx_cls.get_indices(i)[index_space(i)][0] for i in idx
        ]
    else:
        raise Inputerror("Indices need to be provided as string or a list "
                         f"of {Index} objects.")


def order_substitutions(subsdict: dict[Index, Index]) -> list:
    """Returns substitutions ordered in a way one can use the subs method
       without the need to use the 'sumiltanous=True' option. Essentially
       identical to a part of the substitute_dummies function of sympy."""
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


def minimize_tensor_indices(tensor_indices: tuple,
                            target_idx: dict[str, list[str]]):
    """Minimizes the tensor indices using the lowest available non target
       indices. Returns the minimized indices as well as the corresponding
       substitution dict."""
    from .symmetry import Permutation, PermutationProduct

    for target in target_idx.values():
        if not all(isinstance(s, str) for s in target):
            raise TypeError("Target indices need to be provided as string.")

    tensor_indices: list = list(tensor_indices)
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
