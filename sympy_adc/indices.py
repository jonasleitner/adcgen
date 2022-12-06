from sympy import symbols, Dummy
from .misc import Inputerror, Singleton


class indices(metaclass=Singleton):
    """Book keeping class that manages the indices in an expression.
       This ensures that for every index only a single instance exists,
       which allows to correctly identify equal indices.
       i1 = symbols('i', below_fermi=True, cls=Dummy)
       i2 = symbols('i', below_fermi=True, cls=Dummy)
       i1 == i2
       False
       """
    def __init__(self):
        # dict that holds all symbols that have been created previously.
        self.symbols = {'occ': [], 'virt': [], 'general': []}
        # dict that holds the generic indices. Automatically filled by
        # generated index strings.
        self.generic_indices = {'occ': [], 'virt': [], 'general': []}
        # o/v indices that are exclusively available for direct request via
        # get_indices, i.e. they can't be generic indices.
        self.occ = ['i', 'j', 'k', 'l', 'm', 'n', 'o',
                    'i1', 'j1', 'k1', 'l1', 'm1', 'n1', 'o1',
                    'i2', 'j2', 'k2', 'l2', 'm2', 'n2', 'o2']
        self.virt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                     'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
                     'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']
        self.general = ['p', 'q', 'r', 's', 't', 'u', 'v', 'w']

    @property
    def used_names(self):
        return {'occ': set(s.name for s in self.symbols['occ']),
                'virt': set(s.name for s in self.symbols['virt']),
                'general': set(s.name for s in self.symbols['general'])}

    def __gen_generic_idx(self, ov):
        """Generated the next 'generation' of generic indices, i.e. extends
           generic indice list incrementing the integer attached to the index
           base. The first generic indices will increment the integer found
           at the last element of self.o/self.v by one.
           """
        # first call -> counter has not been initialized yet.
        if not hasattr(self, 'counter_' + ov):
            idx_list = sorted(
                getattr(self, ov), key=lambda idx:
                (int(idx[1:]) if idx[1:].isdigit() else 0, idx[0])
            )
            specific = int(idx_list[-1][1:]) \
                if idx_list[-1][1:].isdigit() else 0
            setattr(self, 'counter_' + ov, specific + 1)

        # generate the new index strings
        counter = getattr(self, 'counter_' + ov)
        idx_base = {'occ': ['i', 'j', 'k', 'l', 'm', 'n', 'o'],
                    'virt': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                    'general': ['p', 'q', 'r', 's', 't', 'u', 'v', 'w']}
        used = self.used_names[ov]
        new_idx = [idx + str(counter) for idx in idx_base[ov]
                   if idx + str(counter) not in used]

        # extend the generic list and increment the counter
        self.generic_indices[ov].extend(new_idx)
        setattr(self, 'counter_' + ov, counter + 1)

    def get_indices(self, indices):
        """Obtain the symbols for the provided index string."""
        if not isinstance(indices, str):
            raise Inputerror(f"Indices {indices} need to be of type string.")

        # split the string in the individual indices
        splitted = split_idx_string(indices)

        ret = {}
        for idx in splitted:
            ov = index_space(idx)
            if ov not in ret:
                ret[ov] = []
            # check whether the symbol is already available
            found = False
            for s in self.symbols[ov]:
                if idx == s.name:
                    ret[ov].append(s)
                    found = True
                    break
            # did not find -> create a new symbol + add it to self.symbols
            # and if it is a generic index -> remove it from the current
            # generic index list.
            if not found:
                s = self.__new_symbol(idx, ov)
                ret[ov].append(s)
                self.symbols[ov].append(s)
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
        if ov == 'occ':
            return symbols(idx, below_fermi=True, cls=Dummy)
        elif ov == 'virt':
            return symbols(idx, above_fermi=True, cls=Dummy)
        elif ov == 'general':
            return symbols(idx, cls=Dummy)

    def substitute(self, expr):
        """Substitute all contracted indices in an expression with new,
           'earlier' indices. The function essentially uses all indices
           in [i,j,k,l,m,n,o,i1...] from the beginning that are no target
           indices."""
        import sympy_adc.expr_container as e
        # option 1: leave all target indices untouched -> use container to
        #           find them
        # option 2: leave all indices that are for sure not specific untouched
        #            should be an equivalent solution, but less general

        def substitute_contracted(term):
            # sort all target indices in a dict. Those indices will not be
            # substituted and are therefore, already present in the term
            # (avoid renaming another index to a target index name)
            used = {'occ': [], 'virt': [], 'general': []}
            for s in term.target:
                ov = index_space(s.name)
                used[ov].append(s.name)
            # now iterate over the contracted indices and find the first
            # missing index that may replace the contracted index
            sub = {}
            for s in term.contracted:
                ov = index_space(s.name)
                new_str = get_first_missing_index(used[ov], ov)
                new = self.get_indices(new_str)[ov][0]
                sub[s] = new
                used[ov].append(new_str)
            return term.subs(sub, simultaneous=True)

        expr = expr.expand()
        if not isinstance(expr, e.expr):
            expr = e.expr(expr)
        # iterate over terms and substitute all contracted indices
        substituted = e.compatible_int(0)
        for term in expr.terms:
            substituted += substitute_contracted(term)
        return substituted

    def substitute_with_generic(self, expr):
        """Substitute all contracted indices with new, generic indices."""
        import sympy_adc.expr_container as e

        def substitute_contracted(term):
            contracted = sorted(
                term.contracted, key=lambda s:
                (int(s.name[1:]) if s.name[1:] else 0, s.name[0])
            )
            # count how many indices need to be replaced and get new indices
            old = {}
            for s in contracted:
                ov = index_space(s.name)
                if ov not in old:
                    old[ov] = []
                old[ov].append(s)
            n_ov = {'occ': 'n_o', 'virt': 'n_v', 'general': 'n_g'}
            kwargs = {n_ov[ov]: len(sym_list) for ov, sym_list in old.items()}
            new = self.get_generic_indices(**kwargs)

            # match the old: new pairs and substitute the term
            sub = {}
            for ov, sym_list in old.items():
                if len(sym_list) != len(new[ov]):
                    raise RuntimeError(f"{len(sym_list)} {ov} indices needed "
                                       f"but got {len(new[ov])} new indices.")
                for i, s in enumerate(sym_list):
                    sub[s] = new[ov][i]
            return term.subs(sub, simultaneous=True)

        expr = expr.expand()
        if not isinstance(expr, e.expr):
            expr = e.expr(expr)
        substituted = e.compatible_int(0)
        for term in expr.terms:
            substituted += substitute_contracted(term)
        return substituted


def index_space(idx):
    """Returns the space an index belongs to (occ/virt/geneal)."""
    if idx[0] in {"i", "j", "k", "l", "m", "n", "o"}:
        return 'occ'
    elif idx[0] in {"a", "b", "c", "d", "e", "f", "g", "h"}:
        return 'virt'
    elif idx[0] in {"p", "q", "r", "s", "t", "u", "v", "w"}:
        return 'general'
    else:
        raise Inputerror(f"Could not assign the index {idx} to a space.")


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


def repeated_indices(idx_a, idx_b):
    """Checks whether both index strings share an index."""
    split_a = split_idx_string(idx_a)
    split_b = split_idx_string(idx_b)
    return any(i in split_b for i in split_a)


def get_first_missing_index(idx_list, ov):
    """Returns the first index that is missing in the provided index list.
       The indices need to be provided as strings.
       [i,j,l] -> return k // [i,j,k] -> return l
        """

    idx_base = {'occ': ['i', 'j', 'k', 'l', 'm', 'n', 'o'],
                'virt': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                'general': ['p', 'q', 'r', 's', 't', 'u', 'v', 'w']}
    ordered_idx = idx_base[ov].copy()
    # keep sorting in this function!!!
    idx_list = sorted(
        idx_list, key=lambda i: (int(i[1:]) if i[1:] else 0, i[0])
    )
    for i, idx in enumerate(idx_list):
        if idx != ordered_idx[i]:
            return ordered_idx[i]
        if idx[0] in ['o', 'h', 'w']:
            n = int(idx[1:]) + 1 if idx[1:] else 1
            ordered_idx.extend([b[0] + str(n) for b in ordered_idx])
        new = ordered_idx[i+1]
    if not idx_list:
        new = ordered_idx[0]
    return new


def extract_names(syms):
    """Extracts the names of the provided symbols and returns them in a list.
    """
    from itertools import chain
    if isinstance(syms, dict):
        syms = chain.from_iterable(syms.values())
    return [s.name for s in syms]


def get_symbols(idx):
    """Ensure that all provided indices are sympy symbols. If a string of
       indices is provided the corresponding sympy symbols are
       created automatically."""

    if not idx:
        return []
    elif isinstance(idx, Dummy):  # a single symbol is not iterable
        return [idx]
    elif all(isinstance(i, Dummy) for i in idx):
        return idx
    elif all(isinstance(i, str) for i in idx):
        idx_cls = indices()
        idx = split_idx_string(idx)
        return [
            idx_cls.get_indices(i)[index_space(i)][0] for i in idx
        ]
    else:
        raise Inputerror("Indices need to be provided as string or a list "
                         "of sympy Dummy objects.")
