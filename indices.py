import sympy as sy
from sympy import symbols, Dummy, Add, S
from sympy.physics.secondquant import substitute_dummies
from misc import Inputerror

pretty_indices = {
    "below": "ijklmno",
    "above": "abcdefg",
    "general": "pqrs"
}


# TODO: remove isr_indices
class indices:
    """Book keeping class that keeps track of the used and available indices.
       Necessary, because only 1 Dummy symbols instance should exist for each
       name, i.e. the same symbols instance should be used in each expression.
       This allows sympy to automatically simplify expressions.
       """

    def __init__(self):
        # separated in 'gs', 'isr' and 'specific' used_indices
        # 'gs' is further separated in 'bra'/'ket'
        # 'isr' is further separated by the parent_indices string
        # 'specific' is only separated in 'occ'/'virt'
        # Note that also gs and isr are separated once more according
        # to 'occ'/'virt'
        self.used_indices = {}

        # empty lists that are filled on demand with generic indices that
        # continue the pattern of occ and virt, by incrementing the numbering.
        self.generic_indices = {"occ": [], "virt": []}

        # occ/virt hold indices that are exclusively available for
        # specific indice requests
        self.occ = ['i', 'j', 'k', 'l', 'm', 'n', 'o',
                    'i1', 'j1', 'k1', 'l1', 'm1', 'n1', 'o1',
                    'i2', 'j2', 'k2', 'l2', 'm2', 'n2', 'o2']
        self.virt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                     'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
                     'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']
        self.specific_indices = {"occ": ['o42'] + self.occ.copy(),
                                 "virt": self.virt.copy()}
        self.__setup_used()

    def __setup_used(self):
        """Setting up available and used list for gs and isr indices
           """

        self.used_indices["gs"] = {}
        for braket in ["bra", "ket"]:
            self.used_indices["gs"][braket] = {"occ": [], "virt": []}

        self.used_indices["isr"] = {}

        # just for clarity reasons:
        self.used_indices["new"] = {'occ': [], 'virt': []}

        # all indices that are used in any case are stored in this list
        self.used_indices["specific"] = {'occ': [], 'virt': []}
        # special index that is used for h1 and nowhere else
        # call get indices to get him in the used list for index substitution
        self.get_indices('o42')

    def __gen_generic_indices(self, ov):
        """Generates the next 'generation' of indices of the form i3/a3.
           The integer will be inkremented by one for the next chunk of
           indices.
           The generated indices are added to the generic_indices[occ/virt]
           and the specific_indices[occ/virt] lists.
           """

        # initialise counter depending on the predefined values in occ and virt
        # (see __init__).
        if not hasattr(self, "counter_occ"):
            io = int(self.occ[-1][1:]) + 1 if self.occ[-1][0] == "o" \
                else int(self.occ[-1][1:]) + 2
            iv = int(self.virt[-1][1:]) + 1 if self.occ[-1][0] == "h" \
                else int(self.virt[-1][1:]) + 2
            self.counter_occ = io
            self.counter_virt = iv

        counter = {
            'occ': self.counter_occ,
            'virt': self.counter_virt
        }
        index_base = {
            'occ': ['i', 'j', 'k', 'l', 'm', 'n', 'o'],
            'virt': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        }
        new_indices = []
        for idx in index_base[ov]:
            new_indices.append(idx + str(counter[ov]))
        # this index is used in H1
        if counter[ov] == 42:
            new_indices.remove("o42")

        self.generic_indices[ov].extend(new_indices)
        self.specific_indices[ov].extend(new_indices)
        setattr(self, "counter_" + ov, counter[ov] + 1)

    def get_gs_indices(self, braket, **kwargs):
        """Returns indices as sympy symbols in form of a dict, sorted
           by the keys 'occ' and 'virt'.
           """

        valid = ['n_occ', 'n_virt']
        for key, value in kwargs.items():
            if key not in valid:
                raise Inputerror(f"{key} is not a valid option for requesting",
                                 " gs indices.")
            if not isinstance(value, int):
                raise Inputerror("Number of requested gs indices must be int, "
                                 f"not {type(value)}.")

        used = self.used_indices["gs"][braket]
        return self.__get_generic_indices(used, **kwargs)

    def get_isr_indices(self, parent_indices, **kwargs):
        """Returns indices as sympy symbols in form of a dict, sorted by the
           keys 'occ' and 'virt'.
           """

        valid = ['n_occ', 'n_virt']
        for key, value in kwargs.items():
            if key not in valid:
                raise Inputerror(f"{key} is not a valid option for requesting "
                                 "isr indices.")
            if not isinstance(value, int):
                raise Inputerror("Number of requested gs indices must be int, "
                                 f"not {type(value)}")

        if parent_indices not in self.used_indices["isr"]:
            self.used_indices["isr"][parent_indices] = {'occ': [], 'virt': []}
        used = self.used_indices["isr"][parent_indices]
        return self.__get_generic_indices(used, **kwargs)

    def get_new_gen_indices(self, **kwargs):
        """Returns newly generated generic indices that have not been used
           anywhere yet.
           The indices are stored in the specific used lists so they can be
           reobtained at any point later with get_indices.
           """

        get_ov = {'n_occ': "occ", 'n_virt': "virt"}
        for key, value in kwargs.items():
            if key not in get_ov:
                raise Inputerror(f"{key} is not a valid option for requesting "
                                 "new generic indices.")
            if not isinstance(value, int):
                raise Inputerror("Number of requested generic indices must be "
                                 f"int, not {type(value)}")

        # no need to check any used list, just create new generic indices
        used = self.used_indices["new"]
        ret = {}
        for n_ov, n in kwargs.items():
            ov = get_ov[n_ov]

            while n > len(self.generic_indices[ov]):
                self.__gen_generic_indices(ov)
            idx = self.generic_indices[ov][:n].copy()
            ret[ov] = [self.__get_new_symbol("generic", ov, idxstr, used[ov])
                       for idxstr in idx]
        return ret

    def __get_generic_indices(self, used, **kwargs):
        """Obtaine a certain number of indices from the generic lists.
           Used for gs and isr indices.
           """

        get_ov = {'n_occ': "occ", 'n_virt': "virt"}
        ret = {}
        for n_ov, n in kwargs.items():
            ov = get_ov[n_ov]
            # reuse all symbols
            if n <= len(used[ov]):
                ret[ov] = used[ov][:n]
            # not enough symbols available
            else:
                ret[ov] = used[ov][:n].copy()
                needed = n - len(used[ov])

                while needed > len(self.generic_indices[ov]):
                    self.__gen_generic_indices(ov)
                idx = self.generic_indices[ov][:needed].copy()

                ret[ov].extend(
                    [self.__get_new_symbol("generic", ov, idxstr, used[ov])
                     for idxstr in idx]
                )
        return ret

    def get_indices(self, indices):
        """Returns indices as sympy symbols in form of a dict, sorted
           by the keys 'occ' and 'virt'.
           """

        if not isinstance(indices, str):
            raise Inputerror("Requested indices need to be of type str (e.g. "
                             f"'ai'), not {type(indices)}")

        separated = split_idxstring(indices)

        ret = {}
        used = self.used_indices["specific"]
        for idx in separated:
            ov = assign_index(idx)
            if ov not in ret:
                ret[ov] = []
            # reuse symbol
            found = False
            for symbol in used[ov]:
                if idx == symbol.name:
                    ret[ov].append(symbol)
                    found = True
            # create new symbol
            if not found:
                ret[ov].append(
                    self.__get_new_symbol("specific", ov, idx, used[ov])
                    )
        return ret

    def __get_new_symbol(self, case, ov, idx, used):
        """Returns a new symbol from the available list. Also calls the remove
           method that removes the indice from the available list and appends
           the symbol to the used list.
           """

        available = (self.specific_indices if case == "specific" else
                     self.generic_indices)
        if not available[ov]:
            raise RuntimeError(f"No {ov} indices for case {case} available "
                               "anymore.")
        if idx not in available[ov]:
            raise RuntimeError(f"Could not find {ov} index {idx} in available "
                               f"indices for case {case}.")

        symbol = self.__make_symbol_new(ov, idx)
        self.__remove(case, ov, symbol, used)
        return symbol

    def __make_symbol_new(self, ov, idx):
        if ov == "occ":
            return symbols(idx, below_fermi=True, cls=Dummy)
        elif ov == "virt":
            return symbols(idx, above_fermi=True, cls=Dummy)

    def __remove(self, case, ov, symbol, used):
        """Removes index from available and add symbols to the list of
           used indices."""

        if not isinstance(symbol, sy.core.symbol.Dummy):
            raise Inputerror("Index that is to be removed needs to be a sympy "
                             f"symbol. Not type {type(symbol)}")
        if case not in ["generic", "specific"]:
            raise Inputerror(f"Case is not recognized. Can't remove for case "
                             f"{case}. Valid cases are 'generic' and "
                             "'specific'.")

        idx = symbol.name
        available = (self.specific_indices[ov] if case == "specific" else
                     self.generic_indices[ov])

        available.remove(idx)
        used.append(symbol)
        # attach symbols for gs and isr also to the specific used list
        # (the new_gen_indices are already attached above, because they
        # have the specific used list as used.)
        if case == "generic":
            if symbol in self.used_indices["specific"][ov]:
                raise RuntimeError(f"The symbol {symbol} that was already "
                                   "created and used at some point was "
                                   "attempted to create again, altough it "
                                   "should have been looked up in the used "
                                   "list.")
            self.used_indices["specific"][ov].append(symbol)

    def substitute_indices(self, expr):
        """Substitute the indices in an expression. Leaving the specific indices
           in self.occ and self.virt unchanged.
           This function often not completely simplifies an expression. A few
           terms often may be further simplified by interchanging index names.
           However, it is an alternative that may be used when
           substitute_indices from sympy completely fails.
           """

        expr = expr.expand()
        if isinstance(expr, Add):
            return Add(*[self.substitute_indices(term) for term in expr.args])
        elif expr is S.Zero:
            return expr

        used_idx = {'occ': [], 'virt': []}

        # sort the orignal symbols
        original_sym = list(expr.atoms(Dummy))
        sorted_sym = sorted(
            [s for s in original_sym],
            key=lambda s: (int(s.name[1:]) if len(s.name) > 1
                           else 0, s.name[0])
        )

        sub = {}
        generic = []
        # prescan if we have indices that most likely have been specified by
        # the user
        for s in sorted_sym:
            if s in sub:
                print("FOUND INDEX TWICE!!!!")
                continue
            ov = assign_index(s.name)
            # not do anything with p,q,r,s
            if ov == "general":
                continue
            # catch all generic indices
            ov_list = {"occ": self.occ, "virt": self.virt}
            if s.name not in ov_list[ov]:
                generic.append(s)
                continue

            ref = self.get_indices(s.name)[ov][0]
            if s == ref:
                sub[s] = ref
                used_idx[ov].append(ref.name)
            # catch all indices that have been created by sympy
            # (e.g. multiple 'i' through wicks)
            else:
                generic.append(s)

        # try to substitute the remaining, generic indices
        for s in generic:
            if s in sub:
                print("FOUND GENERIC INDEX TWICE!!!!")
                continue
            ov = assign_index(s.name)

            new = get_first_missing_index(used_idx[ov], ov)
            new_s = self.get_indices(new)[ov][0]

            sub[s] = new_s
            used_idx[ov].append(new_s.name)
        return expr.subs(sub, simultaneous=True)

    def substitute_with_generic_indices(self, expr):
        """Replaces indices in an expression with newly generated
           generic indices that are not in use anywhere else.
           """

        orig_sym = list(expr.atoms(Dummy))
        sorted_sym = sorted(
            [s for s in orig_sym],
            key=lambda s: (int(s.name[1:]) if len(s.name) > 1 else 0,
                           s.name[0])
        )

        old_sym = {}
        # counting how many occ and virt indices are needed
        for s in sorted_sym:
            ov = assign_index(s.name)
            if ov == "general":
                continue
            if ov not in old_sym:
                old_sym[ov] = []
            if s not in old_sym[ov]:
                old_sym[ov].append(s)

        # generate new generic indices for replacing
        n_ov = {'occ': 'n_occ', 'virt': 'n_virt'}
        kwargs = {}
        for ov in old_sym:
            kwargs[n_ov[ov]] = len(old_sym[ov])
        new_sym = self.get_new_gen_indices(**kwargs)

        sub = []
        for ov in old_sym:
            if len(old_sym[ov]) != len(new_sym[ov]):
                raise RuntimeError(f"Cannot replace {len(old_sym[ov])} old ",
                                   f"indices with {len(new_sym[ov])} new "
                                   "generic indices.")
            for i in range(len(old_sym[ov])):
                sub.append((old_sym[ov][i], new_sym[ov][i]))
        return expr.subs(sub, simultaneous=True)


def assign_index(idx):
    """Returns wheter an index belongs to the occ/virt space.
        Assumes a naming convention 'ax'/'ix', where 'x' is some
        number.
        """

    if idx[0] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        return "virt"
    elif idx[0] in ["i", "j", "k", "l", "m", "n", "o"]:
        return "occ"
    elif idx[0] in ["p", "q", "r", "s"]:
        return "general"
    else:
        raise Inputerror(f"Could not assign index {idx} to occ, virt or "
                         "general.")


def split_idxstring(string_tosplit):
    """Splits an index string of the form ij12a3b in a list
       [i,j12,a3,b]
       """

    separated = []
    temp = []
    for i, idx in enumerate(string_tosplit):
        temp.append(idx)
        if i+1 < len(string_tosplit):
            if string_tosplit[i+1].isdigit():
                continue
            else:
                separated.append("".join(temp))
                temp.clear()
        else:
            separated.append("".join(temp))
    return separated


def check_repeated_indices(string_a, string_b):
    """Checks wheter an indices repeat in two index strings."""

    repeated = False
    split_a = split_idxstring(string_a)
    split_b = split_idxstring(string_b)

    for idx in split_a:
        if idx in split_b:
            repeated = True
            break
    return repeated


def get_first_missing_index(idx_list, ov):
    """Returns the first index that is missing in the provided index
       list. If the list is continuous the next index is returned.
       Assumes an ordering like [i,..., o, i1,... o1, i2...].
       """

    idx_order = {'occ': ['i', 'j', 'k', 'l', 'm', 'n', 'o'],
                 'virt': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']}
    sorted_idx = sorted(
        idx_list,
        key=lambda idx: (int(idx[1:]) if len(idx) > 1 else 0, idx[0])
    )
    for i, idx in enumerate(sorted_idx):
        if idx != idx_order[ov][i]:
            return idx_order[ov][i]
        if idx[0] in ['o', 'h']:
            new_n = int(idx[1:]) + 1 if idx[1:] else 1
            idx_order[ov].extend([base[0] + str(new_n)
                                  for base in idx_order[ov]])
        new = idx_order[ov][i+1]
    if not idx_list:
        new = idx_order[ov][0]
    return new


def get_n_ov_from_space(space_str):
    ret = {"n_occ": 0, "n_virt": 0}
    for letter in space_str:
        if letter == "h":
            ret["n_occ"] += 1
        elif letter == "p":
            ret["n_virt"] += 1
        else:
            raise Inputerror(
                f"Invalid letter found in space string {space_str}: {letter}."
            )
    return ret


def make_pretty(expr):
    """Funciton that exchanges the indices in an expression with other,
       pretty indices. Not done by default, because it is not possible
       to perform further calculations with the resulting expressions.
        """

    return substitute_dummies(
        expr, new_indices=True, pretty_indices=pretty_indices)
