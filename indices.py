import sympy as sy
from sympy import symbols, Dummy, Add, S
from sympy.physics.secondquant import substitute_dummies

pretty_indices = {
    "below": "ijklmno",
    "above": "abcdefg",
    "general": "pqrs"
}


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
        occ = ['i', 'j', 'k', 'l', 'm', 'n', 'o',
               'i1', 'j1', 'k1', 'l1', 'm1', 'n1', 'o1',
               'i2', 'j2', 'k2', 'l2', 'm2', 'n2', 'o2']
        virt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
                'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']
        self.specific_indices = {"occ": occ, "virt": virt}
        self.__setup_used()

    def __setup_used(self):
        """Setting up available and used list for gs and isr indices
           """

        self.used_indices["gs"] = {}
        for braket in ["bra", "ket"]:
            self.used_indices["gs"][braket] = {"occ": [], "virt": []}

        self.used_indices["isr"] = {}

        # all indices that are used in any case are stored in this list
        self.used_indices["specific"] = {'occ': [], 'virt': []}

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
            if self.specific_indices["occ"][-1][0] in ["o", "h"]:
                i = int(self.specific_indices[ov][-1][1:]) + 1
            else:
                i = int(self.specific_indices[ov][-1][1:]) + 2
            self.counter_occ = i
            self.counter_virt = i

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
                print(f"{key} is not a valid option for requesting gs",
                      "indices.")
                exit()
            if not isinstance(value, int):
                print("Number of requested gs indices must be int, not",
                      type(value))
                exit()

        used = self.used_indices["gs"][braket]
        return self.__get_generic_indices(used, **kwargs)

    def get_isr_indices(self, parent_indices, **kwargs):
        """Returns indices as sympy symbols in form of a dict, sorted by the
           keys 'occ' and 'virt'.
           """

        valid = ['n_occ', 'n_virt']
        for key, value in kwargs.items():
            if key not in valid:
                print(f"{key} is not a valid option for requesting gs",
                      "indices.")
                exit()
            if not isinstance(value, int):
                print("Number of requested gs indices must be int, not",
                      type(value))
                exit()

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
                print(f"{key} is not a valid option for requesting new",
                      "generic indices.")
                exit()
            if not isinstance(value, int):
                print("Number of requested generic indices must be int, not",
                      type(value))
                exit()

        # no need to check any used list, just create new generic indices
        used = self.used_indices["specific"]
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
            print("Requested indices need to be of type str (e.g. 'ai'), not",
                  type(indices))
            exit()

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
            print(f"No {ov} indices for case {case} available anymore.")
            exit()
        if idx not in available[ov]:
            print(f"Could not find {ov} index {idx} in available indices"
                  f"for case {case}.")
            exit()

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
            print("Index that is to be removed from the available list",
                  f"needs to be a sympy symbol. Not type {type(symbol)}")
            exit()
        if case not in {"generic", "specific"}:
            print(f"Case is not recognized. Can't remove for case {case}."
                  f"Valid cases are 'generic' and 'specific'.")
            exit()

        idx = symbol.name
        available = (self.specific_indices[ov] if case == "specific" else
                     self.generic_indices[ov])

        available.remove(idx)
        used.append(symbol)
        if case == "generic" and symbol not in \
                self.used_indices["specific"][ov]:
            self.used_indices["specific"][ov].append(symbol)

    def substitute_indices(self, expr):
        """Substitutes the indices in an expression.
           To this end, the original indices are sorted like
           a,b,a1,b2,a3,b3... and then substituted in this order
           with the indices a,b,c,..., a1,b1,...
           The same is done for the occ indices i,j,...
           """

        if isinstance(expr, Add):
            return Add(*[self.substitute_indices(term) for term in expr.args])
        elif expr is S.Zero:
            return expr
        idx_order = {'occ': ['i', 'j', 'k', 'l', 'm', 'n', 'o'],
                     'virt': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']}
        idx_in_use = {'occ': [], 'virt': []}
        # sort the original indices as
        # [a, b, a1, b1, a3, b3]
        original_idx = list(expr.atoms(Dummy))
        original_name = sorted(
            [s.name for s in original_idx],
            key=lambda idx: (int(idx[1:]) if len(idx) > 1 else 0, idx[0])
        )
        substitute = []
        for idx in original_name:
            ov = assign_index(idx)
            if idx_in_use[ov]:
                last = idx_in_use[ov][-1]
                last_num = last[1:]
                i = idx_order[ov].index(last[0])
                if i+1 < len(idx_order[ov]):
                    new = idx_order[ov][i+1] + last_num
                else:
                    new_num = 1 if not last_num else int(last_num) + 1
                    new = idx_order[ov][0] + str(new_num)
            else:
                new = idx_order[ov][0]
            idx_in_use[ov].append(new)
            # rather replace everything... that way for sure only one 'i'
            # is in the result
            # if idx == new:
            #     continue
            new = self.get_indices(new)
            new = new[ov][0]
            for s in original_idx:
                if s.name == idx:
                    j = original_idx.index(s)
            old = original_idx[j]
            substitute.append((old, new))
        return expr.subs(substitute)

    def substitute_with_generic_indices(self, expr):
        """Input: expression wich is already correct, wrt substitution,
           i.e. there should only be one index with name 'i' etc. in the
           expression.
           The indices will be replaced with newly generated generic ones
           to continue calculations with the expression.
           """

        orig_idx = list(expr.atoms(Dummy))
        orig_name = sorted(
            [s.name for s in orig_idx],
            key=lambda idx: (int(idx[1:]) if len(idx) > 1 else 0, idx[0])
        )
        old_idx = {}
        kwargs = {}
        n_ov = {'occ': "n_occ", 'virt': "n_virt"}
        for name in orig_name:
            ov = assign_index(name)
            if ov not in old_idx:
                old_idx[ov] = []
            for symbol in orig_idx:
                if symbol.name == name:
                    old_idx[ov].append(symbol)
            if n_ov[ov] not in kwargs:
                kwargs[n_ov[ov]] = 0
            kwargs[n_ov[ov]] += 1
        new_idx = self.get_new_gen_indices(**kwargs)
        substitute = []
        for ov in old_idx:
            if len(old_idx[ov]) != len(new_idx[ov]):
                print(f"Cannot replace {len(old_idx[ov])} old indices",
                      f"with {len(new_idx[ov])} new generic indices.")
                exit()
            for i in range(len(old_idx[ov])):
                substitute.append((old_idx[ov][i], new_idx[ov][i]))
        return expr.subs(substitute)


def assign_index(idx):
    """Returns wheter an index belongs to the occ/virt space.
        Assumes a naming convention 'ax'/'ix', where 'x' is some
        number.
        """

    if idx[0] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        return "virt"
    elif idx[0] in ["i", "j", "k", "l", "m", "n", "o"]:
        return "occ"
    else:
        print(f"Could not assign index {idx} to occ or virt.")
        exit()


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
    return repeated


def make_pretty(expr):
    """Funciton that exchanges the indices in an expression with other,
       pretty indices. Not done by default, because it is not possible
       to perform further calculations with the resulting expressions.
        """

    return substitute_dummies(
        expr, new_indices=True, pretty_indices=pretty_indices)
