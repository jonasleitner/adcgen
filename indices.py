import sympy as sy
from sympy import symbols, Dummy, Add, Mul, S

pretty_indices = {
    "below": "ijklmno",
    "above": "abcdefg",
    "general": "pqrs"
}


class indices:
    """Book keeping class that keeps track of the used and available indices.
       Necessary, because only one instance of Dummy symbol should be used for
       each index, e.g. in each expression the same symbol
       i = symbols('i', below_fermi=True, cls=Dummy)
       should be used. This way sympy recognizes that all i in the expressions
       are equal to each other, which allows for simplifications.
       """
    def __init__(self):
        # dict {'occ': [idx]}
        self.available_indices = {}

        # {case: {x: {occ: []}}}
        # the ground state has the splitting according to bra/ket
        # ISR is splitted according to the idxstring of the parent
        # ISR/Precursor state that requests the indices
        # specific indices (that have been requested specifically to e.g.)
        # construct a specific matrix element are stored like
        # {case: {ov: []}}
        self.used_indices = {}

        # list with possible indices
        # only indices of type i3/a3 etc are used for ground state and ISR
        self.generic_occ = []
        self.generic_virt = []
        # self.occ/virt hold indices that are exclusively available for
        # specific indice request
        self.occ = ['i', 'j', 'k', 'l', 'm', 'n', 'o',
                    'i1', 'j1', 'k1', 'l1', 'm1', 'n1', 'o1',
                    'i2', 'j2', 'k2', 'l2', 'm2', 'n2', 'o2']
        self.virt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                     'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
                     'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']
        self.__setup()

    def __setup(self):
        """Setting up available and used list for gs and isr indices
           """

        # not copying the generic lists, because gs and isr share the
        # same generic indice pool.
        self.available_indices["gs"] = {}
        self.available_indices["gs"]['occ'] = self.generic_occ
        self.available_indices["gs"]['virt'] = self.generic_virt
        self.used_indices["gs"] = {}
        for braket in ["bra", "ket"]:
            self.used_indices["gs"][braket] = {"occ": [], "virt": []}

        # stored with the idxstring of the parent ISR/Precursor state
        # that requests the indices
        self.used_indices["isr"] = {}
        self.available_indices["isr"] = {}
        self.available_indices["isr"]["occ"] = self.generic_occ
        self.available_indices["isr"]["virt"] = self.generic_virt

        # used to store all indices that have been requested specificly
        # However, atm all indices are stored in the specific list, since
        # essentially only get_gs/isr_indices are called to get more
        # indices, while get_indices is only used to get previously
        # created indices or the target indices of the desired state/
        # matrix element
        self.used_indices["specific"] = {'occ': [], 'virt': []}
        self.available_indices["specific"] = {'occ': self.occ,
                                              'virt': self.virt}

        self.available_indices["new"] = {}
        self.available_indices["new"]["occ"] = self.generic_occ
        self.available_indices["new"]["virt"] = self.generic_virt

    def __gen_generic_indices(self, ov):
        """Generates the next 'generation' of indices of the form i3/a3.
           The integer will be inkremented by one for the next chunk of
           indices.
           The generated indices are added to the self.generic_occ/virt,
           the self.occ/virt list and the available lists of all invoked
           spaces.
           """

        if not hasattr(self, "counter_occ"):
            self.counter_occ = 3
            self.counter_virt = 3

        list_to_fill = {
            'occ': self.generic_occ,
            'virt': self.generic_virt
        }
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

        list_to_fill[ov].extend(new_indices)
        self.available_indices["specific"][ov].extend(new_indices)
        setattr(self, "counter_" + ov, counter[ov] + 1)

    def get_gs_indices(self, braket, **kwargs):
        """Returns indices as sympy symbols in form of a dict, sorted
           by the keys 'occ' and 'virt'.
           Ground state indices are removed from the default self.occ/virt
           list and every invoked space. So no space will have access to them.
           The obtained indices are added to the 'gs' used lists as symbols.
           Indices obtained from the generic indices lists.
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
        return self.__get_generic_indices("gs", used, braket=braket, **kwargs)

    def get_isr_indices(self, pre_indices, **kwargs):
        """Returns indices as sympy symbols in form of a dict, sorted by the
           keys 'occ' and 'virt'.
           ISR indices are removed from the default self.occ/virt
           list and every invoked space. So no space will have access to them.
           The obtained indices are added to the used lists of the parent
           precursor state and to the used list of all spaces.
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

        if pre_indices not in self.used_indices["isr"]:
            self.used_indices["isr"][pre_indices] = {'occ': [], 'virt': []}
        used = self.used_indices["isr"][pre_indices]
        return self.__get_generic_indices(
            "isr", used, pre_indices=pre_indices, **kwargs
            )

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
        ret = {}
        for n_ov, n in kwargs.items():
            ov = get_ov[n_ov]
            get_default = {
                "occ": self.generic_occ,
                "virt": self.generic_virt
            }
            while n > len(get_default[ov]):
                self.__gen_generic_indices(ov)
            idx = get_default[ov][:n].copy()
            ret[ov] = [self.__get_new_symbol("new", ov, idxstr)
                       for idxstr in idx]
        return ret

    def __get_generic_indices(self, case, used, braket=None,
                              pre_indices=None, **kwargs):
        """Obtaine a certain number of indices from the generic lists.
           Used for gs and isr indices.
           """

        if case not in ["gs", "isr"]:
            print("Only possible to obtain generic indices for 'gs' or 'isr'",
                  f"The case {case} is not valid.")
            exit()

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
                get_default = {
                    "occ": self.generic_occ,
                    "virt": self.generic_virt
                }
                while needed > len(get_default[ov]):
                    self.__gen_generic_indices(ov)
                idx = get_default[ov][:needed].copy()
                ret[ov].extend(
                    [self.__get_new_symbol(case, ov, idxstr, braket=braket,
                     pre_indices=pre_indices) for idxstr in idx]
                )
        return ret

    def get_indices(self, indices):
        """Returns indices as sympy symbols in form of a dict, sorted
           by the keys 'occ' and 'virt'.
           New indices are taken from the specific available list
           The symbols are safed in the specific used list
           and reused if requested again. That way sympy recognizes symbols
           with the same name as equal.
           """

        if not isinstance(indices, str):
            print("Requested indices need to be of type str (e.g. 'ai'), not",
                  type(indices))
            exit()

        separated = split_idxstring(indices)

        ret = {}
        used = self.used_indices["specific"]
        for idx in separated:
            ov = self.assign_index(idx)
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
                    self.__get_new_symbol("specific", ov, idx)
                    )
        return ret

    def __get_new_symbol(self, case, ov, idx, braket=None, pre_indices=None):
        """Returns a new symbol from the available list. Also calls the remove
           method that removes the indice from the available list and appends
           the symbol to the used list.
           """

        if not self.available_indices[case][ov]:
            print(f"No indices for case {ov} {case} available anymore.")
            exit()
        if idx not in self.available_indices[case][ov]:
            print(f"Could not find {ov} index {idx} in available indices",
                  f"for case {case}.")
            exit()
        symbol = self.__make_symbol_new(ov, idx)
        self.remove(case, braket, pre_indices, ov, symbol)
        return symbol

    def __make_symbol_new(self, ov, idx):
        if ov == "occ":
            return symbols(idx, below_fermi=True, cls=Dummy)
        elif ov == "virt":
            return symbols(idx, above_fermi=True, cls=Dummy)

    def remove(self, case, braket, pre_indices, ov, symbol):
        """Removes index from available and add symbols to the list of
           used indices."""

        if not isinstance(symbol, sy.core.symbol.Dummy):
            print("Index that is to be removed from the available list",
                  f"needs to be a sympy symbol. Not type {type(symbol)}")
            exit()
        if case not in self.available_indices:
            print(f"Space is not recognized. Can't remove from space {case}.",
                  f"Valid cases are {list(self.available_indices.keys())}")
            exit()

        idx = symbol.name
        available = self.available_indices[case][ov]
        if case == "gs":
            used = self.used_indices[case][braket][ov]
            self.used_indices["specific"][ov].append(symbol)
            self.available_indices["specific"][ov].remove(idx)
        elif case == "isr":
            used = self.used_indices[case][pre_indices][ov]
            self.used_indices["specific"][ov].append(symbol)
            self.available_indices["specific"][ov].remove(idx)
        elif case == "new":
            used = self.used_indices["specific"][ov]
        elif case == "specific":
            used = self.used_indices[case][ov]
        available.remove(idx)
        used.append(symbol)

    def assign_index(self, idx):
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
        if not isinstance(expr, Mul):
            print("Expression to substitute should be of type Mul, not"
                  f"{type(expr)}.")
            exit()
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
            ov = self.assign_index(idx)
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
            ov = self.assign_index(name)
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
