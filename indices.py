import sympy as sy
from sympy import symbols, Dummy, Mul, Add


pretty_indices = {
    "below": "ijklmno",
    "above": "abcdefg",
    "general": "pqrs"
}


class indices:
    """Book keeping class that keeps track of the used and available indices.
       Each space (ph, pphh, etc.) have their own list of available indices.
       Indices that are used for generating groundstate wavefunctions are
       shared over all spaces, so they can be used for any space without
       having to change indices.
       Indices may be obtained by calling get_indices by providing the space
       and the number of occ and virt indices. Indices are automatically
       removed from the relevant lists, when calling get_indices.
       If run out of indices: just add more to self.occ or self.virt lists.

       If a new space is build, create_space has to be called for the
       respective space."""
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
        # 'o1' reserved for Hamiltonian
        self.occ = ['i', 'j', 'k', 'l', 'm', 'n', 'o',
                    'i1', 'j1', 'k1', 'l1', 'm1', 'n1']
        self.virt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                     'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1']
        # self.invoked_spaces = []
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
        # list_to_fill = {
        #     'occ': (self.generic_occ, self.occ),
        #     'virt': (self.generic_virt, self.virt)
        # }
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
        # list_to_fill[ov][0].extend(new_indices)
        # list_to_fill[ov][1].extend(new_indices)
        # for space in self.invoked_spaces:
        #     self.available_indices[space][ov].extend(new_indices)
        list_to_fill[ov].extend(new_indices)
        self.available_indices["specific"][ov].extend(new_indices)
        setattr(self, "counter_" + ov, counter[ov] + 1)

    def __set_default(self, space):
        # TODO: remove
        self.available_indices[space] = {}
        self.available_indices[space]['occ'] = self.occ.copy()
        self.available_indices[space]['virt'] = self.virt.copy()

    def invoke_space(self, space):
        # TODO: remove
        """Setup the used and available indice dicts for the respective space.
           """

        if not isinstance(space, str):
            print(f"Space need to be of type str, not of {type(space)}")
            exit()
        # better check this from isr side?
        for letter in space:
            if letter not in ["p", "h"]:
                print("Space need to be of the form 'ph', 'pphh' etc.",
                      f"{space} is not valid.")
                exit()
        if space in self.available_indices:
            print("Space has already been created. Cannot create twice.")
            exit()
        else:
            self.__set_default(space)
            self.invoked_spaces.append(space)
            self.used_indices[space] = {}
            t_occ = []
            t_virt = []
            for dict in self.used_indices["isr"].values():
                t_occ.extend(dict['occ'])
                t_virt.extend(dict['virt'])
            self.used_indices[space]['occ'] = t_occ
            self.used_indices[space]['virt'] = t_virt

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

    def __get_generic_indices_old(self, case, used, braket=None,
                                  pre_indices=None, **kwargs):
        """Obtaine a certain number of indices from the generic lists.
           Used for gs and isr indices.
           """

        if case not in ["gs", "isr"]:
            print("Only possible to obtain generic indices for 'gs' or 'isr'")
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
                if not self.invoked_spaces:
                    # no spaces inoked yet -> grab from self.generic_occ/virt
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
                else:
                    # some spaces invoked yet -> grab from smallest space,
                    # but only indices of of type i1, j1...
                    common = self.__get_common_indices(ov)
                    idx = [av for av in common if len(av) > 1 and
                           int(av[1]) > 2][:needed]
                    while needed > len(idx):
                        self.__gen_generic_indices(ov)
                        common = self.__get_common_indices(ov)
                        idx = [av for av in common if len(av) > 1 and
                               int(av[1]) > 2][:needed]
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

    def get_indices_old(self, space, indices):
        """Returns indices as sympy symbols in form of a dict, sorted
           by the keys 'occ' and 'virt'.
           New indices are taken from the available list of the respective
           space.
           The symbols are safed in the used lists of the respective space
           and reused if requested again. That way sympy recognizes symbols
           with the same name as equal.
           """

        if not isinstance(space, str):
            print(f"Trying to get indices with space of type {type(space)}",
                  "Only string allowed.")
            exit()
        if space == "gs":
            print("Indices for the ground state should be obtained with",
                  "get_gs_indices instead.")
            exit()
        if not isinstance(indices, str):
            print("Requested indices need to be of type str (e.g. 'ai'), not",
                  type(indices))
            exit()

        separated = split_idxstring(indices)

        ret = {}
        used = self.used_indices[space]
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
                    self.__get_new_symbol(space, ov, idx)
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

    def __get_new_symbol_old(self, space, ov, idx, braket=None,
                             pre_indices=None):
        """Returns a new symbol from the available list. Also calls the remove
           method that removes the indice from the available list and appends
           the symbol to the used list.
           """

        if not self.available_indices[space][ov]:
            print(f"No indices for space {ov} {space} available anymore.")
            exit()
        if idx not in self.available_indices[space][ov]:
            print(f"Could not find {ov} index {idx} in available indices",
                  f"for space {space}.")
            exit()
        symbol = self.__make_symbol_new(ov, idx)
        self.remove(space, braket, pre_indices, ov, symbol)
        return symbol

    def __get_common_indices(self, ov):
        # Not needed anymore if no spaces are specified
        """Returns a list of all indices that are common to all invoked spaces.
           """

        get_dict = {
            "occ": self.occ,
            "virt": self.virt
        }
        common = []
        for idx in get_dict[ov]:
            indicator = []
            for space in self.available_indices:
                if space != "gs" and idx in self.available_indices[space][ov]:
                    indicator.append(True)
            if all(indicator) and indicator:
                common.append(idx)
        return common

    def __get_smallest(self, ov):
        lengths = {}
        for space, dic in self.available_indices.items():
            if space not in ["gs", "isr"]:
                lengths[space] = len(dic[ov])
        return min(lengths, key=lengths.get)

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
        elif case == "isr":
            used = self.used_indices[case][pre_indices][ov]
            self.used_indices["specific"][ov].append(symbol)
        elif case == "specific":
            used = self.used_indices[case][ov]
        available.remove(idx)
        used.append(symbol)

    def remove_old(self, space, braket, pre_indices, ov, symbol):
        """Removes index from available and add symbols to the list of
           used indices."""

        if not isinstance(symbol, sy.core.symbol.Dummy):
            print("Index that is to be removed from the available list",
                  f"needs to be a sympy symbol. Not type {type(symbol)}")
            exit()
        if space not in self.available_indices and space != "gs":
            print(f"Space is not recognized. Can't remove from space {space}")
            exit()
        idx = symbol.name
        available = self.available_indices[space][ov]
        all_indices = {"occ": self.occ, "virt": self.virt}
        if space == "gs":
            used = self.used_indices[space][braket][ov]
            available.remove(idx)
            all_indices[ov].remove(idx)
            used.append(symbol)
            for other_space in self.invoked_spaces:
                self.available_indices[other_space][ov].remove(idx)
        elif space == "isr":
            used = self.used_indices[space][pre_indices][ov]
            available.remove(idx)
            all_indices[ov].remove(idx)
            used.append(symbol)
            for other_space in self.invoked_spaces:
                self.available_indices[other_space][ov].remove(idx)
                self.used_indices[other_space][ov].append(symbol)
        # remove from ph/pphh space
        else:
            used = self.used_indices[space][ov]
            available.remove(idx)
            used.append(symbol)
        self.used_indices["all"][ov].append(symbol)

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
           Therefore, the original indices are sorted
           a,b,a1,b2,a3,b3... and then substituted in this order
           with the indices a,b,c,..., a1,b1,...
           The same is done for the occ indices i,j,...
           """

        idx_order = {'occ': ['i', 'j', 'k', 'l', 'm', 'n', 'o'],
                     'virt': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']}
        if isinstance(expr, Add):
            return Add(*[self.substitute_indices(term) for term in expr.args])
        elif isinstance(expr, Mul):
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
                        new_num = 1 if not last_num else int(last_num + 1)
                        new = idx_order[ov][0] + str(new_num)
                else:
                    new = idx_order[ov][0]
                idx_in_use[ov].append(new)
                new = self.get_indices(new)
                new = new[ov][0]
                for s in original_idx:
                    if s.name == idx:
                        j = original_idx.index(s)
                old = original_idx[j]
                substitute.append((old, new))
            return expr.subs(substitute)
        else:
            return expr


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
