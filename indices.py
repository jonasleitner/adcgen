import sympy as sy
from sympy import symbols, Dummy


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
        # dict {'space': {'occ': [idx]}}
        self.available_indices = {}

        # {space: {bra: {occ: []}}}
        # only the ground state has the additional splitting
        # according to bra/ket
        self.used_indices = {}

        # list with possible indices
        # only indices of type i1/a1 etc are used for ground state currently
        self.gs_occ = ['i1', 'j1', 'k1', 'l1', 'm1', 'n1', 'o1', 'i2', 'j2',
                       'k2', 'l2', 'm2', 'n2', 'o2']
        self.gs_virt = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'a2',
                        'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']
        self.occ = ['i', 'j', 'k', 'l', 'm', 'n', 'o'] + self.gs_occ
        self.virt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] + self.gs_virt
        self.invoked_spaces = []
        self.__setup_gs()

    def __setup_gs(self):
        # not copy the default lists, so if removed from available gs
        # it will also be removed from default and new spaces will not
        # be invoked with the respective indices.
        self.available_indices["gs"] = {}
        self.available_indices["gs"]['occ'] = self.gs_occ
        self.available_indices["gs"]['virt'] = self.gs_virt
        self.used_indices["gs"] = {}
        for braket in ["bra", "ket"]:
            self.used_indices["gs"][braket] = {"occ": [], "virt": []}

    def __set_default(self, space):
        self.available_indices[space] = {}
        self.available_indices[space]['occ'] = self.occ.copy()
        self.available_indices[space]['virt'] = self.virt.copy()

    def invoke_space(self, space):
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
            self.used_indices[space] = {"occ": [], "virt": []}
            # for braket in ["bra", "ket"]:
            #     self.used_indices[space][braket] = {"occ": [], "virt": []}

    def get_gs_indices(self, braket, **kwargs):
        """Returns indices as sympy symbols in form of a dict, sorted
           by the keys 'occ' and 'virt'.
           Ground state indices are removed from the default list and
           every invoked space. So no space will have access to them.
           """
        # the Gs indices need to be separated somehow...
        # for now using the standard indices for ADC spaces, while
        # Gs uses the i1,j1,..., i2... indices.

        valid = {'n_occ': "occ", 'n_virt': "virt"}
        for key, value in kwargs.items():
            if key not in valid:
                print(f"{key} is not a valid option for requesting gs",
                      "indices.")
                exit()
            if not isinstance(value, int):
                print("Number of requested gs indices must be int, not",
                      type(value))
                exit()

        ret = {}
        used = self.used_indices["gs"][braket]
        for n_ov, n in kwargs.items():
            ov = valid[n_ov]
            # reuse all symbols
            if n <= len(used[ov]):
                ret[ov] = used[ov][:n]
            # not enough symbols available
            else:
                ret[ov] = used[ov][:n].copy()
                needed = n - len(used[ov])
                if not self.invoked_spaces:
                    # no spaces inoked yet -> grab from self.gs_occ/virt
                    get_default = {
                        "occ": self.gs_occ,
                        "virt": self.gs_virt
                    }
                    idx = get_default[ov][:needed].copy()
                    ret[ov].extend(
                        [self.__get_new_symbol("gs", ov, idxstr, braket=braket)
                         for idxstr in idx]
                    )
                else:
                    # some spaces invoked yet -> grab from smallest space,
                    # but only indices of of type i1, j1...
                    common = self.__get_common_indices(ov)
                    idx = [av for av in common if len(av) > 1][:needed]
                    ret[ov].extend(
                        [self.__get_new_symbol("gs", ov, idxstr, braket=braket)
                         for idxstr in idx]
                    )
        return ret

    def get_indices(self, space, indices):
        """Returns indices as sympy symbols in form of a dict, sorted
           by the keys 'occ' and 'virt'.
           New indices are taken from the available list of the respective
           space.
           The symbols are safed in a dict and reused if requested again.
           That way sympy recognizes symbols with the same name as equal.
           """

        # space: ph/pphh or gs
        # indices: str of indices, e.g. "ai"

        if not isinstance(space, str):  # or not isinstance(braket, str):
            print(f"Trying to get indices with space of type {type(space)}",
                  "Only string allowed.")
            exit()
        if space == "gs":
            print("Indices for the ground state should be obtained with",
                  "get_gs_indices instead.")
            exit()
        # if braket not in ["bra", "ket"]:
        #     print(f"Can only get indices for 'bra' or 'ket' ,not {braket}")
        #     exit()
        if not isinstance(indices, str):
            print("Requested indices need to be of type str (e.g. 'ai'), not",
                  type(indices))
            exit()

        ret = {}
        used = self.used_indices[space]  # [braket]
        for idx in indices:
            ov = self.__assign_index(idx)
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

    def __get_new_symbol(self, space, ov, idx, braket=None):
        if not self.available_indices[space][ov]:
            print(f"No indices for space {ov} {space} available anymore.")
            exit()
        if idx not in self.available_indices[space][ov]:
            print(f"Could not find {ov} index {idx} in available indices",
                  f"for space {space}.")
            exit()
        symbol = self.__make_symbol_new(ov, idx)
        self.remove(space, braket, ov, symbol)
        return symbol

    def __get_common_indices(self, ov):
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
            if space != "gs":
                lengths[space] = len(dic[ov])
        return min(lengths, key=lengths.get)

    def __make_symbol_new(self, ov, idx):
        if ov == "occ":
            return symbols(idx, below_fermi=True, cls=Dummy)
        elif ov == "virt":
            return symbols(idx, above_fermi=True, cls=Dummy)

    def remove(self, space, braket, ov, symbol):
        """removes index from available and add to the list of
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
        if space == "gs":
            used = self.used_indices[space][braket][ov]
            available.remove(idx)
            used.append(symbol)
            for space in self.invoked_spaces:
                self.available_indices[space][ov].remove(idx)
        # remove from ph/pphh space
        else:
            used = self.used_indices[space][ov]
            available.remove(idx)
            used.append(symbol)

    def __assign_index(self, idx):
        # assigns an index to occ or virt
        if idx[0] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            return "virt"
        elif idx[0] in ["i", "j", "k", "l", "m", "n", "o"]:
            return "occ"
        else:
            print(f"Could not assign index {idx} to occ or virt.")
            exit()
