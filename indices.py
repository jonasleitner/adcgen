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
        self.used_indices = {}
        # {'ket': {'occ': [idx]}}
        self.used_groundstate = {}
        # list with possible indices
        # indices that are shared between spaces (like the ones used
        # for the groundstate) are removes from the below lists.
        self.occ = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'i1', 'j1', 'k1', 'l1',
                    'm1', 'n1', 'o1', 'i2', 'j2', 'k2', 'l2', 'm2', 'n2', 'o2']
        self.virt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'a1', 'b1', 'c1',
                     'd1', 'e1', 'f1', 'g1', 'h1', 'a2', 'b2', 'c2', 'd2',
                     'e2', 'f2', 'g2', 'h2']
        self.invoked_spaces = []

    def __set_default(self, space):
        self.available_indices[space] = {}
        self.available_indices[space]['occ'] = self.occ.copy()
        self.available_indices[space]['virt'] = self.virt.copy()

    def create_space(self, space):
        if not isinstance(space, str):
            print(f"Space need to be of type str, not of {type(space)}")
            exit()
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
            for ov in ["occ", "virt"]:
                self.used_indices[space][ov] = []

    def get_indices(self, space, **kwargs):
        # print(f"try to get indices: {kwargs}")
        # kwargs: number of indices for each pool
        # kwargs: "occ" = 2, "virt"=5
        if not isinstance(space, str):
            print(f"Space needs to be of type str, not {type(space)}")
            exit()
        if space not in self.available_indices and space not in ["ket", "bra"]:
            print(f"Space {space} has not been created yet. Not possible to",
                  "obtain indices.")
            exit()
        for key, n in kwargs.items():
            if key not in ["occ", "virt"]:
                print("Can only request indices of type 'occ' or 'virt',",
                      f"not {key}")
                exit()
            if not isinstance(n, int):
                print("Need to provide how many indices of type",
                      f"{key} are requested by providing an integer.",
                      f"{n} of type {type(n)} is not valid.")
                exit()

        # requesting for gs wavefunction
        if space in ["ket", "bra"]:
            # no space constructed yet
            if not self.available_indices:
                # never requested any indices for bra or ket yet.
                # -> use self.occ/self.virt list
                if space not in self.used_groundstate:
                    ret = self.__get_indices_from_default(space, **kwargs)
                # already created waveunftion of e.g. type ket before
                elif space in self.used_groundstate:
                    ret = {}
                    for ov, n in kwargs.items():
                        # already enough indices available (e.g. when
                        # constructing first order after second order ket)
                        # -> reuse indices from used_groundstate
                        if n <= len(self.used_groundstate[space][ov]):
                            ret[ov] = self.used_groundstate[space][ov][:n]
                        # requesting more indices than available (e.g. when
                        # constructing second order after first order ket)
                        # -> use self.occ/virt list
                        elif n > len(self.used_groundstate[space][ov]):
                            ret[ov] = self.used_groundstate[space][ov].copy()
                            # print("reuse indices: ", av_idx)
                            needed = n - len(self.used_groundstate[space][ov])
                            idict = {ov: needed}
                            i1 = self.__get_indices_from_default(
                                space, **idict
                            )
                            ret[ov].extend(i1[ov])
            # at least one space has been created
            # get indices from space with least indices
            else:
                # the desired bra/ket has not been constructed previously
                # get all indices from space
                if space not in self.used_groundstate:
                    ret = {}
                    for ov, n in kwargs.items():
                        least_idx_space = self.__get_smallest(ov)
                        idict = {locals()["ov"]: n}
                        i1 = self.__get_indices_from_space(
                            least_idx_space, **idict
                        )
                        ret[ov] = i1[ov]
                # some indices may be reused from other gs wfns of same type
                elif space in self.used_groundstate:
                    ret = {}
                    for ov, n in kwargs.items():
                        # all indices may be reused
                        if n <= len(self.used_groundstate[space][ov]):
                            ret[ov] = self.used_groundstate[space][ov][:n]
                        # some indices need to be obtained from least_idx_space
                        elif n > len(self.used_groundstate[space][ov]):
                            ret[ov] = self.used_groundstate[space][ov].copy()
                            needed = n - len(self.used_groundstate[space][ov])
                            least_idx_space = self.__get_smallest(ov)
                            idict = {locals()["ov"]: needed}
                            i1 = self.__get_indices_from_space(
                                least_idx_space, **idict
                            )
                            ret[ov].extend(i1[ov])
        # requesting for ISR states
        # here it is way easier since the indices clearly belong to a space
        else:
            ret = {}
            for ov, n in kwargs.items():
                # reuse all indices
                if n <= len(self.used_indices[space][ov]):
                    ret[ov] = self.used_indices[space][ov][:n]
                elif n > len(self.used_indices[space][ov]):
                    needed = n - len(self.used_indices[space][ov])
                    # needed = n if no indices have been used previously
                    if needed == n:
                        idict = {locals()["ov"]: n}
                        i1 = self.__get_indices_from_space(space, **idict)
                        ret[ov] = i1[ov]
                    else:
                        ret[ov] = self.used_indices[space][ov].copy()
                        idict = {locals()["ov"]: needed}
                        i1 = self.__get_indices_from_space(space, **idict)
                        ret[ov].extend(i1[ov])
        return ret

    def __get_indices_from_default(self, braket, **kwargs):
        ret = {}
        if braket not in self.used_groundstate:
            self.used_groundstate[braket] = {}
        # print("Grab from self.occ/virt: ", kwargs)
        for ov, n in kwargs.items():
            ret[ov] = []
            if ov == "occ":
                idx = self.occ[:n]
                # print(f"grabbing from self.occ {idx}")
                symbol = self.__make_symbol(*idx)
                ret[ov].extend(symbol)
                # attach symbols to used list
                if "occ" not in self.used_groundstate:
                    self.used_groundstate[braket]["occ"] = []
                self.used_groundstate[braket]["occ"].extend(symbol)
            elif ov == "virt":
                idx = self.virt[:n]
                # print(f"grabbing from self.virt {idx}")
                symbol = self.__make_symbol(*idx)
                ret[ov].extend(symbol)
                # attach symbols to used list
                if "virt" not in self.used_groundstate[braket]:
                    self.used_groundstate[braket]["virt"] = []
                self.used_groundstate[braket]["virt"].extend(symbol)
            self.remove(braket, *ret[ov])
        return ret

    def __get_indices_from_space(self, space, **kwargs):
        ret = {}
        # print(f"Grab from space {space}")
        for ov, n in kwargs.items():
            idx = self.available_indices[space][ov][:n]
            ret[ov] = self.__make_symbol(*idx)
            self.used_indices[space][ov].extend(ret[ov])
            self.remove(space, *ret[ov])
        return ret

    def __get_smallest(self, ov):
        lengths = {}
        for space, dic in self.available_indices.items():
            lengths[space] = len(dic[ov])
        return min(lengths, key=lengths.get)

    def __make_symbol(self, *args):
        # assumes all passed indices belong to the same class (ovv/virt)!!
        ret = []
        for idx in args:
            if idx in self.occ:
                # print(f"make symbol found {idx} in self.occ")
                symbol = symbols(idx, below_fermi=True, cls=Dummy)
                ret.append(symbol)
            elif idx in self.virt:
                # print(f"make symbol found {idx} in self.virt")
                symbol = symbols(idx, above_fermi=True, cls=Dummy)
                ret.append(symbol)
            else:
                for bk in self.used_groundstate.values():
                    if idx in bk["occ"]:
                        # print(f"make symbol found {idx} in used_gs occ")
                        symbol = symbols(idx, below_fermi=True, cls=Dummy)
                        ret.append(symbol)
                    elif idx in bk["virt"]:
                        # print(f"make symbol found {idx} in used_gd virt")
                        symbol = symbols(idx, above_fermi=True, cls=Dummy)
                        ret.append(symbol)
        return ret

    def remove(self, space, *other):
        # print("try to remove:", space, other)
        for symbol in other:
            if not isinstance(symbol, sy.core.symbol.Dummy):
                print("Index that is to be removed from the available list",
                      f"needs to be a sympy symbol. Not type {type(other)}")
                exit()
            if space not in self.available_indices and \
                    space not in ["ket", "bra"]:
                print(f"The space {space} you want to remove from has not",
                      "been created yet. Can't remove index.")
                exit()
            to_remove = symbol.name
            if to_remove not in self.occ and to_remove not in self.virt:
                print("Index you want to remove, needs to be part of",
                      f"the default lists. '{to_remove}' is not part of",
                      "them or has allready been removed earlier by the",
                      "ground state.")
                exit()
            if space in ["ket", "bra"]:
                if to_remove in self.occ:
                    # print(f"found {to_remove} in self.occ for removing")
                    # remove from default list so new spaces will start without
                    # the ground state indices
                    self.occ.remove(to_remove)
                    # remove from all spaces, since the ground state
                    # wavefunction don't belong to any space
                    # (of the ADC matrix)
                    for indices in self.available_indices.values():
                        if to_remove in indices["occ"]:
                            indices["occ"].remove(to_remove)
                        else:
                            print("Warning: tried to remove groundsate index",
                                  f"from space {space}, but could not find it")
                elif to_remove in self.virt:
                    # print(f"found {to_remove} in self.virt for removing")
                    self.virt.remove(to_remove)
                    for indices in self.available_indices.values():
                        if to_remove in indices["virt"]:
                            indices["virt"].remove(to_remove)
                        else:
                            print("Warning: tried to remove groundsate index",
                                  f"from space {space}, but could not find it")
                else:
                    print(f"Could not find index {to_remove} in default",
                          f"indices for bra/ket {space}")
                    exit()

            elif space in self.available_indices:
                if to_remove in self.occ:
                    self.available_indices[space]["occ"].remove(to_remove)
                elif to_remove in self.virt:
                    self.available_indices[space]["virt"].remove(to_remove)
                else:
                    print("Could not find index in availale_indices for",
                          f"space {space}")
                    exit()
        # print("self used gs at end of removing: ", self.used_groundstate)


# i = symbols('i', below_fermi=True, cls=Dummy)
# a = symbols('a', above_fermi=True, cls=Dummy)
# idx = indices()
# av = idx.available_indices
# gs = idx.used_groundstate
# idx.create_space("ph")
# idx.create_space("pphh")
# idx.remove("ph", i, a)
# print(av)
# print("get GS: ", idx.get_indices("ket", occ=1, virt=1))
# print(gs)
# print("Get GS: ", idx.get_indices("bra", occ=2, virt=2))
# print(gs)
# print(av)
