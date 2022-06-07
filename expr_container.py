from indices import index_space
from misc import Inputerror
from sympy import latex, Add, Mul, Pow
from sympy.physics.secondquant import (
    AntiSymmetricTensor, NO, F, Fd, KroneckerDelta
)


class expr:
    def __new__(cls, e, real=False, sym_tensors=[]):
        if isinstance(e, (obj, term, expr)):
            return expr(e.sympy, real, sym_tensors)
        return super().__new__(cls)

    def __init__(self, e, real=False, sym_tensors=[]):
        self.__expr = e
        self.__real = real
        self.__sym_tensors = set(sym_tensors)
        if real:
            self.__sym_tensors.update('f', 'V')
            self.make_real

    def __str__(self):
        return latex(self.sympy)

    def __len__(self):
        # 0 has length 1
        if self.type == 'expr':
            return len(self.args)
        else:
            return 1

    def __getattr__(self, attr):
        return getattr(self.__expr, attr)

    @property
    def real(self):
        return self.__real

    @property
    def sym_tensors(self):
        return self.__sym_tensors

    @property
    def sympy(self):
        return self.__expr

    @property
    def type(self):
        if isinstance(self.sympy, Add):
            return 'expr'
        elif isinstance(self.sympy, Mul):
            return 'term'
        else:
            return 'obj'

    @property
    def terms(self):
        if self.type == 'expr':
            return [term(self, i) for i in range(len(self.args))]
        else:
            return [term(self, 0)]

    def set_sym_tensors(self, *sym_tensors):
        self.__sym_tensors = set(sym_tensors)
        if self.real:
            self.__sym_tensors.update(['f', 'V'])

    @property
    def make_real(self):
        """Makes the expression real by removing all 'c' in tensor names.
           This only renames the tensor, but their might be more to simplify
           by swapping bra/ket.
           """

        self.__real = True
        self.__sym_tensors.update(['f', 'V'])
        real_expr = 0
        for t in self.terms:
            temp = 1
            for o in t.objects:
                if o.type == 'tensor':
                    old = o.name
                    new = old.replace('c', '')
                    temp *= (Pow(AntiSymmetricTensor(new, o.extract_pow.upper,
                             o.extract_pow.lower), o.exponent)
                             if old != new else o.sympy)
                else:
                    temp *= o.sympy
            real_expr += temp
        self.__expr = real_expr
        return self

    def rename_tensor(self, current, new):
        renamed = compatible_int(0)
        for t in self.terms:
            renamed += t.rename_tensor(current, new)
        self.__expr = renamed.sympy
        return self

    def expand(self):
        self.__expr = self.sympy.expand()
        return self

    def subs(self, *args, **kwargs):
        self.__expr = self.sympy.subs(*args, **kwargs)
        return self

    def __iadd__(self, other):
        if isinstance(other, (expr, term, obj)):
            if other.real != self.real or \
                    other.sym_tensors != self.sym_tensors:
                raise TypeError("Real and symmetric tensors need to be equal.")
            other = other.sympy
        self.__expr = self.sympy + other
        return self

    def __add__(self, other):
        if isinstance(other, (expr, term, obj)):
            if other.real != self.real or \
                    other.sym_tensors != self.sym_tensors:
                raise TypeError("Real and symmetric tensors need to be equal.")
            other = other.sympy
        return expr(self.sympy + other, self.real, self.sym_tensors)

    def __isub__(self, other):
        if isinstance(other, (expr, term, obj)):
            if other.real != self.real or \
                    other.sym_tensors != self.sym_tensors:
                raise TypeError("Real and symmetric tensors need to be equal.")
            other = other.sympy
        self.__expr = self.sympy - other
        return self

    def __sub__(self, other):
        if isinstance(other, (expr, term, obj)):
            if other.real != self.real or \
                    other.sym_tensors != self.sym_tensors:
                raise TypeError("Real and symmetric tensors need to be equal.")
            other = other.sympy
        return expr(self.sympy - other, self.real, self.sym_tensors)

    def __imul__(self, other):
        if isinstance(other, (expr, term, obj)):
            if other.real != self.real or \
                    other.sym_tensors != self.sym_tensors:
                raise TypeError("Real and symmetric tensors need to be equal.")
            other = other.sympy
        self.__expr = self.sympy * other
        return self

    def __mul__(self, other):
        if isinstance(other, (expr, term, obj)):
            if other.real != self.real or \
                    other.sym_tensors != self.sym_tensors:
                raise TypeError("Real and symmetric tensors need to be equal.")
            other = other.sympy
        return expr(self.sympy * other, self.real, self.sym_tensors)

    def __eq__(self, other):
        if isinstance(other, (expr, term, obj)) and self.real == other.real \
                and self.sym_tensors == other.sym_tensors and \
                self.sympy == other.sympy:
            return True
        return False


class term:
    def __new__(cls, t, pos=None, real=False, sym_tensors=[]):
        if isinstance(t, expr):
            if pos is None:
                raise Inputerror('No position provided.')
            return super().__new__(cls)
        else:
            return expr(t, real=real, sym_tensors=sym_tensors)

    def __init__(self, t, pos=None, real=False, sym_tensors=[]):
        self.__expr = t
        self.__pos = pos

    def __str__(self):
        return latex(self.term)

    def __len__(self):
        return len(self.args) if self.type == 'term' else 1

    def __getattr__(self, attr):
        return getattr(self.term, attr)

    @property
    def expr(self):
        return self.__expr

    @property
    def term(self):
        if self.expr.type == 'expr':
            return self.__expr.args[self.__pos]
        else:
            return self.__expr.sympy

    @property
    def real(self):
        return self.expr.real

    @property
    def sym_tensors(self):
        return self.expr.sym_tensors

    @property
    def pos(self):
        return self.__pos

    @property
    def sympy(self):
        return self.term

    @property
    def type(self):
        return 'term' if isinstance(self.term, Mul) else 'obj'

    @property
    def objects(self):
        return [obj(self, i) for i in range(len(self))]

    @property
    def tensors(self):
        return [o for o in self.objects if o.type == 'tensor']

    @property
    def deltas(self):
        return [o for o in self.objects if o.type == 'delta']

    @property
    def make_real(self):
        real_term = compatible_int(1)
        for o in self.objects:
            if o.type == 'tensor':
                old = o.name
                new = old.replace('c', '')
                real_term *= (Pow(AntiSymmetricTensor(new, o.extract_pow.upper,
                              o.extract_pow.lower), o.exponent)
                              if old != new else o.sympy)
            else:
                real_term *= o.sympy
        return expr(real_term, True, self.sym_tensors)

    def rename_tensor(self, current, new):
        """Rename tensors in a terms. Returns a new expr instance."""
        renamed = compatible_int(1)
        for o in self.objects:
            renamed *= o.rename_tensor(current, new)
        return renamed

    def expand(self):
        return expr(self.term.expand(), self.real, self.sym_tensors)

    def subs(self, *args, **kwargs):
        return expr(self.term.subs(*args, **kwargs), self.real,
                    self.sym_tensors)

    @property
    def contracted(self):
        return tuple(sorted(
            [s for s, n in self.__idx_counter.items() if n],
            key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
        ))

    @property
    def target(self):
        return tuple(sorted(
            [s for s, n in self.__idx_counter.items() if not n],
            key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
        ))

    @property
    def __idx_counter(self):
        idx = {}
        for o in self.objects:
            n = o.exponent
            for s in o.idx:
                if s in idx:
                    idx[s] += n
                else:  # start counting at 0
                    idx[s] = n - 1
        return idx

    @property
    def sign_change(self):
        """Returns True if the sign of the term changes upon resorting all
           tensors in the term in their canonical form."""
        sign_change = False
        for o in self.objects:
            sign_change = not sign_change if o.sign_change else sign_change
        return sign_change

    def swap_braket(self, t_string, occurence=1):
        """Method to swap bra and ket for the n-th occurence of a tensor."""
        relevant_tensors = [o for o in self.tensors if o.name == t_string]
        # tensor not often enough included in the term
        if sum(o.exponent for o in relevant_tensors) < occurence:
            return expr(self.sympy, self.real, self.sym_tensors)
        current_occurence = 1
        swapped = 1
        for o in self.objects:
            if o in relevant_tensors:
                if current_occurence <= occurence < \
                        current_occurence + o.exponent:
                    swapped *= (
                        AntiSymmetricTensor(t_string, o.extract_pow.lower,
                                            o.extract_pow.upper) *
                        Pow(AntiSymmetricTensor(t_string, o.extract_pow.upper,
                            o.extract_pow.lower), o.exponent - 1)
                    )
                else:
                    swapped *= o.sympy
                current_occurence += o.exponent
            else:
                swapped *= o.sympy
        return expr(swapped, self.real, self.sym_tensors)

    def pattern(self, coupling=False):
        """Returns the pattern of the indices in the term. If coupling is set,
           the coupling between the objects is taken into account"""
        coupl = self.coupling if coupling else 0
        ret = {'o': {}, 'v': {}}
        for i, o in enumerate(self.objects):
            c = "_" + "_".join(sorted(coupl[i])) if coupl and i in coupl \
                else ''
            for s, pos in o.crude_pos.items():
                ov = index_space(s.name)[0]
                if s not in ret[ov]:
                    ret[ov][s] = []
                ret[ov][s].extend([p + c for p in pos])
        return ret

    @property
    def coupling(self):
        """Returns the coupling between the objects in the term, where two
           objects are coupled when they share common indices."""
        from itertools import product
        # from collections import Counter
        # 1) collect all the couplings (e.g. if a index s occurs at two tensors
        #    t and V: the crude_pos of s at t will be extended by the crude_pos
        #    of s at V. And vice versa for V.)
        coupling = {}
        for i, o in enumerate(self.objects):
            descr = o.description
            if descr not in coupling:
                coupling[descr] = {}
            if i not in coupling[descr]:
                coupling[descr][i] = []
            idx_pos = o.crude_pos
            for other_i, other_o in enumerate(self.objects):
                if i == other_i:
                    continue
                other_idx_pos = other_o.crude_pos
                matches = [comb[0] for comb in
                           product(idx_pos.keys(), other_idx_pos.keys())
                           if comb[0] == comb[1]]
                coupling[descr][i].extend(
                    [p for s in matches for p in other_idx_pos[s]]
                )
        # 2) check whether some identical tensors (according to their
        #    description) also have identical coupling. If this is the case
        #    a counter is added to the coupling to differentiate them.
        ret = {}
        for coupl in coupling.values():
            # tensor is unique in the term no need to track coupling
            if len(coupl.keys()) == 1:
                continue
            ret.update(coupl)
            # TODO: I think this should not be needed... try without
            # identify and collect equal couplings
            # equal_coupl = []
            # matched = []
            # for i, c in coupl.items():
            #     if i in matched:
            #         continue
            #     matched.append(i)
            #     temp = [i]
            #     count_c = Counter(c)
            #     for other_i, other_c in coupl.items():
            #         if other_i in matched:
            #             continue
            #         if count_c == Counter(other_c):
            #             temp.append(other_i)
            #             matched.append(other_i)
            #     equal_coupl.append(temp)
            # # attach a number to the equal coupling
            # for equal_c in equal_coupl:
            #     temp = {}
            #     for n, i in enumerate(equal_c):
            #         temp[i] = coupl[i] + [str(n+1)]
            #     ret.update(temp)
        return ret

    def __iadd__(self, other):
        if isinstance(other, (expr, term, obj)):
            if other.real != self.real or \
                    other.sym_tensors != self.sym_tensors:
                raise TypeError("Real and symmetric tensors need to be equal.")
            other = other.sympy
        return expr(self.sympy + other, self.real, self.sym_tensors)

    def __add__(self, other):
        return self.__iadd__(other)

    def __isub__(self, other):
        if isinstance(other, (expr, term, obj)):
            if other.real != self.real or \
                    other.sym_tensors != self.sym_tensors:
                raise TypeError("Real and symmetric tensors need to be equal.")
            other = other.sympy
        return expr(self.sympy - other, self.real, self.sym_tensors)

    def __sub__(self, other):
        return self.__isub__(other)

    def __imul__(self, other):
        if isinstance(other, (expr, term, obj)):
            if other.real != self.real or \
                    other.sym_tensors != self.sym_tensors:
                raise TypeError("Real and symmetric tensors need to be equal.")
            other = other.sympy
        return expr(self.sympy * other, self.real, self.sym_tensors)

    def __mul__(self, other):
        return self.__imul__(other)

    def __eq__(self, other):
        if isinstance(other, (expr, term, obj)) and self.real == other.real \
                and self.sym_tensors == other.sym_tensors and \
                self.sympy == other.sympy:
            return True
        return False


class obj:
    def __new__(cls, t, pos=None, real=False, sym_tensors=[]):
        types = {
            NO: lambda o: 'no',
            Pow: lambda o: 'polynom' if isinstance(o.args[0], Add) else 'obj'
        }
        if isinstance(t, (term, normal_ordered)):
            if pos is None:
                raise Inputerror('No position provided.')
            o = t.sympy if len(t) == 1 else t.args[pos]
            obj_type = types.get(type(o), lambda x: 'obj')(o)
            if obj_type == 'obj':
                return super().__new__(cls)
            elif obj_type == 'no':
                return normal_ordered(t, pos)
            else:
                raise NotImplementedError()
        else:
            return expr(t, real=real, sym_tensors=sym_tensors)

    def __init__(self, t, pos=None, real=False, sym_tensors=[]):
        self.__expr = t.expr
        self.__term = t
        self.__pos = pos

    def __str__(self):
        return latex(self.sympy)

    def __getattr__(self, attr):
        return getattr(self.obj, attr)

    @property
    def expr(self):
        return self.__expr

    @property
    def term(self):
        return self.__term

    @property
    def obj(self):
        return self.term.sympy if len(self.term) == 1 \
            else self.term.args[self.__pos]

    @property
    def real(self):
        return self.expr.real

    @property
    def sym_tensors(self):
        return self.expr.sym_tensors

    @property
    def sympy(self):
        return self.obj

    @property
    def make_real(self):
        if self.type == 'tensor':
            old = self.name
            new = old.replace('c', '')
            real_obj = (Pow(AntiSymmetricTensor(new, self.extract_pow.upper,
                        self.extract_pow.lower), self.exponent)
                        if old != new else self.sympy)
        else:
            real_obj = self.sympy
        return expr(real_obj, True, self.sym_tensors)

    def rename_tensor(self, current, new):
        """Renames a tensor object."""
        if self.type == 'tensor' and self.name == current:
            new_obj = Pow(AntiSymmetricTensor(new, self.extract_pow.upper,
                          self.extract_pow.lower), self.exponent)
        else:
            new_obj = self.sympy
        return expr(new_obj, real=self.real, sym_tensors=self.sym_tensors)

    def expand(self):
        return expr(self.sympy.expand(), self.real, self.sym_tensors)

    def subs(self, *args, **kwargs):
        return expr(self.obj.subs(*args, **kwargs), self.real,
                    self.sym_tensors)

    @property
    def exponent(self):
        return self.obj.args[1] if isinstance(self.obj, Pow) else 1

    @property
    def extract_pow(self):
        return self.obj if self.exponent == 1 else self.obj.args[0]

    @property
    def type(self):
        types = {
            AntiSymmetricTensor: 'tensor',
            KroneckerDelta: 'delta',
            F: 'annihilate',
            Fd: 'create',
        }
        try:
            return types[type(self.extract_pow)]
        except KeyError:
            if len(self.free_symbols) != 0:
                raise RuntimeError(f"Unknown object: {self.obj} of type "
                                   f"{type(self.obj)}.")
            return 'prefactor'

    @property
    def name(self):
        """Return the name of tensor objects."""
        if self.type == 'tensor':
            return self.extract_pow.symbol.name

    @property
    def symmetric(self):
        return (True if self.type == 'tensor' and self.name in self.sym_tensors
                else False)

    @property
    def idx(self):
        """Return the indices of the canonical ordered object."""
        get_idx = {
            'tensor': lambda o: self.__canonical_idx(o),
            'delta': lambda o: (o.preferred_index, o.killable_index),
            'annihilate': lambda o: (o.args[0], ),
            'create': lambda o: (o.args[0], )
        }
        try:
            return get_idx[self.type](self.extract_pow)
        except KeyError:
            if self.type == 'prefactor':
                return tuple()
            else:
                raise KeyError(f'Unknown obj type {self.type} for obj {self}')

    @property
    def space(self):
        """Returns the canonical space of tensors."""
        return "".join([index_space(s.name)[0] for s in self.idx])

    @property
    def crude_pos(self):
        """Returns the 'crude' position of the indices in the object.
           (e.g. only if they are located in bra/ket, not the exact position)
           """
        ret = {}
        descr = self.description
        type = self.type
        if type == 'tensor':
            can_bk = self.__canonical_bk(self.extract_pow)
            for bk, idx_tpl in can_bk.items():
                for s in idx_tpl:
                    if s not in ret:
                        ret[s] = []
                    if self.symmetric:
                        pos = descr + '_ul'
                    else:
                        pos = descr + '_' + bk[0]
                    ret[s].append(pos)
        elif type in ['delta', 'annihilate', 'create']:
            for s in self.idx:
                if s not in ret:
                    ret[s] = []
                ret[s].append(descr)
        # for prefactor a empty dict is returned
        return ret

    def __canonical_bk(self, o):
        # sorts the indices in bra and ket in canonical order
        can_bk = {}
        for bk in ['upper', 'lower']:
            can_bk[bk] = sorted(
                getattr(o, bk), key=lambda s: (index_space(s.name)[0], s.name)
            )
        return can_bk

    def __canonical_idx(self, o):
        # sorts the overall index string in canonical order (swapps bra/ket)
        from collections import Counter

        can_bk = self.__canonical_bk(o)
        # amplitudes
        if self.name[0] in ['t', 'X', 'Y']:
            can_order = ['lower', 'upper']
        # symmetric tensors
        elif self.symmetric:
            ov = {}
            for bk in can_bk:
                ov[bk] = (
                    "".join([index_space(s.name)[0] for s in can_bk[bk]]),
                    "".join(s.name for s in can_bk[bk])
                )
            can_order = sorted(
                ov, key=lambda ov_idx:
                (Counter(ov[ov_idx][0])['v'], ov[ov_idx][1])
            )
        # non symmetric tensors (also V if the expr is not real)
        else:
            can_order = ['upper', 'lower']
        return tuple(s for bk in can_order for s in can_bk[bk])

    @property
    def sign_change(self):
        """Returns True if the sign of the tensor object changes upon
           reordering the indices in canonical order."""
        if self.type != 'tensor':
            return False
        can_bk = self.__canonical_bk(self.extract_pow)
        tensor = Pow(AntiSymmetricTensor('42', can_bk['upper'],
                     can_bk['lower']), self.exponent)
        if isinstance(tensor, Mul):
            return True
        else:
            return False

    @property
    def description(self):
        """A string that describes the object."""
        descr = self.type
        if descr == 'tensor':
            descr = '_'.join(
                [descr, self.name, str(len(self.extract_pow.upper)),
                 str(len(self.extract_pow.lower)), str(self.exponent)]
            )
        elif descr in ['delta', 'annihilate', 'create']:
            descr = '_'.join([descr, str(self.exponent)])
        return descr

    def __iadd__(self, other):
        if isinstance(other, (expr, term, obj)):
            if other.real != self.real or \
                    other.sym_tensors != self.sym_tensors:
                raise TypeError("Real and symmetric tensors need to be equal.")
            other = other.sympy
        return expr(self.sympy + other, self.real, self.sym_tensors)

    def __add__(self, other):
        return self.__iadd__(other)

    def __isub__(self, other):
        if isinstance(other, (expr, term, obj)):
            if other.real != self.real or \
                    other.sym_tensors != self.sym_tensors:
                raise TypeError("Real and symmetric tensors need to be equal.")
            other = other.sympy
        return expr(self.sympy - other, self.real, self.sym_tensors)

    def __sub__(self, other):
        return self.__isub__(other)

    def __imul__(self, other):
        if isinstance(other, (expr, term, obj)):
            if other.real != self.real or \
                    other.sym_tensors != self.sym_tensors:
                raise TypeError("Real and symmetric tensors need to be equal.")
            other = other.sympy
        return expr(self.sympy * other, self.real, self.sym_tensors)

    def __mul__(self, other):
        return self.__imul__(other)

    def __eq__(self, other):
        if isinstance(other, (expr, term, obj)) and self.real == other.real \
                and self.sym_tensors == other.sym_tensors and \
                self.sympy == other.sympy:
            return True
        return False


class normal_ordered(obj):
    """Container for a normal ordered operator string."""
    def __new__(cls, t, pos=None, real=False, sym_tensors=[]):
        if isinstance(t, term):
            if pos is None:
                raise Inputerror('No position provided.')
            o = t.sympy if len(t) == 1 else t.args[pos]
            if isinstance(o, NO):
                return object.__new__(cls)
            else:
                raise RuntimeError('Trying to use normal_ordered container'
                                   f'for a non NO object: {o}.')
        else:
            return expr(t, real=real, sym_tensors=sym_tensors)

    def __init__(self, t, pos=None, real=False, sym_tensors=[]):
        super().__init__(t, pos)

    def __len__(self):
        # a NO obj can only contain a Mul object.
        return len(self.extract_no.args)

    @property
    def args(self):
        return self.extract_no.args

    @property
    def objects(self):
        return [obj(self, i) for i in range(len(self.extract_no.args))]

    @property
    def extract_no(self):
        return self.obj.args[0]

    @property
    def exponent(self):
        # actually sympy should throw an error if a NO object contains a Pow
        # obj or anything else than a*b*c
        exp = set(o.exponent for o in self.objects)
        if len(exp) == 1:
            return next(iter(exp))
        else:
            raise NotImplementedError(
                'Exponent only implemented for NO objects, where all '
                f'operators share the same exponent. {self}'
            )

    @property
    def type(self):
        return 'normal_ordered'

    @property
    def idx(self):
        objects = self.objects
        ret = tuple(s for o in self.objects for s in o.idx)
        if len(objects) != len(ret):
            raise NotImplementedError('Expected a NO object only to contain'
                                      f"second quantized operators. {self}")
        return ret

    @property
    def crude_pos(self):
        ret = {}
        for o in self.objects:
            for s, pos in o.crude_pos.items():
                if s not in ret:
                    ret[s] = []
                ret[s].extend(pos)
        return ret

    @property
    def description(self):
        # just return NO_ncreate_nannihilate?
        # no -> One needs the information whether an index occurs at create or
        #       annihilate
        raise NotImplementedError("Description not implemented for NO objects")


class polynom(expr):
    pass


class compatible_int(int):
    def __init__(self, num):
        super().__init__()
        self.num = num

    def __iadd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, (obj, term, expr)):
            return expr(self.num + other.sympy, other.real, other.sym_tensors)
        else:
            return super().__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __sub__(self, other):
        if isinstance(other, (obj, term, expr)):
            return expr(self.num - other.sympy, other.real, other.sym_tensors)
        else:
            return super().__sub__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, (obj, term, expr)):
            return expr(self.num * other.sympy, other.real, other.sym_tensors)
        else:
            return super().__mul__(other)

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        if isinstance(other, (obj, term, expr)):
            return expr(self.num / other.sympy, other.real, other.sym_tensors)
        else:
            return super().__truediv__(other)
