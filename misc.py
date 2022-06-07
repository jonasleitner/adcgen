from functools import wraps


class Inputerror(ValueError):
    pass


def cached_member(function):
    """Decorator for a member function thats called with
       at least one argument. The result is cached in the
       member variable '_function_cache' of the class instance.
       """

    from indices import split_idx_string
    from inspect import signature
    fname = function.__name__

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        def sort_spaces(sp):
            return ",".join(["".join(sorted(s, reverse=True)) for s in sp])

        def sort_indices(idx_strings):
            return ",".join(["".join(sorted(split_idx_string(idx),
                            key=lambda i: (int(i[1:]) if i[1:] else 0, i[0])))
                            for idx in idx_strings])

        def validate_bk(braket):
            if len(braket) == 1 and braket[0] in ['bra', 'ket']:
                return braket[0]
            else:
                raise Inputerror(f"Invalid argument for braket: {braket}.")

        def validate_lr(lr):
            if len(lr) == 1 and lr[0] in ['left', 'right']:
                return lr[0]
            else:
                raise Inputerror(f"Invalid argument for lr: {lr}.")

        try:
            fun_cache = self._function_cache[fname]
        except AttributeError:
            self._function_cache = {}
            fun_cache = self._function_cache[fname] = {}
        except KeyError:
            fun_cache = self._function_cache[fname] = {}

        # all spaces and indices that are in kwargs are sorted
        # to bring them in a common form to avoid unneccesary
        # additional calculations
        process = {
            # order, adc_order, min_order: type(int) -> already covered
            'braket': validate_bk,
            'lr': validate_lr,
            'space': sort_spaces,
            'block': sort_spaces,
            'mvp_space': sort_spaces,
            'indices': sort_indices,
        }
        kwargs_key = {}
        for var, value in kwargs.items():
            # catch orders -> hand over as int
            if isinstance(value, int):
                kwargs_key[var] = value
                continue
            val = transform_to_tuple(value)
            kwargs_key[var] = process[var](val)

        # add the kwargs arguments in the appropriate order to args
        for argument in signature(function).parameters.keys():
            try:
                args += (kwargs_key[argument],)
                del kwargs_key[argument]
            except KeyError:
                continue
        if kwargs_key:
            raise TypeError("Wrong or too many arguments provided for function"
                            f" '{function.__name__}'. The function takes: "
                            f"{list(signature(function).parameters.keys())}. "
                            f"Provided: {list(kwargs.keys())}.")

        try:
            return fun_cache[args]
        except KeyError:
            fun_cache[args] = result = function(self, *args)
        return result
    return wrapper


def cached_property(function):
    """Decorator for a cached property."""

    def get(self):
        try:
            return self._property_cache[function]
        except AttributeError:
            self._property_cache = {}
            x = self._property_cache[function] = function(self)
            return x
        except KeyError:
            x = self._property_cache[function] = function(self)
            return x

    get.__doc__ = function.__doc__

    return property(get)


def validate_input(**kwargs):
    # order, min_order, adc_order, braket, space, lr, mvp_space: single input
    # indices, block: 2 strings possible
    validate = {
        'order': lambda o: isinstance(o, int),  # int
        'braket': lambda bk: bk in ['bra', 'ket'],  # bra/ket
        'space': lambda sp: all(s in ['p', 'h'] for s in sp),
        'indices': lambda idx: all(isinstance(i, str) for i in idx),
        'min_order': lambda o: isinstance(o, int),  # int
        'lr': lambda lr: lr in ['left', 'right'],  # left/right
        'block': lambda b: all(validate['space'](sp) for sp in b),
        'mvp_space': lambda sp: all(s in ['p', 'h'] for s in sp),
        'adc_order': lambda o: isinstance(o, int),  # int
    }
    # braket, lr are exprected as str!
    # order, min_order, adc_order are expected as int!
    # space, mvp_space, block and indices as list/tuple or ',' separated string
    for var, val in kwargs.items():
        if var in {'space', 'mvp_space'}:
            tpl = transform_to_tuple(val)
            if len(tpl) != 1:
                raise Inputerror(f'Invalid input for space/mvp_space: {val}.')
            val = tpl[0]
        elif var == 'block':
            tpl = transform_to_tuple(val)
            if len(tpl) != 2:
                raise Inputerror(f'Invalid input for block: {val}')
            val = tpl
        elif var == 'indices':
            tpl = transform_to_tuple(val)
            if len(tpl) not in [1, 2]:
                raise Inputerror(f'Invalid indices input: {val}.')
            val = tpl
        if not validate[var](val):
            raise Inputerror(f'Invalid input for {var}: {val}.')


def transform_to_tuple(input):
    convertor = {
        str: lambda x: tuple(i for i in x.split(",")),
        list: lambda x: tuple(x),
        tuple: lambda x: x
    }
    conversion = convertor.get(type(input))
    if not conversion:
        raise Inputerror(f"{input} of type {type(input)} is not convertable "
                         "to tuple.")
    return conversion(input)
