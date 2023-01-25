from functools import wraps


class Inputerror(ValueError):
    pass


def cached_member(function):
    """Decorator for a member function thats called with
       at least one argument. The result is cached in the
       member variable '_function_cache' of the class instance.
       """

    fname = function.__name__

    @wraps(function)
    def wrapper(self, *args):
        try:
            fun_cache = self._function_cache[fname]
        except AttributeError:
            self._function_cache = {}
            fun_cache = self._function_cache[fname] = {}
        except KeyError:
            fun_cache = self._function_cache[fname] = {}

        try:
            return fun_cache[args]
        except KeyError:
            fun_cache[args] = result = function(self, *args)
        return result
    return wrapper


def process_arguments(function):
    """Decorator for a function thats called with at least one argument.
       All provided arguments (args and kwargs) are processed if necessary to
       avoid unnecessary calculations. For instance, indices are sorted, i.e.,
       indices='ijab' and indices='abji' will both be tranformed to 'abij'.
       Furthermore, all kwargs are forwarded as args to the decorated function.
       """
    from .indices import split_idx_string
    import inspect

    sig = inspect.signature(function)

    @wraps(function)
    def wrapper(*args, **kwargs):
        def sort_spaces(spaces):
            return ",".join(["".join(sorted(s, reverse=True)) for s in spaces])

        def sort_indices(idx_tpl):
            sorted_list = ["".join(sorted(split_idx_string(idx),
                           key=lambda i: (int(i[1:]) if i[1:] else 0, i[0])))
                           for idx in idx_tpl]
            return ",".join(sorted_list)

        # order, min_order, braket, lr, subtract_gs, adc_order
        # -> nothing to sort
        process = {
            'opstring': sort_spaces,
            'space': sort_spaces,
            'indices': sort_indices,
            'block': sort_spaces
        }

        n_args = len(args)
        new_args = []
        for arg_idx, (arg, value) in enumerate(sig.parameters.items()):
            # add all args to kwargs and then process if necessary
            # if this is used to decorate a member function self should also
            # be handled correctly
            if n_args > arg_idx:
                if arg in kwargs:
                    raise RuntimeError(f"found argument {arg} in kwargs, "
                                       "but expected it in args.")
                kwargs[arg] = args[arg_idx]
            try:
                val = kwargs[arg]
                # process the value if needed and not just the default value
                # has been provided
                if val != value.default:
                    fun = process.get(arg, None)
                    val = fun(transform_to_tuple(val)) if fun else val
                new_args.append(val)
                del kwargs[arg]
            except KeyError:
                # argument is not provided in kwargs or args
                # -> has to have a default value
                if value.default is not inspect._empty:
                    new_args.append(value.default)
                    continue
                else:
                    raise TypeError(f"Positional argument {arg} missing.")
        # if everything worked kwargs should be an empty dict here
        if kwargs:
            raise Inputerror(
                "Wrong or too many arguments provided for function "
                f"{function.__name__}. Not possible to assign {kwargs} "
                f"to {inspect.signature(function).parameters}."
            )
        return function(*new_args)

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
    # order, min_order, adc_order, braket, space, lr: single input
    # indices, block: 2 strings possible
    validate = {
        'order': lambda o: isinstance(o, int) and o >= 0,  # int
        'braket': lambda bk: bk in ['bra', 'ket'],  # bra/ket
        'space': lambda sp: all(s in ['p', 'h'] for s in sp),
        'indices': lambda idx: all(isinstance(i, str) for i in idx),
        'min_order': lambda o: isinstance(o, int) and o >= 0,  # int
        'lr': lambda lr: lr in ['left', 'right'],  # left/right
        'block': lambda b: all(validate['space'](sp) for sp in b),
        'adc_order': lambda o: isinstance(o, int) and o >= 0,  # int
        'opstring': lambda op: all(o in ['a', 'c'] for o in op),
        'lr_isr': lambda lr: lr in ['left', 'right'],  # left/right
    }
    # braket, lr are exprected as str!
    # order, min_order, adc_order are expected as int!
    # space, block and indices as list/tuple or ',' separated string
    for var, val in kwargs.items():
        if var in {'space', 'opstring'}:
            tpl = transform_to_tuple(val)
            if len(tpl) != 1:
                raise Inputerror(f'Invalid input for {var}: {val}.')
            val = tpl[0]
        elif var == 'block':
            tpl = transform_to_tuple(val)
            if len(tpl) != 2:
                raise Inputerror(f'Invalid input for {var}: {val}')
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


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = (
                super(Singleton, cls).__call__(*args, **kwargs)
            )
        return cls._instances[cls]
