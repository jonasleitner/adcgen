from functools import wraps
import inspect


class Inputerror(ValueError):
    pass


def cached_member(function):
    """Decorator for a class method thats called with
       at least one argument. The result is cached in the
       member variable '_function_cache' of the class instance.
       """

    fname = function.__name__

    # create the signature of the wrapped function and check that we dont
    # have any keyword only arguments in the wrapped function
    func_sig = inspect.signature(function)
    invalid_arg_types = (inspect.Parameter.KEYWORD_ONLY,
                         inspect.Parameter.VAR_KEYWORD)
    if any(arg.kind in invalid_arg_types for arg in
           func_sig.parameters.values()):
        raise TypeError("Functions with keyword only arguments are not "
                        "supported by cached_member.")

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        # - transform all arguments to positional arguments
        #   and add not provided default arguments
        bound_args: inspect.BoundArguments = (
            func_sig.bind(self, *args, **kwargs)
        )
        bound_args.apply_defaults()
        assert len(bound_args.kwargs) == 0
        args = bound_args.args[1:]  # remove self from the positional arguments

        try:  # load/create the cache
            fun_cache = self._function_cache[fname]
        except AttributeError:
            self._function_cache = {}
            fun_cache = self._function_cache[fname] = {}
        except KeyError:
            fun_cache = self._function_cache[fname] = {}

        try:  # try to load the data from the cache
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
        if var in ['space', 'opstring']:
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
