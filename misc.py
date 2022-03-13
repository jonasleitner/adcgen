from functools import wraps


class Inputerror(ValueError):
    pass


def cached_member(function):
    """Decorator for a member function thats called with
       at least one argument. The result is cached in the
       member variable '_function_cache' of the class instance.
       """

    from indices import split_idxstring
    from inspect import signature
    fname = function.__name__

    @wraps(function)
    def wrapper(self, *args, **kwargs):
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
        # NOTE: This currently requires the name of all function arguments
        # that describe indices to start with an i
        kwargs_key = {}
        for var, value in kwargs.items():
            # catch orders -> hand over as int
            if isinstance(value, int):
                kwargs_key[var] = value
                continue
            val = transform_to_tuple(value)
            # catch braket -> hand them over as str
            if len(val) == 1 and val[0] in ["bra", "ket"]:
                kwargs_key[var] = value
                continue
            if var == "braket":
                raise Inputerror("Probably a typo in 'bra'/'ket'. Provided "
                                 f"braket string '{value}'.")
            # check whether the str is a space or index
            is_index = False
            for sp in val:
                for letter in sp:
                    # the second condition allows to request the single
                    # index "h".. for whatever reason one should do that
                    # but restricts the name of all arguments that describe
                    # indices to start with an "i"
                    if letter not in ["p,h"] and var[0] == "i":
                        is_index = True
            # sort: indices, idx_pre, idx_is -> hand over as str
            if is_index:
                sorted_idx = []
                for idxstring in val:
                    sorted_idx.append("".join(
                        sorted(split_idxstring(idxstring))
                    ))
                kwargs_key[var] = ",".join(sorted_idx)
            # sort: space, block, mvp_space -> hand over as str
            else:
                sorted_sp = []
                for sp in val:
                    sorted_sp.append("".join(sorted(sp)))
                kwargs_key[var] = ",".join(sorted_sp)

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
