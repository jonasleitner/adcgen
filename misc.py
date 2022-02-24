from functools import wraps
from indices import split_idxstring


def cached_member(function):
    """Decorator for a member function thats called with
       at least one argument. The result is cached in the
       member variable '_function_cache' of the class instance.
       """

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

        # Indices should be in kwargs -> sort them before using
        # them as key in the cache.
        # if indices in args, its also fine, however they are not
        # sorted which may cause additional calculations.
        for indices in kwargs.values():
            if type(indices) != str:
                print("Indices provided as kwargs musst be of type str "
                      f"not {type(indices)}.")
                exit()
            idx = indices.split(",")
            sorted_idx = []
            for idxstring in idx:
                sorted_idx.append("".join(
                    sorted(split_idxstring(idxstring))
                ))
            args += (",".join(sorted_idx),)

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
