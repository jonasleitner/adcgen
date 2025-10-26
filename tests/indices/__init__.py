from adcgen.indices import IndexSpace, Indices


# Inherits from IndexSpace to create an independent cache such that
# the tests in this module are completely independent from other tests.
# Otherwise an error could be raised since e.g. a space 'occ' might
# already have been created.
class IsolatedIndexSpace(IndexSpace):
    _available_spaces = tuple()


# Inherit from Indices to create an independent Index cache that
# can be used in conjunction with the IsolatedIndexSpace
# without corrupting the cache of the Indices class.
# However, with this solution it is not possible to provide
# spaces as string through the API on Indices, since the class
# explicitly makes calls to the IndexSpace class!
class IsolatedIndices(Indices):
    pass


# Before running a test (implemented as method on a class):
# ensure that no spaces are cached from another test.
# After running the test: clear the cached spaces.
class EnsureClearedSpaces:
    def setup_method(self):
        # ensure that no space is left from another test
        assert not IsolatedIndexSpace._available_spaces

    def teardown_method(self):
        # clear all spaces from the cache so the different
        # tests don't interfere
        IsolatedIndexSpace.erase_cache()
