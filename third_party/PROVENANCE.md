# Provenance of the vendored Time-Series-Library

The `tslib/` directory is a partial, unmodified copy of

    https://github.com/thuml/Time-Series-Library

vendored so that the study is reproducible from this repository alone, and so that
the code quoted in the paper can be checked against a fixed byte sequence rather
than against a moving branch.

Retrieved from branch `main`, accessed 17 September 2026.

The file quoted in Listing 1 of the paper:

    data_provider/data_factory.py
    sha256 8034b4a9b624d358e0a33fe584b3575c9b61c8aff6ee30537a865a126b63ad9a

Verify with:

    sha256sum third_party/tslib/data_provider/data_factory.py

The two lines the paper depends on are, verbatim:

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False

i.e. every split except `test` is shuffled, and the last partial batch is kept for
long-term forecasting. No line of the vendored library was edited; the audit
instrumentation lives in `src/` and wraps the library from outside.
