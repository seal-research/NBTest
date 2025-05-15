## nbtest
#### A pytest plugin for testing Jupyter Notebooks

The `nbtest-plugin` adds the functionality to define tests (assertions) in Jupyter Notebooks for testing key properties.
These tests are later collected by pytest, when used with the `--nbtest` flag

**Assertions supported**

*Generic assertions*
- assert_equal
- assert_allclose
- assert_true
- assert_false

*DataFrame-specific assertions*
- assert_nanvar: Test the variance of the dataframe
- assert_nanmean: Test the mean of the dataframe
- assert_column_types: Test the data type of each column in the dataframe
- assert_column_names: Test the column names of each column in the dataframe

#### Usage example

##### Testing

```py
import nbtest
import math
import numpy as np

nbtest.assert_equal(round(math.pi, 2), 3.14)
nbtest.assert_true(math.pi == 3)

print(f'PI: {math.pi}')
```

These tests do not report any errors when the notebook is executed

```bash
$ jupyter execute example.ipynb --output=run.ipynb
[NbClientApp] Executing example.ipynb
[NbClientApp] Executing notebook with kernel:
[NbClientApp] Save executed results to run.ipynb
```

And output of the cell is:
```
PI: 3.141592653589793
```

Now, we execute the tests using pytest
```bash
$ pytest -v --nbtest example/nbtest_plugin_example.ipynb
==================================================== test session starts =====================================================
platform linux -- Python 3.11.11, pytest-8.3.4, pluggy-1.5.0 -- /home/verve/miniconda3/envs/test/bin/python
cachedir: .pytest_cache
rootdir: /home/verve/ml_nb_testing/comp_analysis/nbtest-release
plugins: nbtest-0.1.0, anyio-4.6.2
collected 2 items

example/nbtest_plugin_example.ipynb::nbtest_id_0_5 PASSED                                                              [ 50%]
example/nbtest_plugin_example.ipynb::nbtest_id_0_6 FAILED                                                              [100%]

========================================================== FAILURES ==========================================================
________________________________________ example/nbtest_plugin_example.ipynb::Cell 0 _________________________________________
Assertion failed
Cell 0: Assertion error

Input:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
Cell In[1], line 1
----> 1 nbtest.tc.assertTrue(math.pi == 3)

File ~/miniconda3/envs/test/lib/python3.11/unittest/case.py:715, in TestCase.assertTrue(self, expr, msg)
    713 if not expr:
    714     msg = self._formatMessage(msg, "%s is not true" % safe_repr(expr))
--> 715     raise self.failureException(msg)

AssertionError: False is not true

================================================== short test summary info ===================================================
FAILED example/nbtest_plugin_example.ipynb::nbtest_id_0_6
================================================ 1 failed, 1 passed in 1.34s =================================================

```
