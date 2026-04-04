# Development vs Public Version

First, set up two different virtual environments:

1. Development environment with py-feat installed using `pip install -e .` to track changes
2. Base environment with py-feat installed from PyPi (or some other baseline version)

Run the `dev_base.py` benchmarks with both versions to compare results.

# Wrapper Classes Effectiveness

Run `benchmarks_run.py` to compare baseline performance to the performance when using the wrapper classes.
