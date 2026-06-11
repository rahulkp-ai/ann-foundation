# Contributing

Contributions are welcome. This project prioritises correctness and readability.

## Setup

```bash
git clone https://github.com/rahulkp-ai/ann-foundation.git
cd ann-foundation
pip install -e ".[dev]"
```

## Before submitting a PR

- All tests must pass: `pytest tests/ -v`
- New operations must include gradient verification tests
- Code must be formatted with Black: `black src/ tests/`
- Docstrings required on all public functions

## Adding a new operation

1. Implement the forward computation in `src/engine.py`
2. Define the `_backward` closure with correct chain rule
3. Add a test in `tests/test_engine.py` verifying against numerical differentiation
4. Document the gradient formula in the docstring

## Reporting issues

Open a GitHub issue with: the operation that fails, input values, expected gradient, actual gradient.
