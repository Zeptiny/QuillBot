"""Assertion helper shared by the tests."""


def check(name, cond, detail=''):
    """Assert `cond`, labelling the failure with `name` and the value seen."""
    assert cond, f'{name}: {detail}' if detail != '' else name
