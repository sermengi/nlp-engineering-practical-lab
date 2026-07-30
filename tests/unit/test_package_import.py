import pytest

import nlp_lab


@pytest.mark.unit
def test_package_imports() -> None:
    assert nlp_lab.__doc__ == "NLP engineering practical lab package."
