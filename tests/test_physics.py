"""Catch invalid contact physics before trusting model evaluation scores."""

import pytest

from hud_dreamscale import contract


@pytest.mark.parametrize("height", [0.9000337, 0.85, -0.00159, float("nan")])
def test_penetrating_box_cannot_pass_runtime_validation(height):
    # Known bad emulated runtime: a 1 cm thick box centered at 0.90003 m
    # penetrates a platform whose upper surface is at 0.900 m.
    check = getattr(contract, "validate_contact_height", None)
    assert callable(check), "Runtime must reject physically invalid contact results"
    with pytest.raises(RuntimeError, match="physics"):
        check(height)


def test_supported_contact_tolerance():
    check = getattr(contract, "validate_contact_height", None)
    assert callable(check), "Runtime must validate contact results"
    check(0.9047844404707903)
