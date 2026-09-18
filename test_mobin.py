import pytest

from main import DirectiveInterpretation, StructuredAdjustment, validate_directives


def test_mobin_rejects_no_op_with_adjustment():
    item = DirectiveInterpretation(
        note_index=0,
        applies=False,
        directive_type="no_op",
        structured_adjustment=StructuredAdjustment(hours=[1]),
        explanation="invalid no-op",
    )
    with pytest.raises(ValueError):
        validate_directives([item], 1, 20)
