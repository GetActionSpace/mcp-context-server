"""The per-entry access fields and generation state are required keys of every entry-carrying read response type."""

import pytest

from app.types import ContextEntryDict
from app.types import EntryAccessFields
from app.types import FtsSearchResultDict
from app.types import GrepContentMatchDict
from app.types import GrepCountDict
from app.types import GrepFileDict
from app.types import HybridSearchResultDict
from app.types import NavigateContextResultDict
from app.types import ReadContextRangeDict
from app.types import SemanticSearchResultDict

ACCESS_FIELDS = frozenset({'visibility', 'is_owner'})

# The required keys of every response type that carries one entry's access fields and generation state.
REQUIRED_KEYS: dict[str, frozenset[str]] = {
    'ContextEntryDict': ContextEntryDict.__required_keys__,
    'NavigateContextResultDict': NavigateContextResultDict.__required_keys__,
    'GrepFileDict': GrepFileDict.__required_keys__,
    'GrepContentMatchDict': GrepContentMatchDict.__required_keys__,
    'GrepCountDict': GrepCountDict.__required_keys__,
    'ReadContextRangeDict': ReadContextRangeDict.__required_keys__,
    'FtsSearchResultDict': FtsSearchResultDict.__required_keys__,
    'SemanticSearchResultDict': SemanticSearchResultDict.__required_keys__,
    'HybridSearchResultDict': HybridSearchResultDict.__required_keys__,
}


@pytest.mark.parametrize('required', list(REQUIRED_KEYS.values()), ids=list(REQUIRED_KEYS))
def test_access_fields_are_required(required: frozenset[str]) -> None:
    """A response type that omits either access field fails its output schema, so both are required keys."""
    assert ACCESS_FIELDS.issubset(required)


@pytest.mark.parametrize('required', list(REQUIRED_KEYS.values()), ids=list(REQUIRED_KEYS))
def test_generation_status_is_required(required: frozenset[str]) -> None:
    """Every entry-carrying read reports the entry's generation state, so it is a required key."""
    assert 'generation_status' in required


def test_entry_access_fields_holds_only_the_access_fields() -> None:
    """The shared access-fields type carries exactly the two fields and never the owner id."""
    assert EntryAccessFields.__required_keys__ == ACCESS_FIELDS
    assert EntryAccessFields.__optional_keys__ == frozenset()
