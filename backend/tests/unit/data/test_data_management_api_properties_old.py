"""Legacy data management API property tests.

These tests target a removed `/api/v1/data/files|stats|sync` contract and are
kept only as historical reference. The current API contract is covered by
`test_data_management_api_properties.py`.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="legacy data management API contract has been replaced by current data routes"
)
