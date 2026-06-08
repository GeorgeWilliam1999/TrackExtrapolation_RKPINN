"""Tag-guard tests. Smoke-level check that REQUIRED_TAGS is enforced."""
import pytest

from for_allen.tracking.check_tags import REQUIRED_TAGS, MissingTagError, assert_required_tags


def _full_tags():
    return {k: f"<{k}>" for k in REQUIRED_TAGS}


def test_full_tagset_passes():
    assert_required_tags(_full_tags())


@pytest.mark.parametrize("missing", REQUIRED_TAGS)
def test_missing_tag_raises(missing):
    tags = _full_tags()
    del tags[missing]
    with pytest.raises(MissingTagError):
        assert_required_tags(tags)


def test_empty_value_treated_as_missing():
    tags = _full_tags()
    tags["git_sha"] = ""
    with pytest.raises(MissingTagError):
        assert_required_tags(tags)
