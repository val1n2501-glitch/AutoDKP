from autodkp.security.access import is_allowed_user


def test_allowlist_is_default_deny() -> None:
    assert not is_allowed_user(101, "private", frozenset())


def test_only_numeric_id_in_private_chat_is_allowed() -> None:
    allowed = frozenset({101})
    assert is_allowed_user(101, "private", allowed)
    assert not is_allowed_user(202, "private", allowed)
    assert not is_allowed_user(101, "group", allowed)
    assert not is_allowed_user(None, "private", allowed)
