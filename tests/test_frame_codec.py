from capture.frame_codec import (
    escape_payload, unescape_payload, IndexEntry,
    encode_index_entry, decode_index_entry,
)


def test_clean_payload_is_untouched():
    payload = '{"e":"depthUpdate","U":1,"u":2}'
    escaped, was_escaped = escape_payload(payload)
    assert escaped == payload
    assert was_escaped is False


def test_newline_is_escaped_and_roundtrips():
    payload = '{"a":"x\ny"}'
    escaped, was_escaped = escape_payload(payload)
    assert was_escaped is True
    assert "\n" not in escaped
    assert unescape_payload(escaped) == payload


def test_carriage_return_is_escaped_and_roundtrips():
    payload = '{"a":"x\r\ny"}'
    escaped, was_escaped = escape_payload(payload)
    assert was_escaped is True
    assert "\r" not in escaped and "\n" not in escaped
    assert unescape_payload(escaped) == payload


def test_backslash_roundtrips_without_false_escape():
    payload = r'{"a":"C:\path"}'
    escaped, was_escaped = escape_payload(payload)
    assert unescape_payload(escaped) == payload


def test_index_entry_roundtrips():
    entry = IndexEntry(n=7, t_recv_ns=123, t_exch_ms=456,
                       seq={"U": 1, "u": 2}, kind="data", esc=False)
    assert decode_index_entry(encode_index_entry(entry)) == entry


def test_index_entry_allows_missing_exchange_time():
    entry = IndexEntry(n=0, t_recv_ns=1, t_exch_ms=None,
                       seq=None, kind="control", esc=False)
    assert decode_index_entry(encode_index_entry(entry)) == entry
