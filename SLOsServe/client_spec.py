from __future__ import annotations


def _split_client_tokens(raw_clients: str) -> list[str]:
    return [token.strip() for token in raw_clients.split(",") if token.strip()]


def parse_client_spec(raw_clients: str | None) -> list[str]:
    if raw_clients is None:
        return []

    raw_clients = raw_clients.strip()
    if not raw_clients:
        return []

    if "://" in raw_clients:
        return _split_client_tokens(raw_clients)

    if "-" in raw_clients and ":" not in raw_clients and "," not in raw_clients:
        left, right = raw_clients.split("-", 1)
        if left.isdigit() and right.isdigit():
            start = int(left)
            end = int(right)
            if end < start:
                return []
            return [f"r{i}" for i in range(start, end + 1)]

    if ":" in raw_clients and "," not in raw_clients:
        left, right = raw_clients.split(":", 1)
        if left.isdigit() and right.isdigit():
            start = int(left)
            count = int(right)
            if count <= 0:
                return []
            if start < 10:
                return [f"r{i}" for i in range(start, start + count)]
            return [f"http://localhost:{start + i}" for i in range(count)]

    tokens = _split_client_tokens(raw_clients)
    if not tokens:
        return []
    if all(token.isdigit() for token in tokens) and int(tokens[0]) < 10:
        return [f"r{int(token)}" for token in tokens]
    return tokens


def normalize_client_spec(raw_clients: str | None) -> str | None:
    clients = parse_client_spec(raw_clients)
    if not clients:
        return None
    return ",".join(clients)


def count_client_spec(raw_clients: str | None) -> int | None:
    clients = parse_client_spec(raw_clients)
    if not clients:
        return None
    return len(clients)
