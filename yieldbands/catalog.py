"""Editable stock catalog. Any six-digit mainland A-share symbol is also supported."""

STOCKS: dict[str, str] = {
    "600036": "招商银行",
    "600900": "长江电力",
}


def display_options() -> dict[str, str]:
    return {f"{name} · {symbol}": symbol for symbol, name in STOCKS.items()}
