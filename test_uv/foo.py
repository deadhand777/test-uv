import awswrangler as wr

print(f"awswrangler version: {wr.__version__}")


def foo(bar: str) -> str:
    """Summary line.

    Extended description of function. Extended description of function.

    Args:
        bar: Description of input argument.

    Returns:
        Description of return value
    """

    return bar


if __name__ == "__main__":  # pragma: no cover
    pass
