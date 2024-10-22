import feature_engine
import imblearn
import sagemaker
import sklearn

print(f"scikit-learn version: {sklearn.__version__}")
print(f"feature_engine version: {feature_engine.__version__}")
print(f"imblearn version: {imblearn.__version__}")
print(f"sagemaker version: {sagemaker.__version__}")


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
