from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("align_toolbox")
except PackageNotFoundError:
    __version__ = "unknown"
