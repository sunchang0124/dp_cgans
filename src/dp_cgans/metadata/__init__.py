"""Metadata module."""

# from sdv.metadata import visualization
# from sdv.metadata.dataset import Metadata
# from sdv.metadata.errors import MetadataError, MetadataNotFittedError
# from sdv.metadata.table import Table

# __all__ = (
#     'Metadata',
#     'MetadataError',
#     'MetadataNotFittedError',
#     'Table',
#     'visualization'
# )


from dp_cgans.metadata import visualization
from dp_cgans.metadata.errors import MetadataNotFittedError
from dp_cgans.metadata.table import Table

__all__ = (
    'MetadataNotFittedError',
    'visualization',
    'Table'
)
