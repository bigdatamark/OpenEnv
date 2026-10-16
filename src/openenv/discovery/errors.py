# [MODULE DOC]
# Error type for the discovery catalog.
# SPDX-License-Identifier: BSD-3-Clause


class CatalogError(ValueError):
    # [CLASS DOC]
    # Raised (a ValueError subclass) whenever a catalog cannot be built, validated,
    # serialized or loaded without violating its schema/contract. No extra fields.
    """A catalog cannot be produced or consumed without losing its contract."""
