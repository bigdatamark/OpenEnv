# [PACKAGE DOC]
# Template package root for a generated environment (placeholders such as
# __ENV_CLASS_NAME__ are substituted on scaffolding). Re-exports the
# environment client and its Action/Observation models.
# SPDX-License-Identifier: BSD-3-Clause

"""__ENV_TITLE_NAME__ Environment."""

from .client import __ENV_CLASS_NAME__Env
from .models import __ENV_CLASS_NAME__Action, __ENV_CLASS_NAME__Observation

__all__ = [
    "__ENV_CLASS_NAME__Action",
    "__ENV_CLASS_NAME__Observation",
    "__ENV_CLASS_NAME__Env",
]
