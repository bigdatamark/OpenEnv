# [PACKAGE DOC]
# HELPER: Minimal stand-in for the ORS/OpenReward SDK used by the importer example: attribute-bag model classes (Split,
# ToolSpec, ListToolsOutput, TextBlock, ToolOutput), tool-run result wrappers and an Environment base class with
# task spec, secrets and no-op setup/teardown.
class Model:
    # [CLASS DOC]
    # Attribute bag: stores all keyword arguments as instance attributes.
    def __init__(self, **kwargs):
        # [FUNCTION DOC]
        # Sets every keyword argument as an attribute.
        self.__dict__.update(kwargs)

    def model_dump(self):
        # [FUNCTION DOC]
        # Returns a shallow copy of the instance attributes as a dict.
        return dict(self.__dict__)


class Split(Model):
    # [CLASS DOC]
    # Model describing a dataset split (attributes given as keywords).
    pass


class ToolSpec(Model):
    # [CLASS DOC]
    # Model describing a tool (name, description, input schema given as keywords).
    pass


class ListToolsOutput(Model):
    # [CLASS DOC]
    # Model wrapping a list of tool specs.
    pass


class TextBlock(Model):
    # [CLASS DOC]
    # Text content block with text, optional detail and type "text".
    def __init__(self, text, detail=None, type="text"):
        # [FUNCTION DOC]
        # Stores text, detail and type as attributes.
        super().__init__(text=text, detail=detail, type=type)


class ToolOutput(Model):
    # [CLASS DOC]
    # Model holding a tool's output (blocks, metadata, reward, finished flag given as keywords).
    pass


class RunToolSuccess:
    # [CLASS DOC]
    # Successful tool-run wrapper with ok set to True and the output.
    ok = True

    def __init__(self, output):
        # [FUNCTION DOC]
        # Stores the output.
        self.output = output


class RunToolOutput:
    # [CLASS DOC]
    # Tool-run result whose root attribute is a RunToolSuccess wrapping the output.
    def __init__(self, output):
        # [FUNCTION DOC]
        # Wraps the output in a RunToolSuccess stored as root.
        self.root = RunToolSuccess(output)


class Environment:
    # [CLASS DOC]
    # Base environment holding a task spec dict and secrets dict, with no-op lifecycle hooks.
    def __init__(self, task_spec=None, secrets=None):
        # [FUNCTION DOC]
        # Stores the task spec and secrets, defaulting to empty dicts.
        self.task_spec = task_spec or {}
        self.secrets = secrets or {}

    def setup(self):
        # [FUNCTION DOC]
        # No-op setup hook.
        pass

    def teardown(self):
        # [FUNCTION DOC]
        # No-op teardown hook.
        pass
