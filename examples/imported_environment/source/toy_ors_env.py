# [MODULE DOC]
# Toy ORS-style environment used as importer input: a single "train" split with one addition question and an "answer"
# tool that rewards exactly matching the expected answer and finishes the episode.
from ors import (
    Environment,
    ListToolsOutput,
    RunToolOutput,
    Split,
    TextBlock,
    ToolOutput,
    ToolSpec,
)


class ToyMathEnvironment(Environment):
    # [CLASS DOC]
    # ORS environment with class-level split/task/tool listings and per-instance prompt and tool execution.
    @classmethod
    def list_splits(cls):
        # [FUNCTION DOC]
        # Returns a list with a single "train" split.
        return [Split(name="train", type="train")]

    @classmethod
    def list_tasks(cls, split):
        # [FUNCTION DOC]
        # Returns the single-task list (id, question "What is 2 + 2?", answer "4") regardless of split.
        return [
            {
                "id": "addition-1",
                "question": "What is 2 + 2?",
                "answer": "4",
            }
        ]

    @classmethod
    def num_tasks(cls, split):
        # [FUNCTION DOC]
        # Returns the number of tasks in the split.
        return len(cls.list_tasks(split))

    @classmethod
    def get_task(cls, split, index):
        # [FUNCTION DOC]
        # Returns the task at the given index (IndexError if out of range).
        return cls.list_tasks(split)[index]

    @classmethod
    def get_task_range(cls, split, start=None, stop=None):
        # [FUNCTION DOC]
        # Returns the slice of tasks between start and stop.
        return cls.list_tasks(split)[slice(start, stop)]

    @classmethod
    def list_tools(cls):
        # [FUNCTION DOC]
        # Returns a ListToolsOutput with one "answer" tool taking a required string "value".
        return ListToolsOutput(
            tools=[
                ToolSpec(
                    name="answer",
                    description="Submit an answer for the math question",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                )
            ]
        )

    def list_task_tools(self):
        # [FUNCTION DOC]
        # Returns an empty task-specific tool list.
        return ListToolsOutput(tools=[])

    def get_prompt(self):
        # [FUNCTION DOC]
        # Returns the prompt as a single text block containing the task question. Raises KeyError if the task spec lacks a question.
        return [TextBlock(text=self.task_spec["question"])]

    def _call_tool(self, name, input):
        # [FUNCTION DOC]
        # Compares the stripped string "value" input with the expected answer and returns a RunToolOutput whose ToolOutput has a
        # "correct"/"incorrect" text block, metadata with expected and received values, reward 1.0 or 0.0, and finished True.
        # The tool name is ignored.
        guess = str(input.get("value", "")).strip()
        expected = self.task_spec["answer"]
        correct = guess == expected
        return RunToolOutput(
            ToolOutput(
                blocks=[TextBlock(text="correct" if correct else "incorrect")],
                metadata={"expected": expected, "received": guess},
                reward=1.0 if correct else 0.0,
                finished=True,
            )
        )
