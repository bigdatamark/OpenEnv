# [MODULE DOC]
# Thin launcher: runs the packaged ThinkingBox evaluator's main function when executed
# as a script from a source checkout. No definitions of its own.
"""Run the packaged ThinkingBox evaluator from a source checkout."""

from thinkingbox_env.evaluation import main


if __name__ == "__main__":
    main()
