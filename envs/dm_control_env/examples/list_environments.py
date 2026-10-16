#!/usr/bin/env python3
# [MODULE DOC]
# Utility script printing all dm_control.suite benchmarking (domain, task) pairs grouped by domain, with each
# task's action dimension and observation keys.
"""List all available dm_control.suite environments.

This utility prints all available domain/task combinations from dm_control.suite.
"""

from dm_control import suite


def main():
    # [FUNCTION DOC]
    # Loads every benchmarking environment to read its specs (errors printed per task), closes it, and prints the
    # grouped listing and total count. Side effects: stdout, loads MuJoCo models.
    print("Available dm_control.suite environments:")
    print("=" * 50)

    # Group by domain
    domains = {}
    for domain, task in suite.BENCHMARKING:
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(task)

    for domain in sorted(domains.keys()):
        tasks = sorted(domains[domain])
        print(f"\n{domain}:")
        for task in tasks:
            # Load env to get action spec
            try:
                env = suite.load(domain_name=domain, task_name=task)
                action_spec = env.action_spec()
                action_dim = action_spec.shape[0]
                obs_keys = list(env.observation_spec().keys())
                env.close()
                print(f"  - {task:20s} (action_dim={action_dim}, obs={obs_keys})")
            except Exception as e:
                print(f"  - {task:20s} (error: {e})")

    print("\n" + "=" * 50)
    print(f"Total: {len(suite.BENCHMARKING)} environments")


if __name__ == "__main__":
    main()
