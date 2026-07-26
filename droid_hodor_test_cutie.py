#!/usr/bin/env python3
import argparse

from post_process_scene_cutie import main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CUTIE-tracked HODOR representations from DROID episodes."
    )
    parser.add_argument("--config", required=True, help="Path to a task YAML file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
