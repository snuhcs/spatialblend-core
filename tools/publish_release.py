# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

from util import *


def get_argument_parser(desc: str):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--version", type=str, default="v0.0.1",
                        help="Version string")
    parser.add_argument("--caption", type=str, default=None,
                        help="Optional caption for the tag")

    return parser


def main():
    parser = get_argument_parser("Publish release")
    args = parser.parse_args()

    if not args.version.startswith("v"):
        raise ValueError("Version must start with 'v'")

    # Check if the version of tag exists
    ret = run_cmd(f"git tag -l {args.version}")
    if ret != 0:
        print(f"Deleting existing tag {args.version}")
        run_cmd(f"git tag --delete {args.version}")

    # Create a new tag and push it to the remote repository (github action will automatically create a release)
    if args.caption:
        print(
            f"Creating annotated tag {args.version} with caption: {args.caption}")
        run_cmd(f"git tag -a {args.version} -m \"{args.caption}\"")
    else:
        print(f"Creating tag {args.version}")
        run_cmd(f"git tag {args.version}")

    print(f"Pushing tag {args.version} to remote")
    run_cmd(f"git push -f origin {args.version}")


if __name__ == "__main__":
    main()
