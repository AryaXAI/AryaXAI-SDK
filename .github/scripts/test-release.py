#!/usr/bin/env python3
import json
import subprocess

default_version = "0.0.1dev0"

def get_last_version() -> str:
    """Return the version number of the last release."""
    tag_info = (
        subprocess.run(
            "gh release list | grep -E '^.+\\.dev0' | head -n 1",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        .stdout.decode("utf8")
        .strip()
    )
    print(tag_info)
    tag_fields = tag_info.split('\t')
    tag = tag_fields[2] if len(tag_fields) > 2 else default_version
    print(tag)
    return tag

def bump_patch_number(version_number: str) -> str:
    """
    Return a copy of `version_number` with the patch number incremented.
    """
    version_number = version_number.replace("dev", ".")
    major, minor, patch, dev_patch = version_number.split(".")
    new_patch_version = int(patch) + 1

    return f"{major}.{minor}.{new_patch_version}dev{dev_patch}"


def create_new_patch_release():
    """Create a new patch release on GitHub."""
    try:
        last_version_number = get_last_version()
    except subprocess.CalledProcessError as err: 
        # The project doesn't have any releases yet.
        new_version_number = default_version
        print(err)
        print(f'taking default version: {new_version_number}')
    else:
        new_version_number = bump_patch_number(last_version_number)

    subprocess.run(
        ["gh", "release", "create", "--generate-notes", new_version_number],
        check=True,
    )


if __name__ == "__main__":
    create_new_patch_release()