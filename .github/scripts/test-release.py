#!/usr/bin/env python3
import json
import subprocess


def get_last_version() -> str:
    """Return the version number of the last release."""
    json_string = (
        subprocess.run(
            "gh release list | grep -E '^.+\\.dev\\d*' | head -n 1 | awk '{print $2}",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        .stdout.decode("utf8")
        .strip()
    )
    
    print('checking test tag version')
    print(json_string)
    print('checking test tag version')

    return json.loads(json_string)["tagName"]


def bump_patch_number(version_number: str) -> str:
    """Return a copy of `version_number` with the patch number incremented."""
    major, minor, patch = version_number.split(".")
    
    new_patch_version = int(patch[3:]) + 1

    return f"{major}.{minor}.dev{new_patch_version}"


def create_new_patch_release():
    """Create a new patch release on GitHub."""
    try:
        last_version_number = get_last_version()
    except subprocess.CalledProcessError as err: 
        # The project doesn't have any releases yet.
        new_version_number = "0.0.dev1"
        print('taking default version')
    else:
        new_version_number = bump_patch_number(last_version_number)
        
    import sys
    sys.exit(7)

    subprocess.run(
        ["gh", "release", "create", "--generate-notes", new_version_number],
        check=True,
    )


if __name__ == "__main__":
    create_new_patch_release()