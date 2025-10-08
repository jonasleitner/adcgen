import re


def split_idx_string(to_split: str) -> list[str]:
    """
    Splits an index string of the form 'ij12a3b' in a list ['i','j12','a3','b']
    """
    # findall only returns the matching parts of the string
    # -> ensure that we don't loose part of the string
    #    (string starting with a numnber)
    splitted = re.findall(r"\D\d*", to_split)
    if "".join(splitted) != to_split:
        raise ValueError(f"Invalid index string {to_split}")
    return splitted
