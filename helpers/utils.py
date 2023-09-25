import regex


def to_snake_case(name):
    name = regex.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = regex.sub("__([A-Z])", r"_\1", name)
    name = regex.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()
