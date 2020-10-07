def decode_search_string(search_string, **args):
    """Convert the search string at the end of the URL into a dict."""

    # Additional kwargs are set by default

    # Search string is empty
    if search_string is None or search_string == "?" or search_string == "":
        return args

    # Make sure that the string starts with "?"
    assert search_string.startswith("?"), search_string

    # Parse the fields
    for field in search_string[1:].split("&"):
        assert "=" in field, field
        # Parse out the key and value
        k, v = field.split("=", 1)

        # Set the key-value pair
        args[k] = v

    # Return all remaining arguments
    return args


def encode_search_string(args):
    """Convert a dict into a search string."""
    return "?{}".format("&".join([
        "{}={}".format(k, v)
        for k, v in args.items()
    ]))
