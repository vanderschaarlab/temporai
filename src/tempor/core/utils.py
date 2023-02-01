def get_class_full_name(o: object):
    # See: https://stackoverflow.com/a/2020083
    class_ = o.__class__
    module = class_.__module__
    if module == "builtins":
        return class_.__qualname__  # avoid outputs like "builtins.str"
    return module + "." + class_.__qualname__
