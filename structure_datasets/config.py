import difflib
import hashlib
import json
from typing import Union, Dict, Tuple, List

_store: Dict[str, Tuple[Union[bool, int, float, str], bool, bool, bool]] = {}


def _did_you_mean(identifier: str, identifiers: List[str]) -> List[str]:
    return difflib.get_close_matches(identifier, identifiers)


def get(identifier: str) -> Union[bool, int, float, str]:
    """
    Get the configuration field identified by the given identifier.

    This method retrieves the configuration field that is identified by the given identifier. It raises a TypeError if
    the identifier is not a string and a KeyError if no configuration field with the given identifier exists.
    Furthermore, the method sets the `accessed` flag to `True`.

    :param identifier: identifier of the configuration field
    :return: value of the configuration field
    """
    if not isinstance(identifier, str):
        raise TypeError(f'The identifier "{identifier}" is not a string!')

    if identifier not in _store.keys():
        alternatives: str = ", ".join(_did_you_mean(identifier, list(_store.keys())))
        raise KeyError(f'The identifier "{identifier}" does not exist! - Did you mean {alternatives}?')

    value, forced, include_in_hash, accessed = _store[identifier]

    if not accessed:
        _store[identifier] = (value, forced, include_in_hash, True)

    return value


def set(identifier: str, value: Union[bool, int, float, str],
        forced: bool = False, include_in_hash: bool = True) -> None:
    """
    Set the configuration field identified by the given identifier to the given value.

    This method sets the configuration field identified by the given identifier to the given value. It raises a
    TypeError in case the identifier is not a string, the value is not a proper configuration value, the `forced`
    parameter is not a bool, or the `include_in_hash` parameter is not a bool. Furthermore, it raises an AssertionError
    in case the value configuration field has already been accessed or the method call tries to force-set a
    configuration field that has already been force-set.

    :param identifier: identifier of the configuration field
    :param value: new value of the configuration field
    :param forced: whether to force-set the configuration field
    :param include_in_hash: whether to include the configuration field in the hash
    """
    if not isinstance(identifier, str):
        raise TypeError(f'The identifier "{identifier}" is not a string!')

    if not (isinstance(value, bool) or isinstance(value, int) or isinstance(value, float) or isinstance(value, str)):
        raise TypeError(f'The value "{value}" has an invalid type "{type(value)}"!')

    if not isinstance(forced, bool):
        raise TypeError(f'The forced parameter value "{forced}" is not a bool!')

    if not isinstance(include_in_hash, bool):
        raise TypeError(f'The include_in_hash parameter "{include_in_hash}" is not a bool!')

    if identifier not in _store.keys():
        _store[identifier] = (value, forced, include_in_hash, False)
    else:
        old_value, old_forced, old_include_in_hash, old_accessed = _store[identifier]

        if old_accessed:
            raise AssertionError(f'The value of "{identifier}" cannot be changed after it has been accessed!')

        if old_forced and forced:
            raise AssertionError(f'The new value of "{identifier}" cannot be forced since the old value has already'
                                 f'been forced!')

        if not old_forced:
            _store[identifier] = (value, forced, include_in_hash, False)


def load_config(config: Dict[str, Union[bool, int, float, str]],
                forced: bool = False, include_in_hash: bool = True) -> None:
    """
    Load the given dictionary of configuration fields.

    This method loads the configuration fields in the given dictionary. It raises a TypeError if the configuration
    dictionary is not a dictionary, the `forced` parameter is not a bool, or the `include_in_hash` parameter is not a
    bool. Furthermore, it raises a TypeError if a configuration field identifier is not a string or a configuration
    field value is not a proper configuration field value. Finally, it raises an AssertionError in case the value of a
    configuration field has already been accessed or the method call tries to force-set a configuration field that has
    already been force-set.

    :param config: dictionary of configuration fields to load
    :param forced: whether to force-set the configuration fields
    :param include_in_hash: whether to include the configuration fields in the hash
    """
    if not isinstance(config, dict):
        raise TypeError(f'The config parameter value "{config}" is not a dict!')

    if not isinstance(forced, bool):
        raise TypeError(f'The forced parameter value "{forced}" is not a bool!')

    if not isinstance(include_in_hash, bool):
        raise TypeError(f'The include_in_hash parameter value "{include_in_hash}" is not a bool!')

    for identifier, value in config.items():
        set(identifier, value, forced, include_in_hash)


def load_config_json(config_json: str,
                     forced: bool = False, include_in_hash: bool = True) -> None:
    """
    Load the given JSON string of configuration fields.

    This method loads the configuration fields in the given JSON string. It raises a TypeError if the configuration
    JSON string is not a string, the `forced` parameter is not a bool, or the `include_in_hash` parameter is not a
    bool. Furthermore, it raises a TypeError if a configuration field identifier is not a string or a configuration
    field value is not a proper configuration field value. It raises an AssertionError in case the value of a
    configuration field has already been accessed or the method call tries to force-set a configuration field that has
    already been force-set. Finally, it raises errors in case the JSON deserialization fails.

    :param config_json: JSON string of configuration fields to set
    :param forced: whether to force-set the configuration fields
    :param include_in_hash: whether to include the configuration fields in the hash
    """
    if not isinstance(config_json, str):
        raise TypeError(f'The config_json parameter value "{config_json}" is not a string!')

    config = json.loads(config_json)
    load_config(config, forced, include_in_hash)


def load_config_json_file(file_path,
                          forced: bool = False, include_in_hash: bool = True) -> None:
    """
    Load the given JSON file of configuration fields.

    This method loads the configuration fields in the given JSON file. It raises a TypeError if the `forced` parameter
    is not a bool, or the `include_in_hash` parameter is not a bool. Furthermore, it raises a TypeError if a
    configuration field identifier is not a string or a configuration field value is not a proper configuration field
    value. It raises an AssertionError in case the value of a configuration field has already been accessed or the
    method call tries to force-set a configuration field that has already been force-set. Finally, it raises errors in
    case the file reading or the JSON deserialization fails.

    :param file_path: file path of the JSON file with the configuration fields
    :param forced: whether to force-set the configuration fields
    :param include_in_hash: whether to include the configuration fields in the hash
    """
    with open(file_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    load_config(config, forced, include_in_hash)


def get_config(accessed_only: bool = True, forced_only: bool = False,
               include_in_hash_only: bool = False) -> Dict[str, Union[bool, int, float, str]]:
    """
    Get the current configuration as a dictionary.

    This method returns the current configuration in a dictionary. It raises a TypeError if the `accessed_only`
    parameter is not a bool, the `forced_only` parameter is not a bool, or the `include_in_hash_only` parameter is not a
    bool.

    :param accessed_only: whether to only include attributes that have already been accessed
    :param forced_only: whether to only include attributes that have been force-set
    :param include_in_hash_only: whether to only include attributes that should be included in the hash
    :return: dictionary of configuration fields
    """
    if not isinstance(accessed_only, bool):
        raise TypeError(f'The accessed_only parameter value "{accessed_only}" is not a bool!')

    if not isinstance(forced_only, bool):
        raise TypeError(f'The forced_only parameter value "{forced_only}" is not a bool!')

    if not isinstance(include_in_hash_only, bool):
        raise TypeError(f'The include_in_hash_only parameter value "{include_in_hash_only}" is not a bool!')

    config = {}
    for key, (value, forced, include_in_hash, accessed) in _store.items():
        if accessed_only and not accessed:
            continue
        if forced_only and not forced:
            continue
        if include_in_hash_only and not include_in_hash:
            continue

        config[key] = get(key)  # use get to ensure that the accessed flag is set

    return config


def get_config_json(accessed_only: bool = True, forced_only: bool = False, include_in_hash_only: bool = False,
                    indent: Union[int, None] = 4) -> str:
    """
    Get the current configuration as a JSON string.

    This method returns the current configuration in a JSON string. It raises a TypeError if the `accessed_only`
    parameter is not a bool, the `forced_only` parameter is not a bool, the `include_in_hash_only` parameter is not a
    bool, or the `indent` parameter is neither an int nor None.

    :param accessed_only: whether to only include attributes that have already been accessed
    :param forced_only: whether to only include attributes that have been force-set
    :param include_in_hash_only: whether to only include attributes that should be included in the hash
    :param indent: indent for the JSON string
    :return: JSON string of configuration fields
    """
    if not (isinstance(indent, int) or indent is None):
        raise TypeError(f'The indent parameter value "{indent}" is not an int or None!')

    return json.dumps(get_config(accessed_only, forced_only, include_in_hash_only), indent=indent)


def save_config_json_file(file_path, accessed_only: bool = True, forced_only: bool = False,
                          include_in_hash_only: bool = False, indent: Union[int, None] = 4) -> None:
    """
    Save the current configuration to a JSON file.

    This method saves the current configuration to a JSON file. It raises a TypeError if the `accessed_only`
    parameter is not a bool, the `forced_only` parameter is not a bool, the `include_in_hash_only` parameter is not a
    bool, or the `indent` parameter is neither an int nor None. Finally, it raises errors in case the file writing
    fails.

    :param file_path: file path of the JSON file
    :param accessed_only: whether to only include attributes that have already been accessed
    :param forced_only: whether to only include attributes that have been force-set
    :param include_in_hash_only: whether to only include attributes that should be included in the hash
    :param indent: indent for the JSON string
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(get_config(accessed_only, forced_only, include_in_hash_only), file, indent=indent)


def get_config_hash(accessed_only: bool = True, forced_only: bool = False, include_in_hash_only: bool = True) -> str:
    """
    Get a hash value string of the current configuration.

    This method returns a hash value string of the current configuration. It raises a TypeError if the `accessed_only`
    parameter is not a bool, the `forced_only` parameter is not a bool, or the `include_in_hash_only` parameter is not a
    bool.

    :param accessed_only: whether to only include attributes that have already been accessed
    :param forced_only: whether to only include attributes that have been force-set
    :param include_in_hash_only: whether to only include attributes that should be included in the hash
    :return: hash string of the configuration fields
    """
    config_json = get_config_json(accessed_only, forced_only, include_in_hash_only, indent=None)
    h = hashlib.sha256(bytes(config_json, "utf-8")).hexdigest()
    return "-".join(h[i:i + 4] for i in range(0, 16, 4))


def _value_to_mnemonic(value: Union[bool, int, float, str]) -> str:
    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return str(round(value, 8)).replace('.', '-')

    if isinstance(value, str):
        if value == "":
            value = "xxx"

        # split value into parts
        parts = value.split()
        for splitter in ["_", "-", "."]:
            new_parts = []
            for part in parts:
                new_parts += part.split(splitter)
            parts = new_parts

        # lowercase the parts
        parts = [part.lower() for part in parts]

        # remove characters that are not allowed
        allowed = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        new_parts = []
        for part in parts:
            chars_to_keep = []
            for char in part:
                if char in allowed:
                    chars_to_keep.append(char)
            if chars_to_keep != []:
                new_parts.append(''.join(chars_to_keep))
            else:
                new_parts.append('xxx')
        parts = new_parts

        # shorten the parts
        if len(parts) > 1:
            new_parts = []
            for part in parts:
                new_parts.append(part[:6])
        parts = new_parts

        # join the parts and return
        return "-".join(parts)

    raise TypeError(f'The value "{value}" of type "{type(value)}" cannot be transformed to a mnemonic.')


def get_config_mnemonic(identifiers: List[str]) -> str:
    """
    Get a short string describing the values of the configuration fields with the given identifiers.

    This method returns a short string that describes the values of the configuration fields with the given identifiers.
    It raises a TypeError in case the `identifiers` parameter is not a list of strings, and a KeyError in case an
    identifier does not correspond to a configuration field.

    :param identifiers: identifiers of the configuration fields to include
    :return: short string representation of the configuration fields' values
    """
    if not isinstance(identifiers, list):
        raise TypeError(f'The identifiers parameter value "{identifiers}" is not a list!')

    for identifier in identifiers:
        if not isinstance(identifier, str):
            raise TypeError(f'The identifier "{identifier}" is not a string!')

        if identifier not in _store.keys():
            alternatives: str = ", ".join(_did_you_mean(identifier, list(_store.keys())))
            raise KeyError(f'The identifier "{identifier}" does not exist! - Did you mean {alternatives}?')

    parts = []
    for identifier in identifiers:
        value = get(identifier)  # use get to ensure that the accessed flag is set
        parts.append(_value_to_mnemonic(value))

    return "_".join(parts)
