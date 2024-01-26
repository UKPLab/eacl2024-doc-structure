import enum
from typing import Optional

import attrs


class _PatternKind(enum.Enum):
    OBJECT = "object"
    ARRAY = "array"
    STRING = "string"
    NUMBER = "number"
    BOOL = "bool"
    NULL = "null"


_flattenable_patterns = [_PatternKind.NULL, _PatternKind.BOOL, _PatternKind.NUMBER, _PatternKind.STRING]
_INDENT = 2
_DIFF_EMPTY_STRING = True
_DIFF_TRUE_FALSE = True


@attrs.define
class _Point:
    parent: Optional["_Point"] = attrs.field()
    key: Optional[str] = attrs.field()
    data: list = attrs.field()
    patterns: list["_Pattern"] = attrs.field(factory=list)

    def to_string(self, depth, single_line):
        out_suf = "" if single_line else " " * depth * _INDENT
        out_nl = "" if single_line else "\n"

        num_patterns = len(self.patterns)
        if num_patterns <= 3 and all(self.patterns[i].kind in _flattenable_patterns for i in range(num_patterns)):
            single_line = True

        if len(self.patterns) == 1 and self.patterns[0].kind == _PatternKind.ARRAY and len(
                self.patterns[0].children) == 1 and len(
                self.patterns[0].children[0].patterns) == 1 \
                and self.patterns[0].children[0].patterns[0].kind in _flattenable_patterns:
            single_line = True

        suf = "" if single_line else " " * depth * _INDENT
        nl = "" if single_line else "\n"

        parts = []
        if self.key is not None:
            parts.append(f"{out_suf}'{self.key}': {nl}")

        for ix, pattern in enumerate(self.patterns):
            if not single_line and len(self.patterns) > 1:
                if not ix == 0:
                    parts.append("\n")
                parts.append(f"{suf}  ({ix + 1}):\n")
            if single_line and len(self.patterns) > 1:
                parts.append(f"({ix + 1}): ")
            parts.append(pattern.to_string(depth + 1, single_line))
            if single_line and ix != len(self.patterns) - 1:
                parts.append(" || ")

        parts.append(f"{out_nl if single_line else ''}")

        return "".join(parts)


@attrs.define
class _Pattern:
    kind: _PatternKind = attrs.field()
    keys: set = attrs.field()
    point: _Point = attrs.field(eq=False)
    data: list = attrs.field(eq=False)
    count: int = attrs.field(default=1, eq=False)
    children: list["_Point"] = attrs.field(factory=list, eq=False)

    def to_string(self, depth, single_line):
        suf = "" if single_line else " " * depth * _INDENT
        nl = "" if single_line else "\n"
        total = sum(p.count for p in self.point.patterns)
        if self.kind == _PatternKind.ARRAY:
            parts = [f"{suf}{self.count}/{total}x [{nl}"]
            for child in self.children:
                parts.append(child.to_string(depth + 1, single_line))
            parts.append(f"{suf}]{nl}")
            return "".join(parts)
        elif self.kind == _PatternKind.OBJECT:
            parts = [f"{suf}{self.count}/{total}x " + "{" + f"{nl}"]
            for child in self.children:
                parts.append(child.to_string(depth + 1, single_line))
            parts.append(suf + "}" + f"{nl}")
            return "".join(parts)
        elif self.kind == _PatternKind.STRING:
            if _DIFF_EMPTY_STRING:
                return f"{suf}{self.count}/{total}x {list(self.keys)[0]} {self.kind.value}{nl}"
            else:
                return f"{suf}{self.count}/{total}x {self.kind.value}{nl}"
        elif self.kind == _PatternKind.NUMBER:
            return f"{suf}{self.count}/{total}x {self.kind.value}{nl}"
        elif self.kind == _PatternKind.BOOL:
            if _DIFF_TRUE_FALSE:
                return f"{suf}{self.count}/{total}x {list(self.keys)[0]} {nl}"
            else:
                return f"{suf}{self.count}/{total}x {self.kind.value}{nl}"
        elif self.kind == _PatternKind.NULL:
            return f"{suf}{self.count}/{total}x {self.kind.value}{nl}"
        else:
            raise ValueError("Unknown pattern kind!")


def _value_to_pattern(value, point):
    if isinstance(value, dict):
        return _Pattern(kind=_PatternKind.OBJECT, keys=set(value.keys()), point=point, data=[value])
    if isinstance(value, list):
        return _Pattern(kind=_PatternKind.ARRAY, keys=[], point=point, data=[value])
    if isinstance(value, str):
        if _DIFF_EMPTY_STRING:
            return _Pattern(kind=_PatternKind.STRING, keys=["empty" if value == "" else "non-empty"], point=point,
                            data=[value])
        else:
            return _Pattern(kind=_PatternKind.STRING, keys=[], point=point, data=[value])
    if isinstance(value, int) or isinstance(value, float):
        return _Pattern(kind=_PatternKind.NUMBER, keys=[], point=point, data=[value])
    if isinstance(value, bool):
        if _DIFF_TRUE_FALSE:
            return _Pattern(kind=_PatternKind.STRING, keys=["true" if value else "false"], point=point, data=[value])
        else:
            return _Pattern(kind=_PatternKind.BOOL, keys=[], point=point, data=[value])
    if value is None:
        return _Pattern(kind=_PatternKind.NULL, keys=[], point=point, data=[value])

    raise ValueError(f"Unknown JSON value! {type(value)}")


def _process_point(point):
    # cluster the values into patterns
    for value in point.data:
        new_pattern = _value_to_pattern(value, point)
        for pattern in point.patterns:
            if pattern == new_pattern:
                pattern.count += 1
                pattern.data.append(value)
                break
        else:
            point.patterns.append(new_pattern)

    # go deeper into objects and arrays
    for pattern in point.patterns:
        if pattern.kind == _PatternKind.ARRAY:
            data = [v for data in pattern.data for v in data]
            child_point = _Point(parent=point, key=None, data=data)
            pattern.children.append(child_point)
            _process_point(child_point)
        elif pattern.kind == _PatternKind.OBJECT:
            for key in pattern.keys:
                data = [v for data in pattern.data for k, v in data.items() if k == key]
                child_point = _Point(parent=point, key=key, data=data)
                pattern.children.append(child_point)
                _process_point(child_point)


def parse_json_object(json_object):
    point = _Point(None, None, [json_object])
    _process_point(point)
    return point.to_string(0, False)
