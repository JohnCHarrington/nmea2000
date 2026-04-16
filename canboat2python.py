import json
import os
import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Load the JSON data
with open('canboat.json') as f:
    json_data = json.load(f)

def bits_to_hex(len: int) -> str:
    num = 0
    for i in range(len):
        num = (num << 1) ^ 1
    return f'0x{num:X}'

def generate_field_id(field_id, field_type, field_offset):
    if field_type == "RESERVED":
        return 'reserved_' + str(field_offset)
    return field_id

pattern = r'[^a-zA-Z0-9]'
def generate_field_python_name(field_name, field_type, field_offset):
    if field_type == "RESERVED":
        return 'reserved_' + str(field_offset)
    temp =  re.sub(pattern, '_', field_name).lower()
    if temp[0].isdigit() or temp == 'global':
        temp = '__' + temp
    return temp

# Set up the Jinja2 environment
file_loader = FileSystemLoader(searchpath="./")
env = Environment(loader=file_loader)
env.globals['bits_to_hex'] = bits_to_hex
env.globals['generate_field_id'] = generate_field_id
env.globals['generate_field_python_name'] = generate_field_python_name

module_section_pattern = re.compile(r'^def is_fast_pgn_(\d+)\(\) -> bool:')


def split_rendered_pgns(rendered_output):
    lines = rendered_output.splitlines(keepends=True)
    first_section_index = None
    for index, line in enumerate(lines):
        if module_section_pattern.match(line):
            first_section_index = index
            break

    if first_section_index is None:
        raise RuntimeError('Failed to locate generated PGN function section')

    lookups_source = ''.join(lines[:first_section_index])
    sections = {}
    current_pgn = None
    current_lines = []

    for line in lines[first_section_index:]:
        match = module_section_pattern.match(line)
        if match:
            if current_pgn is not None:
                sections[current_pgn] = ''.join(current_lines)
            current_pgn = int(match.group(1))
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_pgn is not None:
        sections[current_pgn] = ''.join(current_lines)

    return lookups_source, sections


def adapt_lookups_source_for_generated_package(lookups_source):
    replacements = {
        'from .utils import *': 'from ..utils import *',
        'from .message import NMEA2000Message, NMEA2000Field, LookupFieldTypeEnumeration, int_to_bytes': 'from ..message import NMEA2000Message, NMEA2000Field, LookupFieldTypeEnumeration, int_to_bytes',
        'from .consts import PhysicalQuantities, FieldTypes, IndirectLookupEncodeMaps': 'from ..consts import PhysicalQuantities, FieldTypes, IndirectLookupEncodeMaps',
    }
    for old, new in replacements.items():
        lookups_source = lookups_source.replace(old, new)
    return lookups_source.rstrip() + '\n\n__all__ = [name for name in globals() if not name.startswith("__")]\n'


def generated_pgn_module_source(section_source):
    return '\n'.join([
        '# pylint: skip-file',
        'from __future__ import annotations',
        '',
        'from .lookups import *',
        '',
        section_source.rstrip(),
        '',
    ])


def generated_pgns_facade_source():
    return '\n'.join([
        'from __future__ import annotations',
        '',
        'import importlib',
        'import re',
        '',
        "_PGN_ATTR_RE = re.compile(r'^(is_fast_pgn|decode_pgn|encode_pgn)_(\\d+)(?:_|$)')",
        '_MODULE_CACHE = {}',
        '',
        '',
        'def _get_module(name):',
        '    match = _PGN_ATTR_RE.match(name)',
        '    if not match:',
        "        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')",
        '',
        "    module_name = f'.pgn_{match.group(2)}'",
        '    module = _MODULE_CACHE.get(module_name)',
        '    if module is None:',
        '        module = importlib.import_module(module_name, __package__)',
        '        _MODULE_CACHE[module_name] = module',
        '    return module',
        '',
        '',
        'def __getattr__(name):',
        '    module = _get_module(name)',
        '    value = getattr(module, name)',
        '    globals()[name] = value',
        '    return value',
        '',
    ])


def write_generated_pgn_modules(rendered_output):
    package_dir = Path('nmea2000')
    generated_dir = package_dir / 'pgns'
    generated_dir.mkdir(exist_ok=True)

    for existing_module in generated_dir.glob('pgn_*.py'):
        existing_module.unlink()

    lookups_source, sections = split_rendered_pgns(rendered_output)
    (generated_dir / '__init__.py').write_text('"""Generated PGN modules."""\n', encoding='utf-8')
    (generated_dir / 'lookups.py').write_text(adapt_lookups_source_for_generated_package(lookups_source), encoding='utf-8')

    for pgn, section_source in sections.items():
        module_path = generated_dir / f'pgn_{pgn}.py'
        module_path.write_text(generated_pgn_module_source(section_source), encoding='utf-8')

    (generated_dir / '__init__.py').write_text(generated_pgns_facade_source(), encoding='utf-8')

# Load the Jinja2 template
template = env.get_template('python.consts.j2')

# Render the template with the JSON data
output = template.render(data=json_data)

# Save the generated Python code to a file
with open(os.path.join('nmea2000', 'consts.py'), 'w') as f:
    f.write(output)

# Load the Jinja2 template
template = env.get_template('python.PGNs.j2')

# Render the template with the JSON data
output = template.render(data=json_data)

write_generated_pgn_modules(output)

print("Python code generated successfully!")
