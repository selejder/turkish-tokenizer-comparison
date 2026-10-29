"""Inspect and minimally clean the OPUS Bianet Turkish corpus."""

import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Bianet.en-tr.tr"
INTERIM_DATA_PATH = (
    PROJECT_ROOT / "data" / "interim" / "Bianet_tr_minimal-cleaned.txt"
)

if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Raw corpus not found at {RAW_DATA_PATH}. See README.md for download steps."
    )

data = RAW_DATA_PATH.read_text(encoding="utf-8")


lines = data.splitlines()


# data information block
print(f"Character length of data is {len(data)}")
print(f"Number of lines in data is {len(lines)}")


empty_lines_index = [
    index
    for index, line in enumerate(lines, start = 1)
    if not line.strip()
]


lines_len = [len(line) for line in lines]
average_len = sum(lines_len) / len(lines_len)
max_len = max(lines_len)
min_len = min(lines_len)
print(f"Average line length is {average_len}")
print(f"Maximum and minimum line lengths are {max_len}, {min_len}")

set_of_data = set(data)
alphabets = []

for char in set_of_data:
    alphabet = (unicodedata.name(char, 'UNKNOWN').split())[:1]
    alphabets.append(alphabet)


alphabets = ["CYRILLIC", 'GREEK', 'ARMENIAN', "ARABIC", "SYRIAC", "GEORGIAN"]
alphabets_counter = {alpha: 0 for alpha in alphabets}

for line_number, line in enumerate(lines, start = 1):

    print_line = False

    for char in line:
        char_name = unicodedata.name(char, '')
        if not char_name:
            continue

        alpha = char_name.split(" ", 1)[0]

        if alpha in alphabets:
            print_line = True
            alphabets_counter[alpha] += 1

    if print_line:
        print(f"{line_number}: {line}\n\n")



not_wanted_chars = {'\xa0': 0, '\xad': 0, '\u200a': 0, '\u200b': 0, '\u200e': 0, '\u200f': 0}

for char in not_wanted_chars:
    examples = []

    for line_number, line in enumerate(lines, start=1):
        if char in line:
            examples.append((line_number, char, line))

        if len(examples) == 3:
            break

    for element in examples:
        print(repr(element) + '\n')


char_map = {
    '\xa0': ' ',
    '\u200a' : ' ',
    '\xad' : '',
    '\u200b' : '',
    '\u200e' : '',
    '\u200f' : ''
}


cleaned_data = data

for ch1, ch2 in char_map.items():
    cleaned_data = cleaned_data.replace(ch1, ch2)


data_len = len(data)
cleaned_data_len = len(cleaned_data)
len_dif = data_len - cleaned_data_len

number_line = len(data.splitlines())
number_line_cd = len(cleaned_data.splitlines())
line_dif = number_line - number_line_cd

print(f"Character count difference after cleaning is {len_dif}")
print(f"Line count difference after cleaning is {line_dif}")



print(data_len)
print(cleaned_data_len)
print(number_line)
print(number_line_cd)


INTERIM_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
INTERIM_DATA_PATH.write_text(cleaned_data, encoding='utf-8', newline='\n')

with open(INTERIM_DATA_PATH, mode='r', encoding='utf-8') as file:
    new_data = file.read()


#len(new_data), cleaned_data == new_data


old_vocab = sorted(list(set(data)))
new_vocab = sorted(list(set(new_data)))

print(f"Raw vocabulary size is {len(old_vocab)}")
print(f"Cleaned vocabulary size is {len(new_vocab)}")
