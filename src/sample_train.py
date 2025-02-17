import random

file_names = [
    "data/beauty/train.txt",
    "data/sports/train.txt",
    "data/toys/train.txt",
    "data/yelp/train.txt",
]

file_contents = []
for name in file_names:
    with open(name, encoding="utf-8") as f:
        file_contents.append(f.readlines())

max_lines = max(len(content) for content in file_contents)

for name, lines in zip(file_names, file_contents):
    needed = max_lines - len(lines)
    sampled = []
    if needed > 0 and len(lines) > 1:
        pool = lines[1:]
        sampled = random.sample(pool, min(needed, len(pool)))
    with open(name.replace(".txt", "_sampled.txt"), "w", encoding="utf-8") as f:
        f.writelines(lines + sampled)
