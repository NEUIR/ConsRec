import random

random.seed(42)

file_paths = [
    "data/beauty/valid.txt",
    "data/sports/valid.txt",
    "data/toys/valid.txt",
    "data/yelp/valid.txt",
]

headers, contents = [], []
for p in file_paths:
    with open(p, encoding="utf-8") as f:
        lines = f.readlines()
        headers.append(lines[0])
        contents.append(lines[1:])

k = min(len(c) for c in contents)

for path, h, c in zip(file_paths, headers, contents):
    with open(path.replace(".txt", "_sampled.txt"), "w", encoding="utf-8") as f:
        f.write(h)
        f.writelines(random.sample(c, k))
