import csv


def save(path, str_):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        for s in str_:
            wr.writerow(s)


def load(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        rd = csv.reader(f)
        data = []
        for s in rd:
            data.append([float(v) for v in s])
    return data
