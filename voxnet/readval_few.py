import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("wfname")

args = parser.parse_args()

all_acc = []
for i in range(1,11):
    each_acc = []
    with open(args.dir+"{}/log.txt".format(i), 'r') as f:
        for line in f:
            if "Val" in line:
                each_acc.append(float(line.split()[2]))
    all_acc.append(each_acc)

all_acc = np.array(all_acc)
avg_acc = np.mean(all_acc, axis=0)
print(avg_acc)

with open(args.wfname,"w") as f:
    for i, acc in enumerate(avg_acc):
        f.write("{} {}\n".format(i, acc))


