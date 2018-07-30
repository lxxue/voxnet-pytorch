import argparse

parser = argparse.ArgumentParser()
parser.add_argument("rfname")
parser.add_argument("wfname")
args = parser.parse_args()

rfname = args.rfname
wfname = args.wfname

val_acc = []
with open(rfname, "r") as f:
    for line in f:
        if "Val" in line:
            val_acc.append(float(line.split()[2]))

print(val_acc)
with open(wfname, "w") as f:
    for i, acc in enumerate(val_acc):
        f.write("{} {}\n".format(i, acc))

