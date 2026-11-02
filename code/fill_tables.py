"""Inserts the generated table bodies (../results/tables/*.tex) into the manuscript between the markers
%BEGIN:name and %END:name.  usage: python3 fill_tables.py path/to/main.tex"""
import os
import re
import sys

TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tables")

if __name__ == "__main__":
    path = sys.argv[1]
    s = open(path).read()
    for name in re.findall(r"%BEGIN:(\S+)", s):
        f = os.path.join(TAB, "tab_%s.tex" % name)
        if not os.path.exists(f):
            print("missing", f)
            continue
        body = open(f).read().rstrip("\n")
        s = re.sub(r"(%%BEGIN:%s\n)(.*?)(%%END:%s)" % (re.escape(name), re.escape(name)),
                   lambda mm: mm.group(1) + body + "\n" + mm.group(3), s, flags=re.S)
        print("filled", name)
    open(path, "w").write(s)
