#!/usr/bin/env python
"""Give the 9 runs from Sep 28 the names they were launched with.

finetune.sh ignored TAG, so the 160-epoch runs landed in the 40-epoch folders.
Each move happens only if the run folder proves it is the right run: 160 epochs,
all completed, expected seed and starting checkpoint, target name still free.

  python rename_ep160.py        # dry run: shows what it would do
  python rename_ep160.py --do   # actually moves
"""
import glob, json, os, sys

BASE = "codebrim-classif-balanced_ResNet50_hta"
MOVES = ([(f"ft_strong_r065_seed{s}", f"ft_strong_r065_ep160_seed{s}", "strong_r065.pth", s) for s in range(3)]
         + [(f"ft_strong_r08_seed{s}", f"ft_strong_r08_ep160_seed{s}", "strong_r08.pth", s) for s in range(3)]
         + [(f"ft_{BASE}_seed{s}", f"ft_control_ep160_seed{s}", f"{BASE}.pth", s) for s in range(3)])


def main(do):
    ok_all = True
    for old, new, start, seed in MOVES:
        h = os.path.join("runs", old, "history.json")
        if not os.path.exists(h):
            print(f"  SKIP {old}: no run folder");  ok_all = False;  continue
        H = json.load(open(h))
        a = H["args"]
        checks = {"160 epochs": int(a["epochs"]) == 160,
                  "all done": len(H["history"]) == int(a["epochs"]),
                  f"seed {seed}": int(a.get("seed", -1)) == seed,
                  f"from {start}": os.path.basename(a["ckpt"]) == start,
                  "target free": not os.path.exists(os.path.join("runs", new))}
        bad = [k for k, v in checks.items() if not v]
        if bad:
            print(f"  SKIP {old}: fails {', '.join(bad)}");  ok_all = False;  continue
        files = sorted(glob.glob(os.path.join("results", f"{old}_test*")))
        print(f"  {'MOVE' if do else 'would move'} runs/{old} -> runs/{new}  (+ {len(files)} results files)")
        if do:
            os.rename(os.path.join("runs", old), os.path.join("runs", new))
            for f in files:
                os.rename(f, f.replace(f"results/{old}_test", f"results/{new}_test"))
    print("\nall 9 verified" + ("" if do else ", run again with --do") if ok_all else
          "\nsome runs were skipped: paste this output before going further")


if __name__ == "__main__":
    main("--do" in sys.argv)
