from datasets import load_dataset

d = load_dataset("sentence-transformers/simple-wiki", split="train", streaming=True)
for i, x in enumerate(d):
    if i >= 8:
        break
    print("---")
    print("keys:", list(x.keys()))
    for k, v in x.items():
        sv = str(v).replace("\n", " ")
        tail = "..." if len(sv) > 220 else ""
        print(f"  {k}: {sv[:220]}{tail}")
