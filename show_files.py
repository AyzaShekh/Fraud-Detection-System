from pathlib import Path

folders = ["data", "models", "notebooks", "src"]

for folder in folders:
    path = Path(folder)

    print(f"\n{'=' * 35}")
    print(f"FOLDER: {folder}")
    print("=" * 35)

    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                print(item)
    else:
        print("Folder not found!")
