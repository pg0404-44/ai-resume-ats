import importlib.metadata as m

for d in m.distributions():
    name = d.metadata.get("Name")
    path = d._path

    try:
        entries = d.entry_points
        print("OK  ", name)
    except Exception as e:
        print("\n==============================")
        print("BAD PACKAGE FOUND")
        print("Name :", name)
        print("Path :", path)
        print("Error:", repr(e))
        print("==============================")