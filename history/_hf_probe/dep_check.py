import importlib
for m in ["numpy", "torch", "transformers", "safetensors"]:
    try:
        mod = importlib.import_module(m)
        print("  %-14s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("  %-14s MISSING (%s)" % (m, type(e).__name__))
