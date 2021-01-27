import importlib


def load_class(cfg, key):
    try:
        module = importlib.import_module(cfg[key]["module"])
    except KeyError:
        print(f'Did not find key [{key}]["module"] in cfg, ensure it is set')
        raise
    except Exception:
        print(f'Failed to load module {cfg[key]}, ensure'
                ' module is set correctly.' )
        raise

    try:
        cls = getattr(module, cfg[key]["class"])
    except KeyError:
        print(f'Did not find key [{key}]["class"] in cfg, ensure it is set')
        raise
    except Exception:
        print(f'Failed to load class {cfg[key]}, ensure class '
                'is set correctly.')
        raise

    return cls
