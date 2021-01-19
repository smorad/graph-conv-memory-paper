import argparse
import importlib
import multiprocessing
import json
import shutil
import ray
import server.render

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('master_cfg', help='Path to the master .json cfg')
    parser.add_argument('--object-store-mem', help='Size of object store in bytes', default=3e+10)
    parser.add_argument('--local', action='store_true', default=False, help='Run ray in local mode')
    args = parser.parse_args()
    return args

def load_master_cfg(path):
    '''Master cfg should be json. It should be of format
    { 
        ray_cfg: {
            env_config: {
                '...', # path to habitat cfg
            }
            ...
        },
        env_wrapper: "module_here.ClassGoesHere"
        ...
    }
    '''
    with open(path, 'r') as f:
        cfg = json.load(f)
    return cfg
        

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


def main():
    args = get_args()
    cfg = load_master_cfg(args.master_cfg)
    env_class = load_class(cfg, 'env_wrapper') 

    # Rendering obs to website for remote debugging
    shutil.rmtree(server.render.RENDER_ROOT, ignore_errors=True)
    render_server = multiprocessing.Process(
        target=server.render.app.run,
        kwargs={'host': '0.0.0.0', 'debug': True, 'use_reloader': False}
    )
    render_server.start()

    ray.init(dashboard_host='0.0.0.0', local_mode=args.local, 
            object_store_memory=args.object_store_mem)
    trainer_class = load_class(cfg, 'trainer')
    print(f'{trainer_class.__name__}: {env_class.__name__}')
    trainer = trainer_class(env=env_class, config=cfg['ray'])
    while True:
        print(trainer.train())


if __name__ == '__main__':
    main()
