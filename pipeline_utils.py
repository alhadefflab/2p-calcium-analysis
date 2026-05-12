from pathlib import Path
import re
import json
import yaml
import os
import inspect
from functools import wraps

from PIL import Image,ImageSequence
import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from collections import defaultdict


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self): 
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

class JSONDictInner(dict):
    def __init__(self, parent_dict, items):
        super().__init__(items) 
        self._parent_dict = parent_dict
    
    def _save_parent(self):
        self._parent_dict._save()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._save_parent()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._save_parent()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._save_parent()

    def clear(self):
        super().clear()
        self._save_parent()

    def __repr__(self):
        return super().__repr__()


def safe_default(obj):
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)  # Fallback: convert to string
    

class JSONDict(defaultdict):
    def __init__(self, filepath):
        super().__init__(lambda: None)  # default to None
        self.filepath = filepath
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.update(json.load(f))
        else:
            self._save()
    
    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self, f, indent=2, default=safe_default)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, dict) and not isinstance(value, JSONDictInner):
            value = JSONDictInner(self, value)
            self[key] = value
        return value
    
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = JSONDictInner(self, value)
        super().__setitem__(key, value)
        self._save()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._save()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._save()

    def clear(self):
        super().clear()
        self._save()



class YAMLDictInner(dict):
    def __init__(self, parent_dict, items):
        super().__init__(items)
        self._parent_dict = parent_dict

    def _save_parent(self):
        self._parent_dict._save()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._save_parent()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._save_parent()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._save_parent()

    def clear(self):
        super().clear()
        self._save_parent()

    def __repr__(self):
        return super().__repr__()


class YAMLDict(defaultdict):
    def __init__(self, filepath):
        super().__init__(lambda: None)  # default to None
        self.filepath = filepath
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    self.update(data)
        else:
            self._save()

    def _save(self):
        def unwrap(obj):
            if isinstance(obj, YAMLDictInner):
                return {k: unwrap(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: unwrap(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, list):
                return [unwrap(o) for o in obj]
            else:
                return obj

        with open(self.filepath, "w") as f:
            yaml.safe_dump(unwrap(dict(self)), f, default_flow_style=False)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, dict) and not isinstance(value, YAMLDictInner):
            value = YAMLDictInner(self, value)
            super().__setitem__(key, value)
        return value

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, YAMLDictInner):
            value = YAMLDictInner(self, value)
        super().__setitem__(key, value)
        self._save()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._save()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._save()

    def clear(self):
        super().clear()
        self._save()


def capture_args(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop('self', None)
        arguments.pop('kwargs', None)
        arguments.pop('provenance', None)
        kwargs['__args_dict'] = arguments
        return func(*args, **kwargs)
    return wrapper

def get_end(x):
    return int(re.findall('.......ome',x.stem)[0][:6])

def get_ch(x):
    return int(re.findall('Ch.',x.stem)[0][2:])

def get_cycle(x):
    return int(re.findall('Cycle.....',x.as_posix())[0][5:])

def combine_tiffs(path, get_z = get_end, get_ch = get_ch, get_frame_cycle = get_cycle, save_dir=None):
    """
    create a single multi-page tiff of all tiffs belonging to the same channel 
    and same z level in a given data directory
    
    path: path to the data directory
    ch: number of the channel
    get_frame_method: function for finding the frame number in the file name
    """
    if save_dir is None:
        save_dir = path

    files = list(path.iterdir())
    files = list(filter(lambda x: x.is_file() & (x.suffix == '.tif'), files))

    zs = np.unique(list(map(get_z, files)))
    channels = np.unique(list(map(get_ch, files)))

    ps = {f'z{z}': {f'ch{c}': None for c in channels} for z in zs}
    for c in channels:
        for z in zs:
            mov_dir = save_dir
            sp = mov_dir/f'{path.name}-z{z}_ch{c}-data.ome.tif'
            ps[f'z{z}'][f'ch{c}'] = sp
            if sp.exists():
                continue

            p = list(filter(lambda x: (get_z(x) == z) & (get_ch(x) == c), files))
            p = sorted(p, key = get_frame_cycle)
            imlist = [Image.open(i) for i in p]
            imlist[0].save(sp.as_posix(), save_all = True, append_images = imlist[1:])
    
    
    return ps

def draw_masks(im,ms,mask,show_plot=True, title='image'):
    """
    function to remove any unwanted neurons
    """
    cv.namedWindow(f'og_{title}')
    cv.imshow(f'og_{title}', im)

    cv.namedWindow(f'{title}')
    down=[False]
    disp_im=im+ms
    color=(65*np.random.rand(1,3)).astype(np.uint8)
    
    def cb(e,x,y,f,z):
        if e==cv.EVENT_LBUTTONDOWN:
            down[0]=True
        if e==cv.EVENT_LBUTTONUP:
            down[0]=False
        if down[0]:
            mask[y,x]=1
            cv.circle(mask,(x,y),1,1,3)
            cv.circle(ms,(x,y),1,tuple(color[0].tolist()),3)
            cv.circle(disp_im,(x,y),1,tuple(color[0].tolist()),3)
            cv.imshow(f'{title}',disp_im)
            
    cv.setMouseCallback(f'{title}',cb)
    while True:
        cv.imshow(f'{title}',disp_im)
        if cv.waitKey(20) & 0xFF == 27:
            break
    
    cv.destroyAllWindows()
    if show_plot:
        _,ax=plt.subplots(1,2)
        ax[0].imshow(mask)
        ax[1].imshow(im)
    return ms,mask.astype(bool)


def remove_neurons(a,im,ms, title='image'):
    """
    function to remove any unwanted neurons
    """
    cv.namedWindow(f'og_{title}')
    cv.imshow(f'og_{title}', im)

    cv.namedWindow(f'{title}')
    neurons=[]
    def cb(e,x,y,f,z):
        if e==cv.EVENT_RBUTTONDOWN:
            if a[x*im.shape[0]+y].sum()==1:
                neuron=np.argmax(a[x*im.shape[0]+y])
                ms[a[:,neuron].reshape(im.shape[:2], order='F')]=0
                neurons.append(neuron)
                cv.imshow(f'{title}',im+ms)

    cv.setMouseCallback(f'{title}',cb)
    while True:
        cv.imshow(f'{title}',im+ms)
        if cv.waitKey(20) & 0xFF == 27:
            break
    
    cv.destroyAllWindows()
    return neurons,ms



