"""Select the real frozen namespace, rejecting installed regular-package shadows."""
import hashlib
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

def need(ok,message):
    if not ok:raise ValueError(message)

def origins(root):
    root=Path(root).resolve();records={}
    for name,module in tuple(sys.modules.items()):
        if name!='libero' and not name.startswith('libero.'):continue
        for entry in getattr(module,'__path__',[]):need(Path(entry).resolve().is_relative_to(root),'foreign LIBERO namespace component')
        source=getattr(module,'__file__',None)
        if source:
            path=Path(source).resolve();need(path.is_relative_to(root),'foreign installed LIBERO module')
            raw=path.read_bytes();records[name]={'relative':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    return records

def install(root):
    root=Path(root).resolve(strict=True);origins(root)
    if 'libero' not in sys.modules:
        spec=importlib.machinery.PathFinder.find_spec('libero',[str(root)])
        need(spec is not None and spec.submodule_search_locations is not None,'frozen official outer package absent')
        need(all(Path(p).resolve().is_relative_to(root) for p in spec.submodule_search_locations),'outer namespace resolves elsewhere')
        module=importlib.util.module_from_spec(spec);sys.modules['libero']=module
        if spec.loader is not None:spec.loader.exec_module(module)
    sys.path.insert(0,str(root));importlib.invalidate_caches()
    return origins(root)
