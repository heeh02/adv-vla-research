from pathlib import Path
import sys,types
B=Path('/Users/researcher/Adv-VLA-π0.5/docs/experiments/paper3-training-study-20261019-v1/eval-closeout-v1/comparison-v1/data-driven-increment-v1');P=B/'run-r1-full504-v1/cpu0-package-v1';sys.path.insert(0,str(B));import root_adapter
R,d,q=root_adapter.prepare(P);f=R.bounded_get_function
R.bounded_get_function=types.FunctionType(f.__code__,{**f.__globals__,'HERE':root_adapter.BASE.parent},f.__name__,f.__defaults__,f.__closure__)
sys.argv=[str(P/'root.py'),'readback','--prepared',str(P/'prepared-v1'),'--state',str(P/'run-v1')]
raise SystemExit(R.main())
