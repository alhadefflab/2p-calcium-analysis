import ast, sys
files = [
    'analysis/manifold.py',
    'visualization/manifold_plots.py',
    'gui/manifold_viewer.py',
    'gui/app.py',
]
ok = True
for f in files:
    try:
        with open(f, encoding='utf-8') as fh:
            ast.parse(fh.read())
        print('OK ', f)
    except SyntaxError as e:
        print('ERR', f, e)
        ok = False
sys.exit(0 if ok else 1)
