#!/bin/bash
set -e   # stop on any failure

echo "=========================================="
echo "  Pre-push verification"
echo "=========================================="

echo ""
echo "1. Running test suite..."
pytest tests/ --cov=src --cov-report=term-missing -q
echo "   Tests: PASS"

echo ""
echo "2. Verifying autograd engine gradients..."
python -c "
from src.engine import Value
a = Value(2.0); b = Value(3.0); c = Value(-1.0)
L = (a * b + c) ** 2
L.backward()
h = 1e-5
def f(av): a_=Value(av); b_=Value(3.0); c_=Value(-1.0); return ((a_*b_+c_)**2).data
num = (f(2.0+h) - f(2.0-h)) / (2*h)
assert abs(a.grad - num) < 1e-4, f'Gradient wrong: {a.grad} vs {num}'
print('   Gradients: PASS')
"

echo ""
echo "3. Verifying MLP learns XOR..."
python -c "
from src.engine import Value
from src.nn import MLP
model = MLP(nin=2, nouts=[4, 4, 1])
data = [([0.,0.],-1.),([0.,1.],1.),([1.,0.],1.),([1.,1.],-1.)]
for _ in range(200):
    model.zero_grad()
    loss = Value(0.0)
    for x,y in data:
        loss = loss + (model(x) - y)**2
    loss.backward()
    for p in model.parameters(): p.data -= 0.05 * p.grad
assert loss.data < 0.5, f'MLP did not learn: final loss = {loss.data:.4f}'
print(f'   MLP training: PASS  (final loss: {loss.data:.4f})')
"

echo ""
echo "4. Verifying app.py functions..."
python -c "
from app import run_autograd_demo, run_mlp_demo
r1 = run_autograd_demo(2.0, 3.0, -1.0)
assert '10.0000' in r1, 'Autograd demo failed'
print('   Autograd demo: PASS')
r2 = run_mlp_demo(200, 0.05)
final = float([l for l in r2.split('\n') if 'Final Loss' in l][0].split(': ')[1])
assert final < 0.5, f'MLP demo failed: loss={final}'
print(f'   MLP demo: PASS  (final loss: {final:.4f})')
"

echo ""
echo "=========================================="
echo "  All checks passed. Safe to push."
echo "=========================================="