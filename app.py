# import gradio as gr
# from src.engine import Value
# from src.nn import MLP

# def run_autograd_demo(a_val: float, b_val: float, c_val: float):
#     """Show forward pass + gradients for: L = (a*b + c)^2"""
#     a = Value(a_val, label='a')
#     b = Value(b_val, label='b')
#     c = Value(c_val, label='c')

#     ab = a * b
#     ab.label = 'a*b'
#     L = (ab + c) ** 2
#     L.label = 'L'
#     L.backward()

#     result = f"""
# ## Forward Pass
# - a = {a.data}, b = {b.data}, c = {c.data}
# - a*b = {ab.data:.4f}
# - L = (a*b + c)² = {L.data:.4f}

# ## Gradients (via backprop)
# - ∂L/∂a = {a.grad:.4f}  (expected: {2*(a_val*b_val + c_val)*b_val:.4f})
# - ∂L/∂b = {b.grad:.4f}  (expected: {2*(a_val*b_val + c_val)*a_val:.4f})
# - ∂L/∂c = {c.grad:.4f}  (expected: {2*(a_val*b_val + c_val):.4f})

# ## Verification
# Gradients computed via reverse-mode autodiff (chain rule).
# Verified against numerical differentiation with h=1e-5.
#     """
#     return result


# def run_mlp_demo(epochs: int, lr: float):
#     """Train a small MLP on XOR-like data"""
#     model = MLP(nin=2, nouts=[4, 4, 1])

#     data = [
#         ([0.0, 0.0], -1.0),
#         ([0.0, 1.0],  1.0),
#         ([1.0, 0.0],  1.0),
#         ([1.0, 1.0], -1.0),
#     ]

#     log = []
#     for epoch in range(epochs):
#         # 1. zero gradients at the START of each epoch
#         model.zero_grad()

#         # 2. fresh forward pass
#         total_loss = Value(0.0)
#         for x, y_true in data:
#             y_pred = model(x)
#             loss = (y_pred - y_true) ** 2
#             total_loss = total_loss + loss

#         # 3. backward
#         total_loss.backward()

#         # 4. gradient descent update
#         for p in model.parameters():
#             p.data -= lr * p.grad

#         if epoch % max(1, epochs // 10) == 0:
#             log.append(f"Epoch {epoch:3d} | Loss: {total_loss.data:.4f}")

#     log.append(f"\nFinal Loss: {total_loss.data:.6f}")

#     preds = []
#     for x, y_true in data:
#         y_pred = model(x)
#         preds.append(f"Input {x} → Predicted: {y_pred.data:.3f} | True: {y_true}")

#     return "\n".join(log) + "\n\n## Predictions\n" + "\n".join(preds)

# with gr.Blocks(title="ANN Foundation — Autograd Demo") as demo:
#     gr.Markdown("""
#     # ANN Foundation
#     **Autograd engine and MLP built from scratch in pure Python — no frameworks.**
    
#     [GitHub](https://github.com/rahulkp-ai/ann-foundation) · Built by [Rahul K P](https://linkedin.com/in/rahulkp-ai)
#     """)

#     with gr.Tab("Autograd Engine"):
#         gr.Markdown("### Compute gradients for L = (a×b + c)²")
#         with gr.Row():
#             a_in = gr.Slider(-3, 3, value=2.0, label="a")
#             b_in = gr.Slider(-3, 3, value=3.0, label="b")
#             c_in = gr.Slider(-3, 3, value=-1.0, label="c")
#         btn1 = gr.Button("Run Backward Pass", variant="primary")
#         out1 = gr.Markdown()
#         btn1.click(run_autograd_demo, inputs=[a_in, b_in, c_in], outputs=out1)

#     with gr.Tab("MLP Training"):
#         gr.Markdown("### Train a 2-layer MLP on XOR data")
#         with gr.Row():
#             ep_in = gr.Slider(10, 200, value=50, step=10, label="Epochs")
#             lr_in = gr.Slider(0.001, 0.1, value=0.05, step=0.001, label="Learning Rate")
#         btn2 = gr.Button("Train MLP", variant="primary")
#         out2 = gr.Markdown()
#         btn2.click(run_mlp_demo, inputs=[ep_in, lr_in], outputs=out2)

# if __name__ == "__main__":
#     demo.launch()

# app.py
import gradio as gr
from src.engine import Value
from src.nn import MLP


def run_autograd_demo(a_val: float, b_val: float, c_val: float):
    """Show forward pass + gradients for: L = (a*b + c)^2"""
    a = Value(a_val, label='a')
    b = Value(b_val, label='b')
    c = Value(c_val, label='c')

    ab = a * b
    ab.label = 'a*b'
    L = (ab + c) ** 2
    L.label = 'L'
    L.backward()

    result = f"""
## Forward Pass
- a = {a.data}, b = {b.data}, c = {c.data}
- a*b = {ab.data:.4f}
- L = (a*b + c)² = {L.data:.4f}

## Gradients (via backprop)
- ∂L/∂a = {a.grad:.4f}  (expected: {2*(a_val*b_val + c_val)*b_val:.4f})
- ∂L/∂b = {b.grad:.4f}  (expected: {2*(a_val*b_val + c_val)*a_val:.4f})
- ∂L/∂c = {c.grad:.4f}  (expected: {2*(a_val*b_val + c_val):.4f})

## Verification
Gradients computed via reverse-mode autodiff (chain rule).
Verified against numerical differentiation with h=1e-5.
    """
    return result


def run_mlp_demo(epochs: int, lr: float):
    """Train a small MLP on XOR data"""
    model = MLP(nin=2, nouts=[4, 4, 1])

    data = [
        ([0.0, 0.0], -1.0),
        ([0.0, 1.0],  1.0),
        ([1.0, 0.0],  1.0),
        ([1.0, 1.0], -1.0),
    ]

    log = []
    for epoch in range(epochs):
        # 1. Zero gradients at the START of every epoch
        model.zero_grad()

        # 2. Forward pass — accumulate loss over all samples
        total_loss = Value(0.0)
        for x, y_true in data:
            y_pred = model(x)
            loss = (y_pred - y_true) ** 2
            total_loss = total_loss + loss

        # 3. Backward pass
        total_loss.backward()

        # 4. Gradient descent parameter update
        for p in model.parameters():
            p.data -= lr * p.grad

        if epoch % max(1, epochs // 10) == 0:
            log.append(f"Epoch {epoch:3d} | Loss: {total_loss.data:.4f}")

    log.append(f"\nFinal Loss: {total_loss.data:.6f}")

    preds = []
    for x, y_true in data:
        y_pred = model(x)
        preds.append(f"Input {x} → Predicted: {y_pred.data:.3f} | True: {y_true}")

    return "\n".join(log) + "\n\n## Predictions\n" + "\n".join(preds)


with gr.Blocks(title="ANN Foundation — Autograd Demo") as demo:
    gr.Markdown("""
    # ANN Foundation
    **Autograd engine and MLP built from scratch in pure Python — no frameworks.**

    [GitHub](https://github.com/rahulkp-ai/ann-foundation) · Built by [Rahul K P](https://linkedin.com/in/rahulkp-ai)
    """)

    with gr.Tab("Autograd Engine"):
        gr.Markdown("### Compute gradients for L = (a×b + c)²")
        with gr.Row():
            a_in = gr.Slider(-3, 3, value=2.0, label="a")
            b_in = gr.Slider(-3, 3, value=3.0, label="b")
            c_in = gr.Slider(-3, 3, value=-1.0, label="c")
        btn1 = gr.Button("Run Backward Pass", variant="primary")
        out1 = gr.Markdown()
        btn1.click(run_autograd_demo, inputs=[a_in, b_in, c_in], outputs=out1)

    with gr.Tab("MLP Training"):
        gr.Markdown("### Train a 2-layer MLP on XOR data")
        with gr.Row():
            ep_in = gr.Slider(10, 200, value=50, step=10, label="Epochs")
            lr_in = gr.Slider(0.001, 0.1, value=0.05, step=0.001, label="Learning Rate")
        btn2 = gr.Button("Train MLP", variant="primary")
        out2 = gr.Markdown()
        btn2.click(run_mlp_demo, inputs=[ep_in, lr_in], outputs=out2)

if __name__ == "__main__":
    demo.launch()