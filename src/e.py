
# %%
import matplotlib.pyplot as plt
import torch
from engression import engression
from engression.data.simulator import preanm_simulator
import torchviz

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
## Simulate data
x, y = preanm_simulator("square", n=10000, x_lower=0, x_upper=2, noise_std=1, train=True, device=device)
x_eval, y_eval_med, y_eval_mean = preanm_simulator("square", n=1000, x_lower=0, x_upper=4, noise_std=1, train=False, device=device)

# x.shape, y.shape, y_eval_med.shape, y_eval_mean.shape

plt.scatter(x.cpu(), y.cpu(), s=1, alpha=0.5)
plt.scatter(x_eval.cpu(), y_eval_med.cpu(), s=1, alpha=0.5, color="orange")
plt.scatter(x_eval.cpu(), y_eval_mean.cpu(), s=1, alpha=0.5, color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Engression Model")
plt.legend(["Training Data", "Median", "Mean"])
plt.show()

# %%
## Fit an engression model
engressor = engression(x, y, lr=0.01, num_epochs=500, batch_size=1000, device=device)
## Summarize model information
engressor.summary()


# %%
model = engressor.model
_x = torch.randn(1, 1).to(device)
output = model(_x) 
dot = torchviz.make_dot(output, params=dict(model.named_parameters()))
dot.format = "png"
dot.render("engressor_model")

# %%
## Evaluation
print("L2 loss:", engressor.eval_loss(x_eval, y_eval_mean, loss_type="l2"))
print("correlation between predicted and true means:", engressor.eval_loss(x_eval, y_eval_mean, loss_type="cor"))

## Predictions
y_pred_mean = engressor.predict(x_eval, target="mean") ## for the conditional mean
y_pred_med = engressor.predict(x_eval, target="median") ## for the conditional median
y_pred_quant = engressor.predict(x_eval, target=[0.025, 0.5, 0.975]) ## for the conditional 2.5% and 97.5% quantiles