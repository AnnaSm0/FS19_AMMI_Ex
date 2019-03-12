...
# load tensors
input_data = th.load("path-to-inputs.pt")
labels = th.load("path-to-labels.pt")

# specify model
class LogisticRegressionModel(th.nn.Module):
	def __init__(self, D_in, D_out):
		...

	def forward(self, x):
		...

# init model
model = ...

# init loss
loss_fn = ...

# setup optimizer
optimizer = ...

for t in range(2000):
	def closure():
		# evaluate model
		predicted = ...

		# evaluate loss
		loss = ...

		# print loss
		print(str(t) + ":\t" + str(loss.data.item()))

		# calculate gradients
		optimizer.zero_grad()
		loss.backward()

		return loss


	optimizer.step(closure)
...
