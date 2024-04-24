import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Generating dummy data
epochs = np.arange(1, 21)  # 20 epochs
train_accuracy = np.random.uniform(0.5, 0.95, size=20)  # Training accuracy
test_accuracy = np.random.uniform(0.5, 0.9, size=20)  # Test accuracy

# Generating dummy loss data
train_loss = np.random.uniform(0.1, 0.5, size=20)  # Training loss
test_loss = np.random.uniform(0.2, 0.6, size=20)  # Test loss

# Simulating some improvement (decrease) over time
train_loss = np.sort(train_loss)[::-1]
test_loss = np.sort(test_loss)[::-1]

# Creating a DataFrame for loss data
loss_data = pd.DataFrame(
    {"Epoch": epochs, "Train Loss": train_loss, "Test Loss": test_loss}
)

# Melting the DataFrame to make it suitable for sns.lineplot()
loss_data_melted = pd.melt(
    loss_data,
    id_vars=["Epoch"],
    value_vars=["Train Loss", "Test Loss"],
    var_name="Type",
    value_name="Loss",
)

# Setting the entire background color to light brown, including the figure's background
plt.figure(figsize=(10, 6), dpi=300, facecolor="linen")
sns.set(
    style="darkgrid", rc={"axes.facecolor": "linen", "figure.facecolor": "linen"}
)  # Applying to both axes and figure
sns.lineplot(data=loss_data_melted, x="Epoch", y="Loss", hue="Type", palette="coolwarm")
plt.title("Dummy Train/Test Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.show()

# Resetting the background color for future plots
sns.set(style="white", rc={"axes.facecolor": "white", "figure.facecolor": "white"})


plt.savefig(os.path.join(os.path.dirname(__file__), "dummy_loss.png"))
