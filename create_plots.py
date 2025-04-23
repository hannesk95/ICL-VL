import matplotlib.pyplot as plt

# Your data
shots = [0, 3, 5, 10]
accuracy = [0.617, 0.667, 0.783, 0.9]
lower_ci = [0.5, 0.55, 0.667, 0.817]
upper_ci = [0.733, 0.783, 0.883, 0.967]

# Calculate error bars
error_lower = [a - l for a, l in zip(accuracy, lower_ci)]
error_upper = [u - a for u, a in zip(upper_ci, accuracy)]

# Create the plot
plt.figure(figsize=(8, 5))
plt.errorbar(shots, accuracy, yerr=[error_lower, error_upper], fmt='o-', capsize=5)
plt.xticks(shots)
plt.xlabel('Number of Few-Shot Examples per Class')
plt.ylabel('Classification Accuracy')
plt.title('Binary Classification on CRC100K (Tumor vs Normal)\nUsing GPT-4V with Few-Shot Learning')
plt.grid(True)
plt.ylim(0.4, 1.0)
plt.savefig("few_shot_accuracy.png", dpi=300, bbox_inches='tight')
plt.show()