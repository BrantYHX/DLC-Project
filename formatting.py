from PIL import Image
import matplotlib.pyplot as plt

# Load images
img_A = Image.open(r"C:\Users\HAOXUAN YIN\Desktop\Dissertation Figure\Figure 4__A.png")
img_B = Image.open(r"C:\Users\HAOXUAN YIN\Desktop\Dissertation Figure\Figure 4_B.png")
img_C = Image.open(r"C:\Users\HAOXUAN YIN\Desktop\Dissertation Figure\Figure 4__C.png")
img_D = Image.open(r"C:\Users\HAOXUAN YIN\Desktop\Dissertation Figure\Figure 4_D.png")

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Flatten axs for easy iteration
axs = axs.flatten()

# List of images
images = [img_A, img_B, img_C, img_D]
labels = ["A", "B", "C", "D"]

# Display each image in the corresponding subplot
for ax, img, label in zip(axs, images, labels):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(label, loc='left', fontsize=18, fontweight='bold', pad=10)


# Adjust layout and save
plt.tight_layout()
plt.savefig("Figure4_combined.png", dpi=300, bbox_inches='tight')
plt.show()