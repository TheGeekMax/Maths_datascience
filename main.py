import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from scipy.cluster.hierarchy import dendrogram, linkage

digits = load_digits()
np.random.seed(0)
idx = np.random.choice(range(len(digits.images)), 30)
X_image = digits.data[idx]
images = digits.images[idx]


plt.figure(figsize=(6, 3))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(images[i], cmap=plt.cm.bone)
    plt.grid(False)
    plt.xticks(())
    plt.yticks(())
    plt.title(i)
plt.show()


# Compute hierarchical clustering
Z = linkage(X_image, 'ward')

# Dendrogram plotting
plt.figure(figsize=(15, 4))
ax = plt.subplot()

ddata = dendrogram(Z)

dcoord = np.array(ddata["dcoord"])
icoord = np.array(ddata["icoord"])
leaves = np.array(ddata["leaves"])
idx = np.argsort(dcoord[:, 2])
dcoord = dcoord[idx, :]
icoord = icoord[idx, :]
idx = np.argsort(Z[:, :2].ravel())
label_pos = icoord[:, 1:3].ravel()[idx][:30]

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
for i in range(30):
    imagebox = OffsetImage(images[i], cmap=plt.cm.bone_r, interpolation="bilinear", zoom=3)
    ab = AnnotationBbox(imagebox, (label_pos[i], 0), box_alignment=(0.5, -0.1),
                        bboxprops={"edgecolor": "none"})
    ax.add_artist(ab)

plt.show()
