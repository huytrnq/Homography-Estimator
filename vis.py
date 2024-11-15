import matplotlib.pyplot as plt

def show_multiple_images(images, titles, cols=1):
    """
    Display multiple images in one plot.

    Parameters:
        images: list of images to display
        titles: list of titles for each image
    """
    fig, axs = plt.subplots(cols, len(images) // cols, figsize=(20, 20))
    axs = axs.flatten()
    for i, (img, title) in enumerate(zip(images, titles)):
        axs[i].imshow(img)
        axs[i].set_title(title)
        axs[i].axis('off')
    plt.show()