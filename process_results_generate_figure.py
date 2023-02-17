import matplotlib.pyplot as plt
import numpy as np


def img_is_color(img):
    # from https://stackoverflow.com/a/67992521/7947996
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10),
                    title_fontsize=30):
    # from https://stackoverflow.com/a/67992521/7947996
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):
        img = list_images[i]
        title = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)
        list_axes[i].axis('off')

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    # _ = plt.show()
    line = plt.Line2D((.5, .5), (.1, .9), color="k", linewidth=3)
    fig.add_artist(line)


#################################


artifacts_directory = "."

datasets_algorithms_seeds = [
    ["circles", "em_hard", "2"],
    ["moons", "em_hard", "2"],
    ["moons", "em_soft", "2"],
    ["moons", "gd", "2"],
    ["moons", "gd_variational", "4"],
    ["pinwheel", "em_soft", "1"],
    ["pinwheel", "gd_variational", "4"],
    ["two_banana", "em_hard", "2"]
]

synonyms = {
    "dataset": {
        "circles": "Circles",
        "moons": "Moons",
        "pinwheel": "Pinwheel",
        "two_banana": "Two bananas"
    },
    "algorithm": {
        "gd": "GD",
        "em_soft": "SoftEM",
        "em_hard": "HardEM",
        "gd_variational": "VarGD"
    }
}

plots = []
titles = []
for dataset, algorithm, seed in datasets_algorithms_seeds:
    boundary_plot_path = f"{artifacts_directory}/artifacts-seed{seed}/{dataset}/{algorithm}/evaluate/plot_decision_boundary_2d.png"
    density_plot_path = f"{artifacts_directory}/artifacts-seed{seed}/{dataset}/{algorithm}/evaluate/plot_density_2d.png"

    plots.append(plt.imread(boundary_plot_path))
    plots.append(plt.imread(density_plot_path))

    title = f"{synonyms['dataset'][dataset]}.{synonyms['algorithm'][algorithm]}.{seed}"
    titles.append(title)
    titles.append(title)

show_image_list(list_images=plots,
                list_titles=titles,
                num_cols=4,
                figsize=(10, 10),
                grid=False,
                title_fontsize=15)

plt.savefig("process_results/figure.png", dpi=400)
