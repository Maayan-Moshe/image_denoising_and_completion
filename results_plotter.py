def plot_result(results, image_num):
    import matplotlib.pyplot as plt

    fig, ((ax_in, ax_prd, ax_tr)) = plt.subplots(1,3)
    plot_image(ax_in, results['input'][image_num], 'input')
    plot_image(ax_prd, results['predicted z'][image_num], 'predicted')
    plot_image(ax_tr, results['truth'][image_num], 'true image')
    plt.suptitle('image result for image {}'.format(image_num))

def plot_image(ax, image, title):

    ax.imshow(image, 'gray', interpolation = 'none')
    ax.set(title=title)
    ax.axis('off')

if __name__ == '__main__':
    import numpy as np

    results = np.load('results/real_dat_19_May_2018_12_24.npy').tolist()
    plot_result(results, 0)

