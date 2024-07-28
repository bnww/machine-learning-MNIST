import matplotlib.pyplot as plt

def plot_filters(filters, cols, rows):
    fig, axes = plt.subplots(rows, cols, figsize = (cols,rows),
                            subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for i, ax in enumerate(axes.flat):
        ax.imshow(filters[:,:,i], cmap='gray')
    plt.savefig('Layer weights')

def plot_feature_maps(feature_map):
    cols = 8
    rows = int(feature_map.shape[-1]/cols)
    fig=plt.figure(figsize=(15,15))
    for i in range(1, rows*cols +1):
        fig = plt.subplot(rows, cols, i)
        fig.set_xticks([])
        fig.set_yticks([])
        plt.imshow(feature_map[0, :, :, i-1], cmap='gray')
        plt.savefig('feature map')
