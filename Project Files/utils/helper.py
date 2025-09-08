import matplotlib.pyplot as plt

def plot_training(history, save_path=None):
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='train_acc')
    ax.plot(history.history['val_accuracy'], label='val_acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig