
import numpy as np
import torch
import itertools
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import torch.utils.tensorboard as tf #use pytorch tensorboard; conda install -c conda-forge tensorboard

class Tensorboard:
    def __init__(self, logdir):
        self.writer = None
        self.logdir = logdir

    def close(self):
        self.writer.close()

    def make_dir(self):  # lazy making to avoid empty dir generation.

        if self.writer == None:
            self.writer = tf.SummaryWriter(self.logdir)

    def log_text(self, tag, string, step):
        self.make_dir()
        self.writer.add_text(tag, str(string), step)

    def get_scalar(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return value

    def log_scalar(self, tag, value, global_step):

        self.make_dir()
        value = self.get_scalar(value)
        self.writer.add_scalar(tag, value, global_step)

    def log_confusion_matrix(self, tag, cm, label_names, global_step, normalize=False):

        value_font_size = 140 // len(label_names)

        if normalize:
            cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm, copy=True)
            cm = cm.astype('int')

        np.set_printoptions(precision=2)
        ###fig, ax = matplotlib.figure.Figure()

        fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, cmap='Oranges')

        # classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in label_names]
        # classes = ['\n'.join(wrap(l, 40)) for l in classes]
        classes = label_names

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted', fontsize=25)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=value_font_size, rotation=0, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=25)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=value_font_size, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center",
                    fontsize=value_font_size,
                    verticalalignment='center', color="black")
        fig.set_tight_layout(True)


        # write confusion matrix in tensorboard
        self.make_dir()

        self.writer.add_figure(tag, fig, global_step)


    def log_tsne(self, tag, global_step):


        # write confusion matrix in tensorboard
        self.make_dir()


        # self.writer.add_summary(summary, global_step)

        self.writer.add_figure(tag, plt.gcf(), global_step)


if __name__ == '__main__':
    pass
