import pickle
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import numpy as np

import argparse

parser = argparse.ArgumentParser(description='model type based on the inputs')
parser.add_argument("inputs", type=str)
parser.add_argument("--acc", action="store_true")

args = parser.parse_args()
model_name = args.inputs
plot_acc = args.acc

with open('saved_models/model_loss_acc/loss_acc_{0}'.format(model_name), 'rb') as file_pi:
    history = pickle.load(file_pi)
    print(history)


plt.gcf().clear()
early_stop = np.argmin(history['val_loss'])
if plot_acc:
    plt.plot(history['acc'], label='train')
    plt.plot(history['val_acc'], label='valid')
    #plt.hlines([history['val_acc'][early_stop]], 0, early_stop, color="#999999")
    plt.scatter([early_stop], [history['val_acc'][early_stop]])
    plt.annotate("Early stoping\nAccuracy={0:0.4f}".format(history['val_acc'][early_stop]), (early_stop, history['val_acc'][early_stop]), textcoords='offset points', xytext=(-0.3, 1.2))
    plt.ylabel('accuracy')
    #yticks = plt.yticks()[0]
    #plt.yticks([t for t in yticks if t < history['val_acc'][early_stop]] + [history['val_acc'][early_stop]] + [t for t in yticks if t > history['val_acc'][early_stop]])
else:
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='valid')
    #plt.hlines([history['val_loss'][early_stop]], 0, early_stop, color="#999999")
    plt.scatter([early_stop], [history['val_loss'][early_stop]])
    plt.annotate("Early stoping\nLoss={0:0.4f}".format(history['val_loss'][early_stop]), (early_stop, history['val_loss'][early_stop]), textcoords='offset points', xytext=(-0.3, 1.2))
    plt.ylabel('loss')
    #yticks = plt.yticks()[0]
    #plt.yticks([t for t in yticks if t < history['val_loss'][early_stop]] + [history['val_loss'][early_stop]] + [t for t in yticks if t > history['val_loss'][early_stop]])
plt.xlabel('epochs')
plt.legend()
plt.title('model {0}'.format(model_name))
if plot_acc:
    plt.savefig('plot_{0}_acc.pdf'.format(model_name), bbox_inches='tight')
else:
    plt.savefig('plot_{0}_loss.pdf'.format(model_name), bbox_inches='tight')


