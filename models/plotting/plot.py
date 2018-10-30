import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model


def plot_classification_history(history) -> None:
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.plot(
		history.epoch, np.array(history.history['acc']), label='Training Accuracy'
	)
	plt.plot(
		history.epoch, np.array(history.history['val_acc']), label='Validation Accuracy'
	)
	plt.legend()
	plt.ylim([0, 1.5])
	plt.show()


def plot_regression_history(history) -> None:
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Absolute Error')
	plt.plot(
		history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss'
	)
	plt.plot(
		history.epoch, np.array(history.history['val_mean_absolute_error']), label='Validation Loss'
	)
	plt.legend()
	plt.ylim([0, 1.5])
	plt.show()


def plot_prediction(model: Model, data_: np.ndarray, labels: np.ndarray, tag: str) -> None:
	predictions = model.predict(data_).flatten()
	plt.scatter(labels, predictions)
	plt.xlabel('True Values (%s)' % tag)
	plt.ylabel('Predictions (%s)' % tag)
	plt.axis('equal')
	plt.xlim(plt.xlim())
	plt.ylim(plt.ylim())
	plt.plot([-5, 20], [-5, 20])
	plt.show()
