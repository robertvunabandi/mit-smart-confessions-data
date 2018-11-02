from keras.models import Model
import os


STORE_DIR = os.path.dirname(__file__)


def save_model(
	model: Model,
	name: str,
	embedding_size: int,
	epochs: int,
	batch_size: int,
	validation_split: float) -> None:
	"""
	save the keras model to the storage directory.

	:param model: Model
	:param name : str
		-> name of this model to differentiate it from other ones
		with the same set of parameters
	:param embedding_size : int (min: 0, max: 999)
	:param epochs : int (min: 0, max: 9999)
	:param batch_size : int (min: 0, max: 999)
	:param validation_split : float (min: 0.0, max: 1.0)
	"""
	model_suffix = _get_model_suffix(embedding_size, epochs, batch_size, validation_split)
	model.save("%s/%s__%s.h5" % (STORE_DIR, name, model_suffix))


def get_model_title(
	name: str,
	embedding_size: int,
	epochs: int,
	batch_size: int,
	validation_split: float) -> str:
	"""
	Same parameters as save_model. See above.
	"""
	model_suffix = _get_model_suffix(embedding_size, epochs, batch_size, validation_split)
	return "%s/%s__%s.h5" % (STORE_DIR, name, model_suffix)


def _get_model_suffix(
	embedding_size: int,
	epochs: int,
	batch_size: int,
	validation_split: float) -> str:
	"""
	returns a suffix to add to the model name when saving. this
	suffix encodes the model's hyperparameters.

	This has the same parameters as save_model. See above.

	:return : str
	"""
	es = str(embedding_size).rjust(3, "0")
	ep = str(epochs).rjust(4, "0")
	bs = str(batch_size).rjust(3, "0")
	vs = str(round(validation_split, 3)).ljust(5, "0")
	return "%s_%s_%s_%s" % (es, ep, bs, vs)
