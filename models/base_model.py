from keras import Model

class BaseModel:
	def create_model(self) -> Model:
		pass

	def train_model(self) -> Model:
		pass

	def run_model(self):
		pass



