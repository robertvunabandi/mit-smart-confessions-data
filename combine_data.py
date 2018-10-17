import json
import utils


def combine_data_from(data_name: str, min_date: str) -> None:
	date = min_date
	today = utils.today()
	output_data = {}
	while utils.is_date_is_before(date, today):
		try:
			with open("data/%s_%s.json" % (data_name, date), "r") as json_file:
				output_data[date] = json.load(json_file)
		except FileNotFoundError:
			pass
		date = utils.add_days_to_date(date, 1)
	with open("data/_%s_data_confessions.json" % data_name, "w") as output_json:
		json.dump(output_data, output_json)


def combine_data_to_array_from(data_name: str, min_date: str) -> None:
	date = min_date
	today = utils.today()
	output_data = []
	while utils.is_date_is_before(date, today):
		try:
			with open("data/%s_%s.json" % (data_name, date), "r") as json_file:
				output_data += json.load(json_file)
		except FileNotFoundError:
			pass
		date = utils.add_days_to_date(date, 1)
	with open("data/_%s_data_confessions_array.json" % data_name, "w") as output_json:
		json.dump(output_data, output_json)
	print(len(output_data))


if __name__ == '__main__':
	_min_date = "2016-10-15"
	combine_data_from("mitafricans", _min_date)
	combine_data_to_array_from("mitafricans", _min_date)
