import re
import datetime


MONTHS_WITH_31_DAYS = {1, 3, 5, 7, 8, 10, 12}
MONTHS_WITH_30_DAYS = {4, 6, 9, 11}
DATE_FORMAT = "YYYY-MM-DD"
DATETIME_FORMAT = "YYYY-MM-DDTHH:MM:SS"


def is_leap_year(year: str) -> bool:
	"""
	All the years that are perfectly divisible by 4 are called as Leap
	years except the century years. This checks for that.

	Parameters
	----------
	:param year : str
		a four digit string of numbers

	Returns
	-------
	:return : bool
	"""
	assert len(year) == 4, "year must be a string of length 4"
	assert len(re.findall("\d", year)) == 4, "year must be composed of digits"
	return (int(year) % 4 == 0) and year[:-2] != "00"


def get_days_in_month(month: str, year: str = "2018") -> int:
	"""
	returns the number of days in the given month

	Parameters
	----------
	:param month : str
	:param year : str (default: "2018")

	Returns
	-------
	:return int
	"""
	month_int = int(month)
	if month_int in MONTHS_WITH_31_DAYS:
		return 31
	if month_int in MONTHS_WITH_30_DAYS:
		return 30
	if month_int == 2:
		if is_leap_year(year):
			return 29
		return 28
	raise ValueError("Invalid month given")


def extract_date_from_date_time(date_time: str) -> str:
	"""
	Given a date in format YYYY-MM-DDTHH:MM:SS, extract the date
	part (i.e. YYYY-MM-DD)

	Parameters
	----------
	:param date_time : str (DATETIME_FORMAT)

	Returns
	-------
	:return str
		a date in DATE_FORMAT
	"""
	assert type(date_time) == str, "date_time must be of type str. Got %s" % str(type(date_time))
	match = re.findall("\d{4}-\d{2}-\d{2}", date_time)
	if len(match) == 1:
		return match[0]
	raise ValueError("Invalid date_time input given. Got (%s)" % date_time)


def pad_str_left(string, length: int, add: str) -> str:
	"""
	pads the string on the left side by adding the add string to the
	left as many times as necessary such that the output is "[add,]string"

	Parameters
	----------
	:param string : str
		the string to pad with add
	:param length : str
		the minimum length that the string needs to have
	:param add : str
		the string to add in case the output is not long enough

	Returns
	-------
	:return : str
	"""
	out_string = string
	while len(out_string) < length:
		out_string = add + out_string
	return out_string


def get_year_month_day_from_date(date: str) -> tuple:
	"""
	returns the year, month, and day from the date string

	Parameters
	----------
	:param date : str (DATE_FORMAT)

	Returns
	-------
	:return : tuple<str, str, str>
	"""
	match = re.findall("\d{2,4}", date[:len(DATE_FORMAT)])
	if len(match) != 3:
		raise ValueError("Invalid date input given at regex match. Got (%s)" % date)
	year, month, day = match
	if len(year) != 4 or len(month) != 2 or len(day) != 2:
		raise ValueError("Invalid date input given. Got (%s)" % date)
	return year, month, day


def add_days_to_date(date: str, additional_days: int) -> str:
	"""
	adds days to the date. this can also subtract if given a negative
	number

	Parameters
	----------
	:param date : str (DATE_FORMAT)
	:param additional_days : int

	Returns
	-------
	:return : str (DATE_FORMAT)
	"""
	year, month, day = get_year_month_day_from_date(date)
	days_in_month = get_days_in_month(month, year)
	new_day = int(day) + additional_days
	# addition case: add enough days to move to the next month or
	# year, then call it recursively to get the date after adding
	# the remaining days to add
	if new_day > days_in_month:
		remaining_dates = additional_days - (days_in_month - int(day) + 1)
		month = str(int(month) + 1)
		if int(month) > 12:
			year = str(int(year) + 1)
			month = "01"
		return add_days_to_date(
			"%s-%s-%s" % (year, pad_str_left(month, length=2, add="0"), "01"),
			remaining_dates
		)
	# subtraction case: subtract enough days to move back to the
	# previous month or year, then call it recursively to get the
	# date after subtracting the remaining days to subtract
	if new_day < 1:
		remaining_dates = additional_days + int(day)
		month = str(int(month) - 1)
		if int(month) < 1:
			year = str(int(year) - 1)
			month = "12"
		new_day = str(get_days_in_month(month, year))
		return add_days_to_date(
			"%s-%s-%s" % (year, pad_str_left(month, length=2, add="0"), new_day),
			remaining_dates
		)
	# if no anomaly happen with new_day (i.e. a valid day), return
	# the result
	return "%s-%s-%s" % (year, month, pad_str_left(str(new_day), length=2, add="0"))


def is_date_is_before(date: str, target_date: str) -> bool:
	"""
	checks whether the date is before the target date

	Parameters
	----------
	:param date : str (DATE_FORMAT)
	:param target_date : str (DATE_FORMAT)

	Returns
	-------
	:return : bool
	"""
	year, month, day = [int(v) for v in get_year_month_day_from_date(date)]
	t_year, t_month, t_day = [int(v) for v in get_year_month_day_from_date(target_date)]
	return datetime.date(year, month, day) < datetime.date(t_year, t_month, t_day)


def get_dates_from(start_date: str, until_date: str) -> list:
	"""
	Give a list of dates from the start date until the until date

	Parameters
	----------
	:param start_date : str (DATE_FORMAT)
	:param until_date : str (DATE_FORMAT)

	Returns
	-------
	:return : list[str]
		A list of string that represent dates
	"""
	date_start_date = extract_date_from_date_time(start_date)
	assert is_date_is_before(date_start_date, until_date), "start_date must come before until_date"
	dates = [date_start_date]
	current_date = date_start_date
	while True:
		if current_date == until_date:
			break
		dates.append(current_date)
		current_date = add_days_to_date(current_date, 1)
	return dates


def today() -> str:
	"""
	returns today's date

	Returns
	-------
	:return : str (DATE_FORMAT)
	"""
	return str(datetime.datetime.today())[:len(DATE_FORMAT)]
