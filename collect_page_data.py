import facebook
import config
import utils
import json


FB_REACTIONS = ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]


def collect_page_data(
	name: str,
	page_id: str,
	access_token: str,
	start_date: str,
	verbose: bool = True) -> None:
	"""
	This will collect the data from the given page for each day
	starting from the start date until the end. This will save the
	data for each day start at start date inside of the data folder
	in the format "name_YYYY-MM-DD.json"

	This is the main method in this file, and it will collect the data
	for the page into the data folder. This method is a bit too long,
	but it's better to leave all of it in one place than break it down
	(since breaking it down requires passing in a lot of parameters)

	Parameters
	----------
	:param name : str
	:param page_id : str
		the facebook object id of the page we're getting data from.
		https://findmyfbid.com can be used to figure out what the
		page id is
	:param access_token : str
		the access token that matches the facebook page id above. this
		must match otherwise none of the operations here would work
	:param start_date : str
		A string of the form YYYY-MM-DD
	:param verbose : bool
		A flag to print out a bunch of status messages
	"""
	fb_graph = facebook.GraphAPI(access_token=access_token, version="3.0")
	# first, collect the page fans from start_date until later
	date_to_page_fans, valid_start_date = get_date_to_page_fans(fb_graph, page_id, start_date)
	date_to_page_daily_engaged_users = get_date_to_daily_engaged_users(fb_graph, page_id, start_date)
	date_to_page_views_total = get_date_to_page_views_total(fb_graph, page_id, start_date)
	date_to_page_consumptions = get_date_to_page_consumptions(fb_graph, page_id, start_date)
	dates = utils.get_dates_from(
		# this will ensure the first start date is labelled with the
		# right number of page views and such
		valid_start_date,
		# subtract 1 just to make sure we have data in
		# date_to_page_fans for this day. facebook's data doesn't
		# update on time sometimes, so we may just not have access to
		# today's data yet
		until_date=utils.add_days_to_date(utils.today(), -1),
	)

	# store page user data
	page_user_data = [{
		"page_fans": date_to_page_fans.get(date, -1),
		"page_engaged_users": date_to_page_daily_engaged_users.get(date, -1),
		"page_views": date_to_page_views_total.get(date, -1),
		"page_consumption": date_to_page_consumptions.get(date, -1),
		"date": date,
	} for date in dates if date_to_page_fans.get(date, -1) != -1]
	store_page_user_data(name, page_user_data)

	# for each date, collect the posts and populate it each
	date_to_posts = collect_date_to_page_post(fb_graph, page_id, dates)

	# for each post that we have, we want to collect reactions
	# and comments counts and add the details about the page
	date_to_labelled_posts = {}
	for date, posts in date_to_posts.items():
		if verbose:
			print("labelling posts from %s" % date)
		try:
			page_data = {
				"page_fans": date_to_page_fans[date],
				"page_engaged_users": date_to_page_daily_engaged_users[date],
				"page_views": date_to_page_views_total[date],
				"page_consumption": date_to_page_consumptions[date],
			}
		except KeyError as e:
			print("skipping %s because of KeyError" % date)
			print(e)
			continue
		labelled_posts = []
		for post in posts:
			post_id = post["post_id"]
			try:
				post_data = get_post_data(fb_graph, post_id)
			except facebook.GraphAPIError as e:
				print(e)
				print(
					"skipping post with post_id %s for date %s to "
					"facebook.GraphAPIError error" % (post_id, date)
				)
				continue
			except Exception as e:
				print(e)
				print(
					"skipping post with post_id %s for date %s to "
					"unexpected error" % (post_id, date)
				)
				continue
			labelled_post = post.copy()  # make a copy
			for key, value in post_data.items():
				labelled_post[key] = value
			labelled_post["page_data"] = page_data
			labelled_posts.append(labelled_post)
		date_to_labelled_posts[date] = labelled_posts
	print("done with collecting and labelling data")

	# save this data in the data folder
	for date, posts in date_to_labelled_posts.items():
		store_data_for_date(name, date, posts)

	# save the comments of each post, do this separately because this
	# will take much longer
	for date, posts in date_to_labelled_posts.items():
		if verbose:
			print("collecting comments and their reactions for date %s" % date)
		posts_with_comments_for_date = []
		for post in posts:
			post_comments_object = get_comments_and_sub_comments_in_fb_object(
				fb_graph,
				post["post_id"],
				verbose=verbose,
			)
			for field in ["reactions", "comments"]:
				if field in post_comments_object:
					post[field] = post_comments_object[field]
			posts_with_comments_for_date.append(post)
		store_data_with_comments_for_date(name, date, posts_with_comments_for_date)
	print("completed! data saved")


def get_date_to_page_fans(
	graph: facebook.GraphAPI,
	page_id: str,
	start_date: str) -> tuple:
	"""
	gets the number of page fans (page likes) on facebook for that page

	Parameters
	----------
	:param graph : facebook.GraphAPI
	:param page_id : str
	:param start_date : str (DATE_FORMAT)

	Returns
	-------
	:return tuple<dict<str, int>, str>
		A tuple with two things:
		- A dictionary mapping date (in format YYYY-MM-DD) to integers
		representing the number of page fans at that date
		- A string representing the earlier valid date that we have
		(this is the date that has at least 1 fan, which may or may
		not be start_date)
	"""
	api_data = graph.get_connections(
		id=page_id,
		connection_name="insights/page_fans",
		# reduce one day so that we're able to get the number
		# of followers at the end of that day as well
		since=utils.add_days_to_date(start_date, -1),
	)["data"][0]["values"]
	date_to_page_fans, valid_start_date = {}, None
	for obj in api_data:
		value, date = obj.get("value", None), obj.get("end_time", None)
		if value is None or date is None:
			continue
		date_to_page_fans[utils.extract_date_from_date_time(date)] = value
		if valid_start_date is None and value > 0:
			valid_start_date = date
	return date_to_page_fans, valid_start_date


def get_date_to_daily_engaged_users(
	graph: facebook.GraphAPI,
	page_id: str,
	start_date: str) -> dict:
	"""
	returns a dictionary mapping dates to daily engaged users for this page

	Parameters
	----------
	:param graph : facebook.GraphAPI
	:param page_id : str
	:param start_date : str (DATE_FORMAT)

	Returns
	-------
	:return dict<str, int>
		a dictionary mapping dates to engaged users
	"""
	api_data = graph.get_connections(
		id=page_id,
		connection_name="insights/page_engaged_users",
		# reduce one day so that we're able to get the number
		# of followers at the end of that day as well
		since=utils.add_days_to_date(start_date, -1),
		period="day",
	)["data"][0]["values"]
	date_to_daily_engaged_users = {}
	for obj in api_data:
		value, date = obj["value"], utils.extract_date_from_date_time(obj["end_time"])
		date_to_daily_engaged_users[date] = value
	return date_to_daily_engaged_users


def get_date_to_page_views_total(
	graph: facebook.GraphAPI,
	page_id: str,
	start_date: str) -> dict:
	"""
	returns a dictionary mapping dates to daily page views total
	that is, the number of times a Page's profile has been viewed
	by logged in and logged out people

	Parameters
	----------
	:param graph : facebook.GraphAPI
	:param page_id : str
	:param start_date : str (DATE_FORMAT)

	Returns
	-------
	:return dict<str, int>
		a dictionary mapping dates to daily page views
	"""
	api_data = graph.get_connections(
		id=page_id,
		connection_name="insights/page_views_total",
		# reduce one day so that we're able to get the page views
		# total for the start date as well
		since=utils.add_days_to_date(start_date, -1),
		period="day",
	)["data"][0]["values"]
	date_to_page_views_total = {}
	for obj in api_data:
		value, date = obj["value"], utils.extract_date_from_date_time(obj["end_time"])
		date_to_page_views_total[date] = value
	return date_to_page_views_total


def get_date_to_page_consumptions(
	graph: facebook.GraphAPI,
	page_id: str,
	start_date: str) -> dict:
	"""
	returns a dictionary mapping dates to daily page consumptions
	that is, the number of times people clicked on any of your content

	Parameters
	----------
	:param graph : facebook.GraphAPI
	:param page_id : str
	:param start_date : str (DATE_FORMAT)

	Returns
	-------
	:return dict<str, int>
		a dictionary mapping dates to page consumptions
	"""
	api_data = graph.get_connections(
		id=page_id,
		connection_name="insights/page_consumptions",
		# reduce one day so that we're able to get the page
		# consumption for the start date as well
		since=utils.add_days_to_date(start_date, -1),
		period="day",
	)["data"][0]["values"]
	date_to_daily_page_consumptions = {}
	for obj in api_data:
		value, date = obj["value"], utils.extract_date_from_date_time(obj["end_time"])
		date_to_daily_page_consumptions[date] = value
	return date_to_daily_page_consumptions


def store_page_user_data(page_name: str, page_user_data: list) -> None:
	"""
	Stores the pages's user data into for the page name given in
	parameter with the page data given in the parameters as well

	Parameters
	----------
	:param page_name : str
	:param page_user_data : list[dict<str, int>]
	"""
	with open("data/_page_user_data__%s.json" % page_name, "w") as data_file:
		json.dump(page_user_data, data_file)


def collect_date_to_page_post(
	graph: facebook.GraphAPI,
	page_id: str,
	dates: list,
	verbose: bool=True) -> dict:
	"""
	TODO - do docs

	Parameters
	----------
	:param graph : facebook.GraphAPI
	:param page_id : str
	:param dates : list[str (DATE_FORMAT)]
	:param verbose : bool

	Returns
	-------
	:return : dict<key, PostObj>
	"""
	date_to_posts = {}
	if verbose:
		print("collecting posts from %s until %s" % (dates[0], dates[-1]))
	error_count = 0
	for date in dates:
		if verbose:
			print("collecting posts for %s" % date)
		try:
			posts = get_page_post(graph, page_id, date, utils.add_days_to_date(date, 1))
			if len(posts) > 0:
				date_to_posts[date] = posts
		except facebook.GraphAPIError as e:
			print(e)
			print(
				"skipping date %s due to GraphAPIError "
				"@collect_date_to_page_post" % date
			)
			error_count += 1
			if error_count > 10:
				raise Exception("Too many exceptions! Check API for correctness")
	return date_to_posts


def get_page_post(
	graph: facebook.GraphAPI,
	page_id: str,
	start_date: str,
	end_date: str) -> list:
	"""
	gets the posts from the page within that time frame

	Parameters
	----------
	:param graph : facebook.GraphAPI
	:param page_id : str
	:param start_date : str (DATE_FORMAT)
	:param end_date : str (DATE_FORMAT)

	Returns
	-------
	:return : list[dict<str, primitive>]
		a list of post dictionary
	"""
	api_data = graph.get_connections(
		id=page_id,
		connection_name="posts",
		limit=100,
		since=start_date,
		until=end_date,
	)["data"]
	posts = []
	for post_data in api_data:
		message = post_data.get("message", None)
		if message is None:
			continue
		posts.append({
			"created_time": post_data["created_time"],
			"story": post_data.get("story", None),
			"post_id": post_data["id"],
			"message": message,
		})
	return posts


def get_post_data(graph: facebook.GraphAPI, post_id: str) -> dict:
	"""
	get the comments and each facebook reactions from the post and
	its post id

	Parameters
	----------
	:param graph : facebook.GraphAPI
	:param post_id : str

	Returns
	-------
	:return : dict<key, int | dict<key, int>>
	"""
	comments = graph.get_connections(
		id=post_id,
		connection_name="comments",
		summary=True,
	)["summary"]["total_count"]
	reactions = graph.get_connections(
		id=post_id,
		connection_name="insights/post_reactions_by_type_total",
	)["data"][0]["values"][0]["value"]
	return {
		"comments": comments,
		"reactions": reactions,
	}


def store_data_for_date(name: str, date: str, posts: list) -> None:
	"""
	store the posts in the data folder. It stores it with the name
	name_date.json inside of data

	Parameters
	----------
	:param name : str
	:param date : str
	:param posts : list[PostData]
		PostData is a dictionary with the following keys:
		- created_time : str (DATE_FORMAT),
		- story : str,
		- post_id : str,
		- message : str,
		- comments : int,
		- reactions : dict<str, int>
		- page_data : dict<str, int>
	"""
	with open("data/%s_%s.json" % (name, date), "w") as data_file:
		json.dump(posts, data_file)


def get_comments_and_sub_comments_in_fb_object(
	graph: facebook.GraphAPI,
	object_id: str,
	verbose: bool = False) -> dict:
	"""
	Returns all the comments with their reactions and comments


	Parameters
	----------
	:param graph : facebook.GraphAPI
	:param object_id : str
	:param verbose : bool
		if true, this will print status messages

	Returns
	-------
	:return dict<k, v>
		the output has the following keys:
		- id : str (same as object_id)
		- message : str
		- created_time : str (DATETIME_FORMAT)
		- [reactions] : dict<enum{FB_REACTIONS}, int>
		- [comments] : list[dict]
			this dictionary is the same as this dict return output,
			which will keep being added until the object doesn't have
			any more comments

	Notes
	-----
	the facebook objects that have the edge /comments and /reactions
	are album, comment, photo, posts, and reactions. these edges are
	used below. make sure that the object id is one of these facebook
	objects.
	"""
	output = {}
	graph_object = graph.get_object(id=object_id)
	for field in ["message", "created_time", "id"]:
		output[field] = graph_object[field]
	# get all the reactions to this fb object
	reactions = {}
	for fb_reaction_type in FB_REACTIONS:
		try:
			count = graph.get_connections(
				id=object_id,
				connection_name="reactions",
				summary=True,
				type=fb_reaction_type,
			)["summary"]["total_count"]
			if count > 0:
				reactions[fb_reaction_type.lower()] = count
		except facebook.GraphAPIError:
			print(
				"GraphAPIError Occurred for reaction type %s for "
				"object id %s" % (fb_reaction_type, object_id)
			)
	if bool(reactions):
		output["reactions"] = reactions
	# get all the comments recursively
	graph_comments = graph.get_connections(
		id=object_id,
		connection_name="comments",
		summary=True,
	)
	comments, summary = graph_comments["data"], graph_comments["summary"]
	comment_count = summary.get("total_count", 0)
	if comment_count > 0:
		if verbose:
			print(
				"getting %d comment recursively from object id "
				"%s" % (comment_count, object_id)
			)
		for comment in comments:
			try:
				comment_obj = get_comments_and_sub_comments_in_fb_object(
					graph,
					comment["id"],
					verbose,
				)
				output.setdefault("comments", []).append(comment_obj)
			# sometimes the facebook GraphAPI returns objects with
			# invalid ids. I am not sure why this happens, but it seems
			# to happen due to an internal error on facebook's end. for
			# that reason, we handle the error by just returning an error
			# object
			except facebook.GraphAPIError:
				if verbose:
					print(
						"facebook.GraphAPIError occurred at object "
						"id %s for comment id "
						"%s" % (object_id, comment["id"])
					)
				output.setdefault("comments", []).append({
					"error": "got facebook.GraphAPIError for comment with id (%s)" % comment["id"]
				})
	return output


def store_data_with_comments_for_date(
	name: str,
	date: str,
	posts_with_comments: list) -> None:
	"""
	store the posts in the data folder. It stores it with the name
	name_date.json inside of data

	Parameters
	----------
	:param name : str
	:param date : str
	:param posts_with_comments : list[PostData]
		PostData is a dictionary with the following keys:
		- created_time : str (DATE_FORMAT),
		- story : str,
		- post_id : str,
		- message : str,
		- comments : int,
		- reactions : dict<str, int>
		- page_data : dict<str, int>
	"""
	with open("data/%s_%s_comments.json" % (name, date), "w") as data_file:
		json.dump(posts_with_comments, data_file)


should_collect_data = True
if __name__ == '__main__':
	_page_title = "mit_confessions_g_"
	_page_id = config.MIT_CONFESSIONS_ID
	_access_token = config.MIT_CONFESSIONS_ACCESS_TOKEN
	_start_date = config.MIT_CONFESSIONS_CREATION_DATE
	if should_collect_data:
		collect_page_data(
			_page_title,
			_page_id,
			access_token=_access_token,
			start_date=_start_date,
		)
