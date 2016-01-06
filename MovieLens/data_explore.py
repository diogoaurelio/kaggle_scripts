""" Data source: http://grouplens.org/datasets/movielens/ """

import os
from spark_env import spark_env as env

import numpy as np
import matplotlib.pyplot as plt

from pyspark import SparkConf, SparkContext

# CONSTANTS
APP_NAME = "movie_lens"
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CURR_DIR, "ml-100k")
USER_DATA_FILE = os.path.join(DATA_PATH, "u.user")
MOVIE_DATA_FILE = os.path.join(DATA_PATH, "u.item")
RATING_DATA_FILE = os.path.join(DATA_PATH, "u.data")


def load_spark_context():
    conf = (SparkConf().setMaster("local")\
               .setAppName(APP_NAME)\
               .set("spark.executor.memory", "1g"))
    sc = SparkContext(conf = conf)
    return sc


def load_user_data(sc):
    return sc.textFile(USER_DATA_FILE)


def load_movie_data(sc):
    return sc.textFile(MOVIE_DATA_FILE)


def load_rating_data(sc):
    return sc.textFile(RATING_DATA_FILE)


def explore_rating_data(rdata, num_users, num_movies):
    print("First rating row: {}".format(rdata.first()))
    ratings = rdata\
        .map(lambda line: line.split("\t"))\
        .map(lambda fields: int(fields[2]))
    num_ratings = ratings.count()
    max_rating = ratings.reduce(lambda x,y: max(x,y))
    min_rating = ratings.reduce(lambda x,y: min(x,y))
    mean_rating = ratings.reduce(lambda x, y: x+y) /num_ratings
    median_rating = np.median(ratings.collect())
    ratings_per_user = num_ratings / num_users
    ratings_per_movie = num_ratings / num_movies
    print("Min rating: {}".format(min_rating))
    print("Max rating: {}".format(max_rating))
    print("Avg rating: {}".format(mean_rating))
    print("Median rating: {}".format(median_rating))
    print("Avg ratings per user: %2.2f" % ratings_per_user)
    print("Avg ratings per movie: %2.2f" % ratings_per_movie)


def explore_movie_data(mdata):
    print("First Movie row: {}".format(mdata.first()))
    num_movies = mdata.count()
    print("Num. movies: {}".format(num_movies))
    movie_fields = mdata.map(lambda lines: lines.split("|"))
    years = movie_fields.map(lambda fields: fields[2]).map(lambda x: _convert_year(x))
    years_filtered = years.filter(lambda x: x!= 1900)
    # countByValue is of type action, thus caching
    movie_ages = years_filtered.map(lambda yr: 1998-yr).countByValue().cache()
    values = list(movie_ages.values())
    bins = list(movie_ages.keys())
    plt.hist(values, bins=bins, color='lightblue', normed=True)
    fig = plt.gcf()
    fig.set_size_inches(16,10)
    plt.show()

def _convert_year(x):
    try:
        return int(x[-4:])
    except:
        return 1900 # 'bad' data point


def explore_user_data(user_data):
    u1_id, u1_age, u1_gender,u1_occupation,u1_zip=user_data.first().split("|")
    print("First user data row: "
          "\nUser ID:{0} \tAge:{1}\tGender:{2}"
          "\tOccupation:{3}\tZip code:{4}"
          .format(u1_id, u1_age, u1_gender,u1_occupation,u1_zip))
    user_fields = user_data.map(lambda line: line.split("|")).cache()
    num_users = user_fields.map(lambda fields: fields[0]).count()
    num_genders = user_fields.map(lambda fields: fields[2]).distinct().count()
    num_occupations = user_fields.map(lambda fields: fields[3]).distinct().count()
    num_zip_codes = user_fields.map(lambda fields: fields[4]).distinct().count()
    print("Users: {0}\tgenders: {1}\tOccupations: {2}\tZip codes: {3}"
          .format(num_users, num_genders, num_occupations, num_zip_codes))
    print("Exploring user ages ..")
    _user_ages(user_fields)
    print("Exploring user occupations..")
    _user_occupations(user_fields)


def _user_ages(user_fields):
    ages = user_fields.map(lambda x: int(x[1])).collect()
    plt.hist(ages, bins=20, color='lightblue', normed=True)
    fig = plt.gcf()
    fig.set_size_inches(16,10)
    plt.show()


def _user_occupations(user_fields):
    count_by_occupation = user_fields\
        .map(lambda fields: (fields[3],1) )\
        .reduceByKey(lambda x, y: x+y)\
        .collect()
    x_axis1 = np.array([c[0] for c in count_by_occupation])
    y_axis1 = np.array([c[1] for c in count_by_occupation])
    # np.argsort - sort by y axis, e.g. counts
    x_axis = x_axis1[np.argsort(y_axis1)]
    y_axis = y_axis1[np.argsort(y_axis1)]
    pos = np.arange(len(x_axis))
    width = 1.0
    ax = plt.axes()
    ax.set_xticks(pos + (width/2))
    ax.set_xticklabels(x_axis)

    plt.bar(pos, y_axis, width, color='lightblue')
    plt.xticks(rotation=30)
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
    plt.show()


def main():
    sc=load_spark_context()
    # Set to true or false to 'pseudo-comment' execution
    explore_user = False
    explore_movie = False
    explore_rating = True

    if explore_user:
        user_data = load_user_data(sc)
        explore_user_data(user_data)
    if explore_user:
        mdata = load_movie_data(sc)
        explore_movie_data(mdata)
    if explore_rating:
        if 'user_data' not in locals():
            user_data = load_user_data(sc)
        if 'movie_data' not in locals():
            mdata = load_movie_data(sc)
        num_users = user_data\
            .map(lambda line: line.split("|"))\
            .map(lambda fields: fields[0])\
            .distinct()\
            .count()
        num_movies = mdata\
            .map(lambda line: line.split("|"))\
            .map(lambda fields: fields[1])\
            .distinct()\
            .count()
        rdata = load_rating_data(sc)
        explore_rating_data(rdata, num_users, num_movies)

    # terminating SparkContext
    sc.stop()


if __name__ == "__main__":
    print("Starting PySpark..")
    main()




