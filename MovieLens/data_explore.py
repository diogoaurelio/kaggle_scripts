
import os
from spark_env import spark_env as env
import matplotlib as plt

from pyspark import SparkConf, SparkContext


# CONSTANTS
APP_NAME="movie_lens"
CURR_DIR=os.path.dirname(os.path.realpath(__file__))
DATA_PATH=os.path.join(CURR_DIR, "ml-100k")
USER_DATA_FILE=os.path.join(DATA_PATH, "u.user")
DATA_FILE=os.path.join(DATA_PATH, "u.data")


def load_spark_context():
    conf = (SparkConf().setMaster("local")\
               .setAppName(APP_NAME)\
               .set("spark.executor.memory", "1g"))
    sc = SparkContext(conf = conf)
    return sc


def load_user_data(sc):
    return sc.textFile(USER_DATA_FILE)


def load_data(sc):
    return sc.textFile(DATA_FILE).cache()


def explore_user_data(user_data):
    u1_id, u1_age, u1_gender,u1_occupation,u1_zip=user_data.first().split("|")
    print("First user data raw: "
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
    _user_ages(user_fields)


def _user_ages(user_fields):
    ages = user_fields.map(lambda x: int(x[1])).collect()
    plt.hist(ages, bins=20, color='loghtblue', normed=True)
    fig = plt.pyplot.gcf()
    fig.set_size_inches(16,10)

def main():
    sc=load_spark_context()
    user_data = load_user_data(sc)
    explore_user_data(user_data)
    # terminating SparkContext
    sc.stop()

if __name__ == "__main__":
    print("Starting PySpark..")
    main()




