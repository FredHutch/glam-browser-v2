#!/usr/bin/env python3
from expiringdict import ExpiringDict
import logging
import mysql.connector
from sqlalchemy import create_engine
import pandas as pd
import random
import string
from .index import GLAM_INDEX

class GLAM_DB:

    def __init__(
        self,
        db_name="glam_db",
        db_username=None,
        db_password=None,
        db_host=None,
        db_port=None,
        **kwargs,
    ):

        # Store all of the variables for the database
        assert db_name is not None, "Must specify database name"
        assert db_username is not None, "Must specify database username"
        assert db_password is not None, "Must specify database password"
        assert db_host is not None, "Must specify database host"
        assert db_port is not None, "Must specify database port"
        self.db_name = db_name
        self.db_username = db_username
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port

        # Set up a sqlalchemy engine
        self.sqlalchemy_engine = self.connect_sqlalchemy()
        
        # Set up a mysql engine
        self.mysql_engine = self.connect_mysql()

        # Set up an expiring cache
        self.cache = ExpiringDict(max_len=100, max_age_seconds=10)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
        
    def close(self):
        return

    def connect_sqlalchemy(self):
        logging.info("Connecting to {host}:{port}/{database} as {username}".format(
            username=self.db_username,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name
        ))

        return create_engine(
            "mysql://{username}:{password}@{host}:{port}/{database}".format(
                username=self.db_username,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port,
                database=self.db_name
            )
        )

    def connect_mysql(self):
        # CONNECT TO THE DATABASE
        return mysql.connector.connect(
            buffered=True,
            user=self.db_username,
            password=self.db_password,
            host=self.db_host,
            port=int(self.db_port),
            auth_plugin='mysql_native_password',
            database=self.db_name,
        )

    def setup(self):
        """Set up all of the tables in the database that we need."""

        # Connect to the database and open a cursor
        with self.mysql_engine.cursor() as mycursor:
        
            # `user`: name, password, email
            mycursor.execute("""CREATE TABLE IF NOT EXISTS user(name VARCHAR(40) NOT NULL UNIQUE, password VARCHAR(40) NOT NULL, email VARCHAR(255) NOT NULL);""")

            # `dataset`: id, uri
            mycursor.execute("""CREATE TABLE IF NOT EXISTS dataset(id VARCHAR(100) NOT NULL UNIQUE, uri VARCHAR(255) NOT NULL UNIQUE, name VARCHAR(255) NOT NULL);""")

            # `dataset_access`: dataset_id, user_name
            mycursor.execute("""CREATE TABLE IF NOT EXISTS dataset_access(dataset_id VARCHAR(100) NOT NULL UNIQUE, user_name VARCHAR(40) NOT NULL UNIQUE);""")

            # `dataset_public`: dataset_id
            mycursor.execute("""CREATE TABLE IF NOT EXISTS dataset_public(dataset_id VARCHAR(100) NOT NULL UNIQUE);""")

            # `user_bookmark`: name, bookmark, bookmark_order
            mycursor.execute("""CREATE TABLE IF NOT EXISTS user_bookmark(name VARCHAR(40) NOT NULL, bookmark VARCHAR(255) NOT NULL, bookmark_name VARCHAR(255) NOT NULL, bookmark_order INT(2) NOT NULL);""")

            # Commit all changes
            self.mysql_engine.commit()

    def read_table(self, table_name, check_cache=True):
        if check_cache and self.cache.get(table_name) is not None:
            return self.cache.get(table_name)

        logging.info("Reading table '{}'".format(table_name))
        # Connect to the database and open a cursor
        with self.sqlalchemy_engine.connect() as conn:

            df = pd.read_sql("SELECT * FROM {};".format(table_name), conn)

        # Add to the cache
        self.cache[table_name] = df

        return df

    def write_table(self, table_name, df, if_exists="fail"):
        logging.info("Writing to table '{}'".format(table_name))
        # Connect to the database and open a cursor
        with self.sqlalchemy_engine.connect() as conn:

            df.to_sql(table_name, conn, if_exists=if_exists, index=False)

    def valid_username_password(self, username, password):
        user_table = self.read_table("user").set_index("name")

        if username not in user_table.index.values:
            return False

        return password == user_table.loc[username, "password"]

    def public_datasets(self):
        """Return the list of datasets which are marked 'public'."""
        return self.read_table("dataset_public")["dataset_id"].tolist()

    def user_datasets(self, username, password):
        """Return the list of datasets which this user is allowed to access."""
        if self.valid_username_password(username, password) is False:
            return []

        # Get the list of datasets that each user can access
        dataset_access = self.read_table("dataset_access")

        if username not in dataset_access["user_name"].values:
            return False

        return dataset_access.loc[
            dataset_access["user_name"] == username,
            "dataset_id"
        ].tolist()

    def user_can_access_dataset(self, dataset, username, password):
        """Return True/False indicating whether this user can access this password."""
        if self.valid_username_password(username, password) is False:
            return False

        # Get the list of datasets that each user can access
        dataset_access = self.read_table("dataset_access")

        if username not in dataset_access["user_name"].values:
            return False

        return dataset in dataset_access.loc[
            dataset_access["user_name"] == username,
            "dataset_id"
        ].values

    # ADD A USER TO THE DATABASE
    def add_user(self, username=None, email=None, password_length=20):

        assert username is not None
        assert email is not None

        # Read the whole list of users
        user_table = self.read_table("user")

        if username in user_table["name"].values:
            logging.info("User ({}) already exists".format(username))
            return

        # Make a new password
        new_password = self.random_string(password_length)

        # Add to the table
        user_table = pd.DataFrame([
            {
                "name": username, 
                "password": new_password, 
                "email": email
        }])

        # Write the table
        self.write_table("user", user_table, if_exists="append")

        logging.info("New user: {}\nPassword: {}".format(username, new_password))


    # SET A USER'S PASSWORD
    def set_password(self, username=None, password=None):
        assert username is not None
        assert password is not None

        # Read the whole list of users
        user_table = self.read_table("user").set_index("name")

        if username not in user_table.index.values:
            logging.info("User ({}) does not exist".format(username))
            return

        # Set the password for this user
        user_table.at[username, "password"] = password

        # Write the table
        self.write_table("user", user_table.reset_index(), if_exists="replace")

        logging.info("Set new password for {}".format(username))

    # INDEX A DATASET
    def index_dataset(self, fp=None, uri=None):
        glam_index = GLAM_INDEX(
            input_hdf=fp,
            output_base=uri,
        )
        glam_index.index()

    # ADD A DATASET
    def add_dataset(self, dataset_id=None, name=None, uri=None):
        assert dataset_id is not None
        assert name is not None
        assert uri is not None
        assert uri.startswith("s3://")

        # Read the whole list of datasets
        dataset_table = self.read_table("dataset")

        if dataset_id in dataset_table["id"].values:
            logging.info("Dataset ({}) already exists".format(dataset_id))
            return

        # Add to the table
        dataset_table = pd.DataFrame([
            {
                "id": dataset_id,
                "uri": uri,
                "name": name,
            }
        ])

        # Write the table
        self.write_table("dataset", dataset_table, if_exists="append")

        logging.info("ID: {}".format(dataset_id))
        logging.info("Name: {}".format(name))
        logging.info("URI: {}".format(uri))

    # Get the name of a dataset
    def get_dataset_name(self, dataset_id=None):
        assert dataset_id is not None, "Please specify a dataset ID"

        # Read the whole list of datasets
        dataset_table = self.read_table("dataset")

        if dataset_id not in dataset_table["id"].values:
            return None

        return dataset_table.set_index("id").loc[dataset_id, "name"]

    # Get the base location of a dataset
    def get_dataset_uri(self, dataset_id=None):
        assert dataset_id is not None, "Please specify a dataset ID"

        # Read the whole list of datasets
        dataset_table = self.read_table("dataset")

        if dataset_id not in dataset_table["id"].values:
            return None

        return dataset_table.set_index("id").loc[dataset_id, "uri"]

    # GRANT ACCESS FOR A USER TO ACCESS A DATASET
    def grant_access(self, user_name=None, dataset_name=None):
        assert user_name is not None
        assert dataset_name is not None

        # Make sure that this is a valid user name
        msg = "User does not exist: {}".format(user_name)
        assert user_name in self.read_table("user")["name"].values, msg

        msg = "Dataset does not exist: {}".format(dataset_name)
        assert dataset_name in self.read_table("dataset")["id"].values, msg

        # Read the whole table of dataset access
        access_table = self.read_table("dataset_access")

        if any((access_table["dataset_id"] == dataset_name) & (access_table["user_name"] == user_name)):
            logging.info("User {} already has access to {}".format(
                user_name, dataset_name
            ))
            return

        # Add to the table
        access_table = pd.DataFrame([
            {
                "dataset_id": dataset_name,
                "user_name": user_name,
            }
        ])

        # Write the table
        self.write_table("dataset_access", access_table, if_exists="append")

        logging.info("User {} has been granted access to {}".format(
            user_name, dataset_name
        ))

    def random_string(self, length):
        return ''.join(random.choice(string.ascii_letters) for i in range(length))

    # Get a user's bookmarks
    def user_bookmarks(self, username, password):

        # If the username and password is not valid
        if self.valid_username_password(username, password) is False:

            # Return an empty list
            return []

        else:
            # Read the table of bookmarks
            bookmark_df = self.read_table("user_bookmark", check_cache=False)

            # If this user doesn't have any bookmarks
            if "name" not in bookmark_df.columns.values or username not in bookmark_df["name"].values:

                # Return an empty list
                return []

            else:
                # Return the bookmarks for this user
                return bookmark_df.loc[
                    bookmark_df["name"].values == username
                ].sort_values(
                    by="bookmark_order"
                ).to_dict(orient="records")

    # Save a bookmark for a user
    def save_bookmark(self, username, password, location, description):

        # If the username and password is valid
        if self.valid_username_password(username, password):

            # Read the table of bookmarks
            bookmark_df = self.read_table("user_bookmark", check_cache=False)

            # If this user doesn't have any bookmarks
            if "name" not in bookmark_df.columns.values or username not in bookmark_df["name"].values:

                # Then the newest index position is 0
                ix = 0

            else:
                # Otherwise, the newest index position is the maximum existing index + 1
                ix = bookmark_df.loc[
                    bookmark_df["name"].values == username,
                    "bookmark_order"
                ].max() + 1

            df_to_write = pd.DataFrame([{
                "name": username,
                "bookmark": location,
                "bookmark_name": description,
                "bookmark_order": ix
            }])

            # Append to the table in the database
            self.write_table(
                "user_bookmark",
                df_to_write,
                if_exists="append"
            )

    # Delete a bookmark for a user
    def delete_bookmark(self, username, password, bookmark_ix):

        logging.info(f"Deleting bookmark {bookmark_ix} for {username}")

        # If the username and password is valid
        if not self.valid_username_password(username, password):
            logging.info(f"Username is not valid ({username})")
            return

        # Read the table of bookmarks
        bookmark_df = self.read_table("user_bookmark", check_cache=False)

        # Stop if this user doesn't have any bookmarks
        if "name" not in bookmark_df.columns.values or username not in bookmark_df["name"].values:
            logging.info(f"User ({username}) does not have any bookmarks")
            return

        # Get the list of bookmarks for this user
        user_bookmark_df = bookmark_df.loc[
            bookmark_df["name"].values == username
        ]

        # If this index isn't in the table, stop
        if bookmark_ix not in user_bookmark_df["bookmark_order"].values:
            logging.info(f"User ({username}) does not have a bookmark with index {bookmark_ix}")
            return

        else:

            # Remove this bookmark from the table
            user_bookmark_df = user_bookmark_df.loc[
                user_bookmark_df["bookmark_order"] != bookmark_ix
            ]

            # Now reset the bookmark order
            user_bookmark_df = user_bookmark_df.drop(
                columns=["bookmark_order"]
            ).assign(
                bookmark_order = list(range(user_bookmark_df.shape[0]))
            )

            # Connect to the database and open a cursor
            with self.mysql_engine.cursor() as mycursor:

                # Drop all of the bookmarks for this user
                mycursor.execute(f"DELETE FROM user_bookmark WHERE name = '{username}';")

            # Commit all changes
            self.mysql_engine.commit()

            # Now write the new table of bookmarks for this user
            self.write_table(
                "user_bookmark",
                user_bookmark_df,
                if_exists="append"
            )
