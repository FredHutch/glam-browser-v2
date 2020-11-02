#!/usr/bin/env python3
from expiringdict import ExpiringDict
import logging
import mysql.connector
from sqlalchemy import create_engine
from time import time
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
        cache=None,
        cache_timeout=3600,
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
        self.cache_timeout = cache_timeout

        # Set up a sqlalchemy engine
        self.sqlalchemy_engine = self.connect_sqlalchemy()
        
        # Set up a mysql engine
        self.mysql_engine = self.connect_mysql()

        # If there is no redis cache available
        if cache is None:
            # Set up an expiring cache 
            self.cache = ExpiringDict(
                max_len=100, 
                max_age_seconds=self.cache_timeout
            )
            self.using_redis = False
        else:
            # Otherwise attach the redis cache to this object
            self.cache = cache
            self.using_redis = True

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
        mycursor = self.mysql_engine.cursor()
    
        # `user`: name, password, email
        mycursor.execute("""CREATE TABLE IF NOT EXISTS user(name VARCHAR(40) NOT NULL UNIQUE, password VARCHAR(40) NOT NULL, email VARCHAR(255) NOT NULL);""")

        # `dataset`: id, uri
        mycursor.execute("""CREATE TABLE IF NOT EXISTS dataset(id VARCHAR(100) NOT NULL UNIQUE, uri VARCHAR(255) NOT NULL UNIQUE, name VARCHAR(255) NOT NULL);""")

        # `dataset_access`: dataset_id, user_name
        mycursor.execute("""CREATE TABLE IF NOT EXISTS dataset_access(dataset_id VARCHAR(100) NOT NULL, user_name VARCHAR(40) NOT NULL);""")

        # `dataset_public`: dataset_id
        mycursor.execute("""CREATE TABLE IF NOT EXISTS dataset_public(dataset_id VARCHAR(100) NOT NULL UNIQUE);""")

        # `user_bookmark`: name, bookmark, bookmark_order
        mycursor.execute("""CREATE TABLE IF NOT EXISTS user_bookmark(name VARCHAR(40) NOT NULL, bookmark VARCHAR(255) NOT NULL, bookmark_name VARCHAR(255) NOT NULL, bookmark_order INT(2) NOT NULL);""")

        # `user_tokens`: user, token, expiration
        mycursor.execute("""CREATE TABLE IF NOT EXISTS user_tokens(user VARCHAR(40) NOT NULL, token VARCHAR(255) NOT NULL, expiration VARCHAR(255) NOT NULL);""")

        # Commit all changes
        self.mysql_engine.commit()

    def read_table(self, table_name, check_cache=True):

        # Make a unique key for this table in the database
        cache_key = f"database-{table_name}"

        if check_cache and self.cache.get(cache_key) is not None:
            return self.cache.get(cache_key)

        logging.info("Reading table '{}'".format(table_name))
        # Connect to the database and open a cursor
        with self.sqlalchemy_engine.connect() as conn:

            df = pd.read_sql("SELECT * FROM {};".format(table_name), conn)

        # Add to the cache
        if self.using_redis:
            self.cache.set(
                cache_key, 
                df,
                timeout=self.cache_timeout,
            )
        else:
            self.cache[cache_key] = df

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

    def add_public_dataset(self, dataset_id=None):
        """Add a dataset to the list which is marked 'public'."""
        assert dataset_id is not None, "Please specify dataset"
        public_datasets = self.read_table("dataset_public")["dataset_id"].tolist()
        if dataset_id in public_datasets:
            logging.info(f"Dataset is already public, stopping ({dataset_id})")
        
        # Make sure this is a valid dataset
        all_datasets = self.read_table("dataset")["id"].tolist()
        assert dataset_id in all_datasets, f"Dataset ID is not valid ({dataset_id})"

        # Add it to the list
        self.write_table(
            "dataset_public", 
            pd.DataFrame([{"dataset_id": dataset_id}]),
            if_exists="append"
        )
        logging.info(f"Made dataset public: {dataset_id}")

    def list_datasets(self):
        """List all datasets in a human-readable format."""

        df = self.read_table("dataset")

        print("\nDATASETS\n\n")
        
        for _, r in df.iterrows():
            print("\n".join([
                f"{k}:\t{v}"
                for k, v in r.items()
            ]))
            print("\n------\n")

    def user_datasets(self, username):
        """Return the list of datasets which this user is allowed to access."""
        # Get the list of datasets that each user can access
        dataset_access = self.read_table("dataset_access")

        if username not in dataset_access["user_name"].values:
            return False

        return dataset_access.loc[
            dataset_access["user_name"] == username,
            "dataset_id"
        ].tolist()

    def user_can_access_dataset(self, dataset, username):
        """Return True/False indicating whether this user can access this password."""

        # Check if this dataset is publicly accessible
        if dataset in self.public_datasets():
            return True

        # Get the list of datasets that each user can access
        dataset_access = self.read_table("dataset_access")

        if username is None or username not in dataset_access["user_name"].values:
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

    # REMOVE A DATASET
    def remove_dataset(self, dataset_id=None):
        assert dataset_id is not None

        # Read the whole list of datasets
        dataset_table = self.read_table("dataset")

        if dataset_id not in dataset_table["id"].values:
            logging.info(f"Dataset ({dataset_id}) does not exist")
            return

        # Remove the dataset from the table
        dataset_table = dataset_table.loc[
            dataset_table["id"] != dataset_id
        ]

        # Write the table
        self.write_table("dataset", dataset_table, if_exists="replace")

        logging.info(f"Removed dataset: {dataset_id}")

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
    def grant_access(self, user=None, dataset=None):
        assert user is not None
        assert dataset is not None

        # Make sure that this is a valid user name
        msg = "User does not exist: {}".format(user)
        assert user in self.read_table("user")["name"].values, msg

        msg = "Dataset does not exist: {}".format(dataset)
        assert dataset in self.read_table("dataset")["id"].values, msg

        # Read the whole table of dataset access
        access_table = self.read_table("dataset_access")

        if any((access_table["dataset_id"] == dataset) & (access_table["user_name"] == user)):
            logging.info("User {} already has access to {}".format(
                user, dataset
            ))
            return

        # Add to the table
        access_table = pd.DataFrame([
            {
                "dataset_id": dataset,
                "user_name": user,
            }
        ])

        # Write the table
        self.write_table("dataset_access", access_table, if_exists="append")

        logging.info("User {} has been granted access to {}".format(
            user, dataset
        ))

    # REVOKE ACCESS FOR A USER TO ACCESS A DATASET
    def revoke_access(self, user=None, dataset=None):
        assert user is not None
        assert dataset is not None

        # Read the whole table of dataset access
        access_table = self.read_table("dataset_access")

        if not any((access_table["dataset_id"] == dataset) & (access_table["user_name"] == user)):

            # This user cannot access this dataset
            logging.info(f"User {user} does not have access to {dataset}")

            # Check to see if the username or password might not exist
            msg = "User does not exist: {}".format(user)
            if user not in self.read_table("user")["name"].values:
                print(msg)

            msg = "Dataset does not exist: {}".format(dataset)
            if dataset not in self.read_table("dataset")["id"].values:
                print(msg)

            return

        # Remove the line from the table
        access_table = access_table.loc[
            (
                (access_table["dataset_id"] != dataset) | (access_table["user_name"] != user)
            )
        ]

        # Write the table
        self.write_table("dataset_access", access_table, if_exists="replace")

        logging.info("User {} has had access to {} revoked".format(
            user, dataset
        ))

    def random_string(self, length):
        return ''.join(random.choice(string.ascii_letters) for i in range(length))

    # Get a user's bookmarks
    def user_bookmarks(self, username):

        # If the username and password is not valid
        if username is None:

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
    def save_bookmark(self, username, location, description):

        # If the username is valid
        if username is not None:

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
    def delete_bookmark(self, username, bookmark_ix):

        logging.info(f"Deleting bookmark {bookmark_ix} for {username}")

        # If the username and password is valid
        if username is None:
            logging.info(f"Username is not valid")
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

    # Generate a login token for a given user
    def get_token(self, username, password, token_length=255):

        # Make sure that the provided username and password are valid
        assert self.valid_username_password(username, password), "Invalid username/password"

        # Generate a random string
        token_string = self.random_string(token_length)

        # Set the expiration date 1 day in the future
        expiration_date = str(int(time() + (3600 * 24)))

        # Add this token to the database
        self.write_table(
            "user_tokens",
            pd.DataFrame([{
                "user": username,
                "token": token_string,
                "expiration": expiration_date
            }]),
            if_exists="append"
        )

        return token_string

    # Given a token, return the username (if they are logged in)
    def decode_token(self, token_string):

        # Make sure there is a token provided
        if token_string is None or len(token_string) < 10:
            return None

        # See if the username and expiration date are in the cache
        username, expiration_date = self.get_token_from_cache(token_string)

        # If there was no key in the cache, check the database
        if username is None:
            username, expiration_date = self.get_token_from_db(token_string)

            # If this is a valid token, then add it to the cache
            if username is not None:
                self.set_token_in_cache(
                    token_string,
                    username,
                    expiration_date
                )

        # Check the expiration date
        if expiration_date <= time():
            return None
        else:
            # Return the username
            return username
    
    def get_token_from_cache(self, token_string):
        """Check the cache for a given token."""

        # Make a unique key for this token in the cache
        cache_key = f"token-{token_string}"

        # Check the cache
        if self.cache.get(cache_key) is not None:
            return self.cache.get(cache_key)
        else:
            return None, None

    def set_token_in_cache(self, token_string, username, expiration_date):
        """Add a given token to the cache."""

        # Make a unique key for this token in the cache
        cache_key = f"token-{token_string}"

        # Add to the cache
        if self.using_redis:
            self.cache.set(
                cache_key,
                (username, expiration_date),
                timeout=self.cache_timeout,
            )
        else:
            self.cache[cache_key] = (username, expiration_date)

    def get_token_from_db(self, token_string):
        """Check the database for a given token."""

        # Get the list of all tokens
        token_df = self.read_table("user_tokens", check_cache=False)
        if token_df.shape[0] == 0:
            return None, None

        # Set the index by token
        token_df.set_index("token", inplace=True)

        # Check to see if this token is in the table
        if token_string not in token_df.index.values:
            return None, None

        # Return the username and expiration date
        username = token_df.loc[token_string, "user"]
        expiration_date = float(token_df.loc[token_string, "expiration"])

        return username, expiration_date
