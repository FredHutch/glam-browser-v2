#!/usr/bin/env python3

import boto3
from expiringdict import ExpiringDict
from collections import defaultdict
from filelock import Timeout, FileLock
import io
import json
import logging
import numpy as np
import os
import pandas as pd
from time import sleep

class GLAM_IO:

    def __init__(
        self,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        region=os.environ.get("AWS_REGION", "us-west-2"),
        **kwargs,
    ):

        # Open a session with boto3
        self.session = boto3.session.Session(
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        # Open a connection to the AWS S3 bucket
        self.client = self.session.client("s3")

        # Set up an expiring cache for data objects
        self.cache = defaultdict(lambda: ExpiringDict(max_len=100, max_age_seconds=3600))
        # and for keys
        self.key_cache = ExpiringDict(max_len=100, max_age_seconds=3600)

    def read_item(self, base_path, suffix):

        # Check the cache
        cached_value = self.cache.get(base_path, {}).get(suffix)
        if cached_value is not None:
            return cached_value
        
        # Get the full path to the object
        full_path = os.path.join(base_path, suffix)
        logging.info("Reading from {}".format(full_path))

        # Split up the bucket and prefix
        bucket, prefix = full_path[5:].split("/", 1)

        # Read the file object
        obj = self.client.get_object(Bucket=bucket, Key=prefix)

        # Parse with Pandas
        if suffix.endswith(".feather"):
            df = pd.read_feather(io.BytesIO(obj['Body'].read()))
        else:
            df = pd.read_csv(io.BytesIO(obj['Body'].read()), compression="gzip")

        # Store in the cache
        self.cache[base_path][suffix] = df

        # Return the value
        return df

    def read_keys(self, base_path, prefix=None):
        """Read all of the keys available for a single dataset."""

        # Make sure that the base path ends with "/"
        if not base_path.endswith("/"):
            base_path = f"{base_path}/"

        # Check the cache
        cached_keys = self.key_cache.get(base_path)
        if cached_keys is not None:
            return self.filter_keys_by_prefix(cached_keys, prefix=prefix)

        logging.info("Reading keys below {}".format(base_path))

        # Split up the bucket and prefix
        bucket, prefix = base_path[5:].split("/", 1)

        # List objects
        r = self.client.list_objects_v2(
            Bucket=bucket, 
            Prefix=prefix
        )

        # Add the keys to a master list
        keys = [
            i["Key"].replace(prefix, "")
            for i in r["Contents"]
        ]

        # If there are more items remaining to read
        while r["IsTruncated"]:
            # Read more using the continuation token
            r = self.client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                ContinuationToken=r["NextContinuationToken"]
            )

            # Add each item to the list
            for i in r["Contents"]:
                keys.append(i["Key"].replace(prefix, ""))

        # Store in the cache
        self.key_cache[base_path] = keys

        return self.filter_keys_by_prefix(keys, prefix=prefix)

    def filter_keys_by_prefix(self, keys, prefix=None):
        """Subset the keys from a repository to those below a given 'folder'"""
        if prefix is None:
            return keys

        else:
            return [
                i[len(prefix):]
                for i in keys
                if i.startswith(prefix)
            ]

    def get_manifest(self, base_path):
        df = self.read_item(base_path, "manifest.csv.gz")

        if "Unnamed: 0" in df.columns.values:
            df = df.drop(columns=["Unnamed: 0"])

        assert "specimen" in df.columns.values
        
        return df.set_index("specimen")

    def get_specimen_metrics(self, base_path):
        df = self.read_item(base_path, "specimen_metrics.feather")

        # Fix the mismatch in index variable type, if any
        df = pd.DataFrame({
            col_name: {
                str(k): v
                for k, v in zip(
                    df["specimen"].values,
                    df[col_name].values
                )
                if pd.isnull(v) is False
            }
            for col_name in df.columns.values
            if col_name != "specimen"
        })

        return df

    def get_dataset_metrics(self, base_path):
        df = self.read_item(base_path, "experiment_metrics.csv.gz")
        if df is not None:
            return df.set_index("variable")["value"]

    def get_cag_abundances(self, base_path):
        return self.read_item(
            base_path, 
            "cag_abundances.feather"
        ).set_index(
            "CAG"
        )

    def get_cag_annotations(self, base_path):
        return self.read_item(
            base_path, 
            "cag_annotations.feather"
        ).set_index(
            "CAG"
        )

    def get_distances(self, base_path, distance_metric):
        return self.read_item(
            base_path, 
            f"distances/{distance_metric}.feather"
        ).set_index(
            "specimen"
        )

    def get_parameter_list(self, base_path):
        return [
            fp.replace(".feather", "")
            for fp in self.read_keys(
                base_path,
                prefix="cag_associations/"
            )
            if fp.startswith()
        ]

    def get_cag_associations(self, base_path, parameter_name):
        return self.read_item(
            os.path.join(base_path, "cag_associations/"), 
            f"{parameter_name}.feather"
        ).set_index(
            "CAG"
        )

    def get_enrichment_list(self, base_path, parameter_name):
        return [
            fp.replace(".feather", "")
            for fp in self.read_keys(
                base_path,
                prefix=f"cag_associations/{parameter_name}/"
            )
        ]

    def get_enrichments(self, base_path, parameter_name, annotation_type):
        return self.read_item(
            os.path.join(base_path, "cag_associations/"), 
            f"{parameter_name}/{annotation_type}.feather"
        )

    def get_cag_taxa(self, base_path, cag_id, taxa_rank):
        # The gene-level annotations of each CAG are sharded by CAG_ID % 1000
        return self.read_item(
            os.path.join(base_path, f"gene_annotations/taxonomic/{taxa_rank}/"),
            f"{cag_id % 1000}.feather"
        ).query(
            f"CAG == {cag_id}"
        )

    def has_genomes(self, base_path):
        return "genome_manifest.feather" in self.read_keys(base_path)

def hdf5_get_keys(
    fp, 
    group_path, 
    timeout=5, 
    retry=5,
):
    """Read keys from a group in the HDF5 store."""

    # Set up a file lock to prevent multiple concurrent access attempts
    lock = FileLock("{}.lock".format(fp), timeout=timeout)

    # Read in the keys
    try:
        with lock:
            with h5py.File(fp, "r") as f:
                try:
                    key_list = list(f[group_path].keys())
                except:
                    return None
    except Timeout:

        sleep(retry)
        return hdf5_get_keys(
            fp, 
            group_path, 
            timeout=timeout,
            retry=retry,
        )

    return key_list


def hdf5_taxonomy(fp):
    return hdf5_get_item(
        fp, 
        "/taxonomy"
    ).apply(
        lambda c: c.fillna(0).apply(float).apply(int) if c.name in ["parent", "tax_id"] else c,
    ).set_index(
        "tax_id"
    )
