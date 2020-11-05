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
        cache=None,
        cache_timeout=3600,
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

        # If there is no redis cache available
        self.cache_timeout = cache_timeout
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


        # # Set the index of genome IDs, keyed by dataset
        # self.genome_ix = dict()

    def read_item(self, base_path, suffix):

        # Format a unique key for this item in the cache
        cache_key = f"object-{base_path}-{suffix}"

        # Check the cache
        cached_value = self.cache.get(cache_key)
        if cached_value is not None:
            return cached_value
        
        # Get the full path to the object
        full_path = os.path.join(base_path, suffix)
        logging.info("Reading from {}".format(full_path))

        # Split up the bucket and prefix
        bucket, prefix = full_path[5:].split("/", 1)

        # Read the file object
        try:
            obj = self.client.get_object(Bucket=bucket, Key=prefix)
        except self.client.exceptions.NoSuchKey:
            # The file object does not exist
            return None

        # Parse with Pandas
        if suffix.endswith(".feather"):
            df = pd.read_feather(io.BytesIO(obj['Body'].read()))
        else:
            df = pd.read_csv(io.BytesIO(obj['Body'].read()), compression="gzip")

        # Store in the cache
        logging.info(f"Saving to the cache - {cache_key}")
        if self.using_redis:
            self.cache.set(cache_key, df, timeout=self.cache_timeout)
        else:
            self.cache[cache_key] = df

        # Return the value
        return df

    def read_keys(self, base_path, prefix=None):
        """Read all of the keys available for a single dataset."""

        # Make sure that the base path ends with "/"
        if not base_path.endswith("/"):
            base_path = f"{base_path}/"

        # Make a unique key for the cache
        cache_key = f"keys-{base_path}"

        # Check the cache
        cached_keys = self.cache.get(cache_key)
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
        logging.info(f"Saving to the cache - {cache_key}")
        if self.using_redis:
            self.cache.set(cache_key, keys, timeout=self.cache_timeout)
        else:
            self.cache[cache_key] = keys

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
        assert df is not None, "No data object was found"

        if "Unnamed: 0" in df.columns.values:
            df = df.drop(columns=["Unnamed: 0"])

        assert "specimen" in df.columns.values
        
        return df.set_index("specimen")

    def get_specimen_metrics(self, base_path):
        df = self.read_item(base_path, "specimen_metrics.feather")

        assert df is not None, "No data object was found"

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
        
        assert df is not None, "No data object was found"

        return df.set_index("variable")["value"]

    def get_cag_abundance(self, base_path, cag_id):
        df = self.read_item(
            base_path, 
            f"cag_abundances/CAG/{cag_id % 1000}.feather"
        )
        
        assert df is not None, "No data object was found"

        return df.set_index("CAG").loc[cag_id]

    def get_specimen_abundance(self, base_path, specimen_name):
        # Get the index of this specimen
        specimen_ix = self.get_specimen_ix(base_path, specimen_name)

        # Read the table with this specimen
        df = self.read_item(
            base_path, 
            f"cag_abundances/specimen/{specimen_ix}.feather"
        )
        
        assert df is not None, "No data object was found"

        m = f"Unexpected object contents (specimen: {specimen_name}, expected index: {specimen_ix}"
        assert specimen_name in df.columns.values, m

        return df.set_index("CAG")[specimen_name]

    def get_specimen_ix(self, base_path, specimen_name):
        """Get the index of a given specimen."""

        # Get the manifest
        manifest_df = self.get_manifest(base_path)

        # Make sure that this specimen is in the manifest
        assert specimen_name in manifest_df.index.values, "Invalid specimen name"

        # Return the index position of this specimen name
        for specimen_ix, n in enumerate(manifest_df.index.values):
            if n == specimen_name:
                return specimen_ix

    def get_cag_annotations(self, base_path):
        df = self.read_item(
            base_path, 
            "cag_annotations.feather"
        )
        
        assert df is not None, "No data object was found"

        return df.set_index("CAG")

    def get_distances(self, base_path, distance_metric):
        df = self.read_item(
            base_path, 
            f"distances/{distance_metric}.feather"
        )

        assert df is not None, "No data object was found"

        return df.set_index("specimen")

    def get_parameter_list(self, base_path):
        return [
            fp.replace(".feather", "")
            for fp in self.read_keys(
                base_path,
                prefix="cag_associations/"
            )
        ]

    def has_parameters(self, base_path):
        return len(self.get_parameter_list(base_path)) > 0

    def get_cag_associations(self, base_path, parameter_name):
        df = self.read_item(
            os.path.join(base_path, "cag_associations/"), 
            f"{parameter_name}.feather"
        )
        
        assert df is not None, "No data object was found"

        # Add the wald and absolute wald, also set the index
        return df.assign(
            wald=df["estimate"] / df["std_error"]
        ).assign(
            abs_wald=lambda d: d["wald"].abs()
        ).set_index(
            "CAG"
        )

    def get_enrichment_list(self, base_path, parameter_name):
        return [
            fp.replace(".feather", "")
            for fp in self.read_keys(
                base_path,
                prefix=f"enrichments/{parameter_name}/"
            )
        ]

    def get_enrichments(self, base_path, parameter_name, annotation_type):
        df = self.read_item(
            base_path,
            f"enrichments/{parameter_name}/{annotation_type}.feather"
        )

        # Not all datasets have enrichments
        if df is None:
            return None

        return df.set_index("label")

    def get_cag_taxa(self, base_path, cag_id, taxa_rank):
        # The gene-level annotations of each CAG are sharded by CAG_ID % 1000
        df = self.read_item(
            os.path.join(base_path, f"gene_annotations/taxonomic/{taxa_rank}/"),
            f"{cag_id % 1000}.feather"
        )

        # Not all CAGs have taxonomic annotations
        if df is None:
            return None
        
        return df.query(
            f"CAG == {cag_id}"
        ).drop(
            columns="CAG"
        ).apply(
            lambda c: c.fillna(0).apply(float).apply(int) if c.name in ["parent", "tax_id"] else c,
        )

    def get_cag_genome_containment(self, base_path, cag_id):
        # Genome containment is sharded by CAG_ID % 1000
        df = self.read_item(
            os.path.join(base_path, "genome_containment/CAG/"),
            f"{cag_id % 1000}.feather"
        )

        # Not all CAGs have genome containment
        if df is None:
            return None

        return df.query(
            f"CAG == {cag_id}"
        ).drop(columns="CAG")

    def get_genome_cag_containment(self, base_path, genome_id):
        # Genome containment is sharded by GENOME_ID % 1000

        # Format the key which is used to store the list of genomes for each dataset
        cache_key = f"genomes-{base_path}"

        # Check to see if we have the genome index for this dataset
        genome_index_list = self.cache.get(cache_key)
        if genome_index_list is None:
            logging.info(f"Fetching genome index for {base_path}")
            genome_index_list = pd.Series(
                range(self.get_genome_manifest(base_path).shape[0]),
                index=self.get_genome_manifest(base_path)["id"].values,
            ).apply(
                lambda ix: ix % 1000
            )

            logging.info(f"Saving to the cache - {cache_key}")
            if self.using_redis:
                self.cache.set(
                    cache_key,
                    genome_index_list,
                    timeout=self.cache_timeout
                )
            else:
                self.cache[cache_key] = genome_index_list

        # Get the index for this particular genome
        genome_ix = genome_index_list.loc[genome_id]

        # Read the containment for this genome
        df = self.read_item(
            os.path.join(base_path, "genome_containment/genome/"),
            f"{genome_ix}.feather"
        )

        # Not all CAGs have genome containment
        if df is None:
            return None

        df = df.query(
            f"genome == '{genome_id}'"
        ).drop(columns="genome")
        
        # If this genome has no containment, return None
        if df.shape[0] == 0:
            return None
        else:
            return df

    def get_top_genome_containment(self, base_path):
        # The top genome containment for all CAGs is stored in a single table
        df = self.read_item(
            base_path,
            "genome_top_hit.feather"
        )

        # Not all CAGs have genome containment
        if df is None:
            return None

        return df.set_index("CAG")

    def get_genome_manifest(self, base_path):
        return self.read_item(base_path, "genome_manifest.feather")

    def has_genomes(self, base_path):
        return "genome_manifest.feather" in self.read_keys(base_path)

    def get_genome_parameters(self, base_path):
        """Instead of returning all of the parameters, just return those which have genome summaries."""
        return [
            fp.replace(".feather", "")
            for fp in self.read_keys(
                base_path,
                prefix="genome_summary/"
            )
            if fp != "Intercept.feather"
        ]

    def has_genome_parameters(self, base_path):
        return len(self.get_genome_parameters(base_path)) > 0

    def get_genome_associations(self, base_path, parameter):
        return self.read_item(
            base_path,
            f"genome_summary/{parameter}.feather"
        )

    def get_genome_details(self, base_path, genome_id):
        return self.read_item(
            base_path,
            f"genome_details/{genome_id}.feather"
        ).sort_values(
            by=["contig", "contig_start"]
        )

    def get_genomes_with_details(self, base_path):
        # There's enough computing in this call to merit caching
        cache_key = f"genomes-with-details-{base_path}"

        # Check the cache
        cached_value = self.cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        # Get the list of genomes which have details available
        genome_id_list = [
            fp.replace(".feather", "")
            for fp in self.read_keys(
                base_path,
                prefix="genome_details/"
            )
            if fp.endswith(".feather")
        ]

        # Get the number of genes in the top hit for each CAG
        top_hits_df = self.get_top_genome_containment(base_path).reset_index()
        
        # Make sure that we're only considering genomes which are in both lists
        genome_id_list = list(set(genome_id_list) & set(top_hits_df["genome"].tolist()))

        # Filter down to the top hit per genome
        top_hits_df = top_hits_df.sort_values(
            by="n_genes",
            ascending=False
        ).groupby(
            "genome"
        ).head(
            1
        )

        # Get the list sorted by top hit
        genome_id_list = top_hits_df.set_index(
            "genome"
        ).reindex(
            index=genome_id_list
        ).sort_values(
            by="n_genes",
            ascending=False
        ).index.values

        # Store in the cache
        logging.info(f"Saving to the cache - {cache_key}")
        if self.using_redis:
            self.cache.set(cache_key, genome_id_list, timeout=self.cache_timeout)
        else:
            self.cache[cache_key] = genome_id_list

        return genome_id_list


    def get_genome_annotations(self, base_path, genome_id):
        return self.read_item(
            base_path,
            f"genome_annotations/{genome_id}.feather"
        )

    def has_functional_annotations(
        self, 
        base_path, 
        func_prefix="gene_annotations/functional/"
    ):
        return len(self.read_keys(base_path, prefix=func_prefix)) > 0

    def get_cag_functions(self, base_path, cag_id):
        # All gene annotations are sharded by CAG_ID % 1000
        df = self.read_item(
            os.path.join(base_path, f"gene_annotations/functional/"),
            f"{cag_id % 1000}.feather"
        )

        # Not all CAGs have genome containment
        if df is None:
            return None

        return df.query(
            f"CAG == {cag_id}"
        ).drop(columns="CAG")

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
