#!/usr/bin/env python3

import argparse
import boto3
from collections import defaultdict
from functools import lru_cache
import h5py
import io
import logging
import numpy as np
import os
import pandas as pd
import tempfile
from .taxonomy import Taxonomy


class GLAM_INDEX:

    def __init__(
        self,
        input_hdf=None,
        output_base=None,
        overwrite=False,
        skip_enrichments=["eggNOG_desc"],
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region=os.environ.get("AWS_REGION", "us-west-2"),
    ):

        # Store the --overwrite flag
        self.overwrite = overwrite

        # Parse the S3 path
        assert output_base.startswith("s3://")
        assert "/" in output_base[5:]
        self.bucket, self.prefix = output_base[5:].split("/", 1)
        logging.info("S3 Bucket: {}".format(self.bucket))
        logging.info("S3 Prefix: {}".format(self.prefix))

        # Parameter used to skip certain elements of the input
        self.skip_enrichments = skip_enrichments    

        # Open a connection to the input HDF
        logging.info("Opening a connection to {}".format(input_hdf))
        self.store = pd.HDFStore(input_hdf, "r")

        # Walk through the HDF and store all of the groups and leaves
        self.tree = {
            path: {
                "groups": groups,
                "leaves": leaves
            }
            for path, groups, leaves in self.store.walk()
        }

        # Keep a dict to cache some of the tables from the HDF5
        self.cache = {}

        # Open a session with boto3
        self.session = boto3.session.Session(
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        # Open a connection to the AWS S3 bucket
        self.client = self.session.client("s3")

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def close(self):
        self.store.close()

    def leaves(self, path):
        return self.tree.get(path, {}).get("leaves", [])

    def write(self, df, key_path, filetype="feather"):
        """Write a single DataFrame in feather format to S3."""

        assert filetype in ["feather", "csv.gz"]

        # Append the key path to the overall base prefix
        upload_prefix = self.key_uri(key_path, filetype=filetype)

        # Check if the path already exists in S3
        if self.key_exists(key_path):
            # Check if the --overwrite flag was set
            if self.overwrite:

                # If so, we will log and continue
                logging.info("Writing over: {}".format(upload_prefix))
            else:
                # If not, we will log and stop
                logging.info("Exists: {}".format(upload_prefix))
                return

        # implicit else
        logging.info("Uploading to {}".format(upload_prefix))

        # Open a file object in memory
        with tempfile.NamedTemporaryFile() as f:

            # Save as a file object in memory
            if filetype == "feather":

                # Drop any non-standard index
                try:
                    df = df.reset_index(drop=True)
                    df.to_feather(f.name)
                except:
                    df.to_feather(f.name)

            elif filetype == "csv.gz":
                df.to_csv(f.name, compression="gzip", index=None)

            # Write the file to S3
            self.client.upload_file(
                f.name,
                self.bucket,
                upload_prefix
            )

    def key_exists(self, key_path, filetype="feather"):
        # Helper function to quickly check if a key exists in S3
        return self.path_exists(
            self.key_uri(
                key_path, filetype=filetype
            )
        )

    def key_uri(self, key_path, filetype="feather"):
        # Single definition of how the URI is formatted
        return os.path.join(
            self.prefix, "{}.{}".format(key_path, filetype)
        )

    def path_exists(self, prefix):
        """Check to see if an object exists in S3."""

        r = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
        )

        for obj in r.get('Contents', []):
            if obj['Key'] == prefix:
                return True

        return False

        
    def parse_manifest(self):
        """Read in the manifest and filter columns for visualization."""

        # Read the whole manifest
        logging.info("Reading in /manifest")
        manifest = pd.read_hdf(self.store, "/manifest")
        logging.info("Read in data from {:,} rows and {:,} columns".format(
            manifest.shape[0],
            manifest.shape[1],
        ))

        # Make sure that we have a specimen column
        assert "specimen" in manifest.columns.values

        # Drop any columns with paths to reads
        for k in ["R1", "R2", "I1", "I2"]:
            if k in manifest.columns.values:
                logging.info("Removing column {}".format(k))
                manifest = manifest.drop(columns=k)

        # Drop any columns which do not have unique values for each specimen
        for k in manifest.columns.values:
            if k == "specimen":
                continue
            # Look at just the unique values for this column
            d = manifest.reindex(columns=["specimen", k]).drop_duplicates()
            # Make sure that every specimen has only one value in this column
            if d["specimen"].value_counts().max() > 1:
                logging.info("Removing column with duplicate values: {}".format(k))
                manifest = manifest.drop(columns=k)

        # Now drop duplicates and make sure the specimen column is still unique
        manifest = manifest.drop_duplicates()
        assert manifest["specimen"].value_counts().max() == 1

        # Return the filtered manifest
        return manifest


    def parse_specimen_metrics(self):
        """Read in the specimen metrics"""
        key_name = "/summary/all"

        logging.info("Reading in {}".format(key_name))

        df = pd.read_hdf(self.store, key_name)

        # Compute `prop_reads`
        return df.assign(
            prop_reads = df["aligned_reads"] / df["n_reads"]
        )


    def parse_cag_annotations(self, max_n_cags=250000):
        """Read in the CAG annotations."""
        key_name = "/annot/cag/all"

        logging.info("Reading in {}".format(key_name))

        df = pd.read_hdf(self.store, key_name)

        # Store information for no more than `max_n_cags`
        if max_n_cags is not None:
            df = df.head(
                max_n_cags
            )

        # Compute `prop_reads`
        return df.assign(
            size_log10 = df["size"].apply(np.log10)
        ).drop(
            columns=["std_abundance"]
        )


    def parse_gene_annotations(self, tax, max_n_cags=250000):
        """Make a summary of the gene-level annotations for this subset of CAGs."""
        key_name = "/annot/gene/all"

        logging.info("Reading in {}".format(key_name))

        df = pd.read_hdf(self.store, key_name)

        logging.info("Read in {:,} annotations for {:,} CAGs".format(
            df.shape[0],
            df["CAG"].unique().shape[0],
        ))

        # Store information for no more than `max_n_cags`
        if max_n_cags is not None:
            logging.info("Limiting annotations to {:,} CAGs".format(max_n_cags))
            df = df.query(
                "CAG < {}".format(max_n_cags)
            )
            logging.info("Filtered down to {:,} annotations for {:,} CAGs".format(
                df.shape[0],
                df["CAG"].unique().shape[0],
            ))

        # Trim the `eggNOG_desc` to 100 characters, if present
        df = df.apply(
            lambda c: c.apply(lambda n: n[:100] if isinstance(n, str) and len(n) > 100 else n) if c.name == "eggNOG_desc" else c
        )

        # Summarize the number of genes with each functional annotation, if available
        if "eggNOG_desc" in df.columns.values:
            logging.info("Summarizing functional annotations")
            functional_df = self.summarize_annotations(df, "eggNOG_desc")
        else:
            functional_df = None

        # Summarize the number of genes with each taxonomic annotation, if available
        # This function also does some taxonomy parsing to count the assignments to higher levels
        if "tax_id" in df.columns.values and tax is not None:
            logging.info("Summarizing taxonomic annotations")
            counts_df, rank_summaries = self.summarize_taxonomic_annotations(
                df, tax)
        else:
            counts_df, rank_summaries = None, None

        return functional_df, counts_df, rank_summaries


    def summarize_taxonomic_annotations(self, df, tax):
        # Make a DataFrame with the number of taxonomic annotations per CAG
        counts_df = pd.concat([
            tax.make_cag_tax_df(
                cag_df["tax_id"].dropna().apply(int).value_counts()
            ).assign(
                CAG = cag_id
            )
            for cag_id, cag_df in df.reindex(
                columns=["CAG", "tax_id"]
            ).groupby(
                "CAG"
            )
        ]).reset_index(
            drop=True
        )

        logging.info("Made tax ID count table")

        rank_summaries = {
            rank: counts_df.query(
                "rank == '{}'".format(rank)
            ).drop(
                columns=["rank", "parent"]
            )
            for rank in ["phylum", "class", "order", "family", "genus", "species"]
        }

        return counts_df, rank_summaries


    def summarize_annotations(self, df, col_name):
        """Count up the unique annotations for a given CAG."""
        assert col_name in df.columns.values, (col_name, df.columns.values)
        assert "CAG" in df.columns.values, ("CAG", df.columns.values)

        return pd.DataFrame([
            {
                "label": value,
                "count": count,
                "CAG": cag_id
            }
            for cag_id, cag_df in df.groupby("CAG")
            for value, count in cag_df[col_name].dropna().value_counts().items()
        ])

    def parse_cag_abundances(self, max_n_cags=250000):
        """Read in the CAG abundances."""
        key_name = "/abund/cag/wide"

        logging.info("Reading in {}".format(key_name))

        df = pd.read_hdf(self.store, key_name)

        # Store information for no more than `max_n_cags`
        if max_n_cags is not None:
            df = df.head(
                max_n_cags
            )

        return df

    
    def add_to_cache(self, key_name):

        # If we don't already have the data
        if key_name not in self.cache:

            # Check for the data in the HDF5
            if key_name in self.store:

                logging.info("Reading in {}".format(key_name))

                self.cache[key_name] = pd.read_hdf(self.store, key_name)

            # If the data is not present
            else:

                # Add None to the cache
                self.cache[key_name] = None


    def parse_genome_containment(self, max_n_cags=250000, min_cag_prop=0.25):
        """Read in a summary of CAGs aligned against genomes."""
        key_name = "/genomes/cags/containment"

        # Make sure the data is in the cache
        self.add_to_cache(key_name)

        # Load the data from the cache
        df = self.cache[key_name]

        # First, stop if no data is present
        if df is None:

            logging.info("No genome alignments found")

            yield None, None

        # Otherwise, data is present

        # First, filter data by `max_n_cags` and `min_cag_prop`
        df = self.filter_genome_containment(df, max_n_cags, min_cag_prop)

        # Yield the subset for each CAG
        for group_ix, group_df in df.reindex(
            columns=[
                "CAG",
                "genome",
                "n_genes",
                "cag_prop",
            ]
        ).assign(
            group=df["CAG"].apply(lambda v: v % 1000)
        ).groupby("group"):
            yield group_ix, group_df.drop(columns="group")

    def parse_top_genome_containment(self, max_n_cags=250000, min_cag_prop=0.25):
        """Format a table of the top genome containment per CAG."""
        key_name = "/genomes/cags/containment"

        # Make sure the data is in the cache
        self.add_to_cache(key_name)

        # Load the data from the cache
        df = self.cache[key_name]

        # First, stop if no data is present
        if df is None:

            logging.info("No genome alignments found")

            return None

        # Otherwise, data is present

        # First, filter data by `max_n_cags` and `min_cag_prop`
        df = self.filter_genome_containment(df, max_n_cags, min_cag_prop)

        # Yield the top hit for each CAG
        return df.sort_values(
            by="n_genes", 
            ascending=False
        ).groupby(
            "CAG"
        ).head(
            1
        )

    def filter_genome_containment(self, df, max_n_cags, min_cag_prop):

        cache_key = "filtered_genome_containment"
        if self.cache.get(cache_key) is not None:
            return self.cache.get(cache_key)

        # Store information for no more than `max_n_cags`
        if max_n_cags is not None:
            logging.info("Subsetting to {:,} CAGs".format(max_n_cags))
            df = df.query(
                "CAG < {}".format(max_n_cags)
            )
            logging.info("Retained {:,} alignments for {:,} CAGs".format(
                df.shape[0],
                df["CAG"].unique().shape[0]
            ))

        # Filter alignments by `min_cag_prop`
        if min_cag_prop is not None:
            logging.info("Filtering to cag_prop >= {}".format(min_cag_prop))
            df = df.query("cag_prop >= {}".format(min_cag_prop))
            logging.info("Retained {:,} alignments for {:,} CAGs".format(
                df.shape[0],
                df["CAG"].unique().shape[0]
            ))

        # Cache the result in memory
        self.cache[cache_key] = df

        return df


    def parse_distance_matrices(self):
        """Read in each of the distance matrices in the store."""

        for distance_metric in self.leaves("/distances"):
            df = pd.read_hdf(
                self.store, 
                "/distances/{}".format(distance_metric)
            )
            yield distance_metric, df

    def parse_corncob_results(self, max_n_cags=250000):
        """Read in and parse the corncob results from the store."""

        if "corncob" in self.leaves("/stats/cag"):

            # Read in the complete set of results
            df = pd.read_hdf(
                self.store,
                "/stats/cag/corncob"
            )

            # Store information for no more than `max_n_cags`
            if max_n_cags is not None:
                logging.info("Limiting annotations to {:,} CAGs".format(max_n_cags))
                df = df.query(
                    "CAG < {}".format(max_n_cags)
                )
                logging.info("Filtered down to {:,} associations for {:,} CAGs".format(
                    df.shape[0],
                    df["CAG"].unique().shape[0],
                ))

            # Compute the log10 p_value and q_value
            logging.info("Corncob: Calculating -log10 p-values and q-values")
            df = self.add_neg_log10_values(df)

            for parameter, parameter_df in df.groupby(
                "parameter"
            ):
                if parameter != "(Intercept)":
                    yield parameter, parameter_df.drop(
                        columns="parameter"
                    ).reset_index(
                        drop=True
                    )


    def parse_enrichment_results(self):
        """Read in and parse the betta results for annotation-level associations."""

        if "betta" in self.leaves("/stats/enrichment"):

            # Read in the complete set of results
            df = pd.read_hdf(
                self.store,
                "/stats/enrichment/betta"
            )

            # Compute the log10 p_value and q_value
            logging.info("Betta: Calculating -log10 p-values and q-values")
            df = self.add_neg_log10_values(df)

            for parameter, parameter_df in df.groupby(
                "parameter"
            ):

                if parameter != "(Intercept)":

                    for annotation, annotation_df in parameter_df.groupby(
                        "annotation"
                    ):
                        yield parameter, annotation, annotation_df.drop(
                            columns=["parameter", "annotation"]
                        ).reset_index(
                            drop=True
                        )


    def add_neg_log10_values(self, df):
        for k in ["p", "q"]:

            old_col = "{}_value".format(k)
            new_col = "neg_log10_{}value".format(k)

            df = df.assign(
                NEW_COL=df[
                    old_col
                ].clip(
                    lower=df[old_col][df[old_col] > 0].min()
                ).apply(
                    np.log10
                ) * -1
            ).rename(
                columns={
                    "NEW_COL": new_col
                }
            )

        # Also add the abs(wald) as abs_wald
        if "wald" in df.columns.values:
            df = df.assign(
                abs_wald = df["wald"].abs()
            )

        return df


    def index(self):

        # GENOME ALIGNMENT DETAILS
        # To start with, we will find whether there are any genomes
        # which contain CAGs that are associated with the experimental design
    
        # Keep a summary of the analysis results which are present
        analysis_features = []

        # Copy the experiment metrics
        self.write(
            pd.read_hdf(
                self.store, 
                "/summary/experiment"
            ).reset_index(
                drop=True
            ),
            "experiment_metrics",
            filetype="csv.gz"
        )

        # Copy the genome manifest
        if "manifest" in self.leaves("/genomes"):
            self.write(
                pd.read_hdf(self.store, "/genomes/manifest"),
                "genome_manifest"
            )

        # Copy the genome summary tables
        for parameter_name in self.leaves("/genomes/summary"):

            # Write to S3
            self.write(
                pd.read_hdf(
                    self.store,
                    "/genomes/summary/{}".format(parameter_name)
                ),
                "genome_summary/{}".format(parameter_name)
            )

        # Copy over data for each individual genome
        for source_prefix, dest_prefix in [
            ("/genomes/annotations", "genome_annotations"),
            ("/genomes/detail", "genome_details"),
        ]:
            for genome_id in self.leaves(source_prefix):
                source_key = "{}/{}".format(source_prefix, genome_id)
                dest_key = "{}/{}".format(dest_prefix, genome_id)

                self.write(
                    pd.read_hdf(self.store, source_key), 
                    dest_key
                )

        # Experiment manifest
        self.write(
            self.parse_manifest(),
            "manifest",
            filetype="csv.gz"
        )
        # Specimen metrics
        self.write(
            self.parse_specimen_metrics(),
            "specimen_metrics"
        )

        # CAG annotations
        self.write(
            self.parse_cag_annotations(),
            "cag_annotations"
        )

        # Read in the CAG abundances
        self.write(
            self.parse_cag_abundances(),
            "cag_abundances"
        )

        # Read in the genome containments
        for group_ix, group_df in self.parse_genome_containment():
            if group_ix is not None:
                self.write(
                    group_df,
                    "genome_containment/{}".format(group_ix)
                )

        # Write out the table of the top genome hits for each CAG
        if "containment" in self.leaves("/genomes/cags"):
            self.write(
                self.parse_top_genome_containment(),
                "genome_top_hit"
            )


        # Read in the distance matrices
        for metric_name, metric_df in self.parse_distance_matrices():

            # Record which distance matrices are present
            analysis_features.append({
                "group": "distances",
                "key": "metric",
                "value": metric_name
            })

            # Store the actual distance matrix
            self.write(
                metric_df,
                "distances/{}".format(metric_name)
            )

        # Read in the corncob results
        for parameter, parameter_df in self.parse_corncob_results():

            # Record which corncob results are present
            analysis_features.append({
                "group": "cag_associations",
                "key": "parameter",
                "value": parameter
            })

            # Store the actual corncob results
            self.write(
                parameter_df,
                "cag_associations/{}".format(parameter)
            )

        # Read in the betta results (for enrichment of corncob associations by annotation)
        for parameter, annotation, df in self.parse_enrichment_results():

            # Skip a user-defined set of annotations
            # By default this is intended to include eggNOG_desc
            if annotation in self.skip_enrichments:
                continue

            # NOTE: The eggNOG_desc entrichments will be skipped by default. However,
            # if the user provides skip_enrichments=[], then this code block will help
            # to format that long text string.
            # Trim the eggNOG_desc labels to 100 character
            if annotation == "eggNOG_desc":
                # Trim the `eggNOG_desc` to 100 characters, if present
                df = df.apply(
                    lambda c: c.apply(lambda n: n[:100] if isinstance(n, str) and len(n) > 100 else n) if c.name == "label" else c
                )

            # Record which enrichment results are present
            analysis_features.append({
                "group": "enrichments",
                "key": "annotation",
                "value": annotation
            })

            # Store the actual betta results
            self.write(
                df,
                "enrichments/{}/{}".format(parameter, annotation)
            )

        # Assemble the `analysis_features` table
        self.write(
            pd.DataFrame(
                analysis_features
            ).drop_duplicates(),
            "analysis_features"
        )

        # Read in the taxonomy, if present
        if "taxonomy" in self.leaves("/ref"):
            tax = Taxonomy(self.store)

            # Write out the taxonomy
            self.write(
                tax.taxonomy_df.reset_index(),
                "taxonomy"
            )
        else:
            tax = None

        # Check to see if the annotations have been written out
        process_gene_annotations = True
        if all([
            self.key_exists(f"gene_annotations/{prefix}/0")
            for prefix in [
                "functional",
                "taxonomic/all",
                "taxonomic/species",
                "taxonomic/genus",
                "taxonomic/family",
            ]
        ]):
            # Check to see if the --overwrite flag is set
            if self.overwrite:
                # Log, set process_gene_annotations, and continue
                logging.info("Gene annotations were found, but will be overwritten")
                process_gene_annotations = True
            else:
                # Log, set process_gene_annotations, and continue
                logging.info("Gene annotations were found and --overwrite was not set, skipping")
                process_gene_annotations = False

        if process_gene_annotations:
            # Read in the gene annotations for just those CAGs
            functional_annot_df, counts_df, rank_summaries = self.parse_gene_annotations(tax)

        # The gene annotations should be skipped
        else:
            functional_annot_df = None
            counts_df = None
            rank_summaries = None

        # Store the summary annotation tables if the annotations are available
        if functional_annot_df is not None:
            for group_ix, group_df in functional_annot_df.assign(
                group = functional_annot_df["CAG"].apply(lambda v: v % 1000)
            ).groupby("group"):
                key_name = "gene_annotations/functional/{}".format(group_ix)
                self.write(
                    group_df.drop(columns="group"),
                    key_name
                )

        if counts_df is not None:
            for group_ix, group_df in counts_df.assign(
                group = counts_df["CAG"].apply(lambda v: v % 1000)
            ).groupby("group"):
                key_name = "gene_annotations/taxonomic/all/{}".format(group_ix)
                self.write(
                    group_df.drop(columns="group"),
                    key_name
                )

            for rank, rank_df in rank_summaries.items():
                for group_ix, group_df in rank_df.assign(
                    group = rank_df["CAG"].apply(lambda v: v % 1000)
                ).groupby("group"):
                    key_name = "gene_annotations/taxonomic/{}/{}".format(
                        rank, 
                        group_ix
                    )
                    self.write(
                        group_df.drop(columns="group"),
                        key_name
                    )

if __name__ == "__main__":

    log_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [GLAM Index] %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Write logs to STDOUT
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(
        description="""
        Index a set of geneshot results for visualization with the GLAM Browser.

        Example Usage:

        index.py <INPUT_HDF_FP> <OUTPUT_HDF_FP>

        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to results HDF5 file generated by geneshot"
    )

    parser.add_argument(
        "output",
        type=str,
        help="Base path to S3 to write out data which can be read by the GLAM Browser"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="By default, any existing data will be ignored. Use this flag to write over existing data."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Make sure the input file exists
    assert os.path.exists(args.input), "Cannot find {}".format(args.input)

    glam_index = GLAM_INDEX(
        input_hdf=args.input,
        output_base=args.output,
        overwrite=args.overwrite,
    )

    glam_index.index()
