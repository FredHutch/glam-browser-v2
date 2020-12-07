#!/usr/bin/env python3

import argparse
import boto3
from collections import defaultdict
from copy import deepcopy
from fastcluster import linkage
from functools import lru_cache
import json
import logging
import math
import networkx as nx
import numpy as np
import os
import pandas as pd
from scipy.cluster.hierarchy import cophenet, optimal_leaf_ordering
from scipy.spatial.distance import squareform, euclidean, pdist
import sys
import tempfile
from time import time


def glam_network(
    summary_hdf,
    detail_hdf,
    output_folder,
    metric="euclidean",
    method="complete",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region=os.environ.get("AWS_REGION", "us-west-2"),
    min_node_size=10,
    testing=False,
):
    """Use co-abundance and co-assembly to render genes in a network layout."""

    if testing:
        logging.info("RUNNING IN TESTING MODE")

    # Set up the connection to the output
    writer = DataWriter(
        output_folder=output_folder,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region=region,
    )

    # 1 - Build a set of gene membership for all contigs
    contig_gene_sets = build_contig_dict(
        summary_hdf, 
        detail_hdf, 
        testing=testing,
    )

    # 2 - Combine contigs which satisfy a conservative overlap threshold
    merged_contig_sets = combine_overlapping_contigs(contig_gene_sets)

    # 3 - Build a network of gene linkage groups
    LN = LinkageNetwork(summary_hdf, merged_contig_sets, testing=testing)

    #####################
    # PRUNE THE NETWORK #
    #####################

    # Trim terminal nodes
    merge_terminal_nodes(LN, max_size=min_node_size)

    # Remove small unconnected nodes
    remove_unconnected_nodes(LN, max_size_unconnected=min_node_size)

    # Trim by connection ratio
    for max_ratio in [-3, -2, -1, 0]:
        merge_nodes_by_ratio(LN, max_ratio=max_ratio)

    # Trim terminal nodes
    merge_terminal_nodes(LN, max_size=min_node_size)

    # Remove small unconnected nodes
    remove_unconnected_nodes(LN, max_size_unconnected=min_node_size)

    # Rename the nodes ordinally, sorted by size
    rename_nodes(LN)

    ############################
    # FINISHED PRUNING NETWORK #
    ############################

    # Read in the genome information
    genome_gene_sets = read_genome_gene_sets(detail_hdf)
    
    genome_overlap_df = count_genome_overlap(LN, genome_gene_sets)
    
    G = make_network(LN)

    assert len(G.nodes) > 0, "No linkage groups found, stopping"
    
    if len(genome_overlap_df) > 0:
        logging.info("Adding genome data to graph")
        add_genomes_to_graph(LN, G, genome_overlap_df)
    else:
        logging.info("No genome data found -- skipping")

    # Read the taxonomy
    tax = Taxonomy(summary_hdf)

    # Read the taxonomic assignment per gene
    taxonomy_df = read_taxonomy_df(summary_hdf)

    # Count up the number of genes assigned to each linkage group, in a long Data Frame
    tax_counts_df = count_tax_assignments(LN, taxonomy_df, tax)

    # Write out the tax counts in shards
    write_out_shards(writer, tax_counts_df, 'LG/tax-counts')

    # Summarize each connected set of linkage groups on the basis of taxonomic spectrum
    # In the process, record the those discrete connected linkage groups
    tax_spectrum_df, node_groupings = make_tax_spectrum_df(LN, G, taxonomy_df, tax)

    # Summarize each linkage group on the basis of abundance
    lg_abund_df = get_linkage_group_abundances(LN, detail_hdf)

    # Write out the table of relative abundances for each linkage group
    logging.info("Writing out relative abundance of linkage groups")
    write_out_shards(
        writer,
        lg_abund_df.reset_index(
        ).rename(
            columns={'index': 'linkage_group'}
        ),
        "LG/abundance/all"
    )

    # Write out the graph also in tabular format
    logging.info("Writing out graph object in tabular format")
    write_edges(writer, G, output_folder)

    # Write out the table of which genes are part of which linkage group
    logging.info("Writing out gene index")
    gene_index_df = pd.DataFrame([
        {
            "gene": gene_name,
            "linkage_group": group_name
        }
        for group_name, gene_name_list in LN.group_genes.items()
        for gene_name in gene_name_list
    ])
    writer.write(
        "LG/gene-index",
        gene_index_df,
    )

    # Write out the annotations for the genes in each linkage group
    write_gene_annotations(
        writer,
        gene_index_df,
        summary_hdf
    )

    # Write out a small JSON with the number of genes in the network
    writer.write(
        "LG/network-size",
        {
            "genes": int(gene_index_df.shape[0]),
            "nodes": int(gene_index_df["linkage_group"].unique().shape[0])
        }
    )

    # Compute the size of each linkage group
    lg_size = gene_index_df["linkage_group"].value_counts()

    # Write out the coordinates to plot each linkage group
    linkage_partition(
        writer,
        tax_spectrum_df,
        gene_index_df,
        node_groupings,
        G,
        method=method,
        metric=metric,
    )

    logging.info("Computing taxonomic labels")

    # For each taxonomic level, write out the best hit for each linkage group
    for tax_rank in ["phylum", "class", "order", "family", "genus", "species"]:

        writer.write(
            f"LG/taxonomic/{tax_rank}",
            pick_top_taxon(
                tax_spectrum_df, # The proportional assignment of genes per subnetwork
                node_groupings,  # Subnetwork grouping of linkage groups
                lg_size,         # The number of genes per linkage group
                tax.tax,         # The taxonomy Data Frame
                tax_rank         # Specific taxonomic rank
            )
        )

        logging.info(
            f"Wrote out taxonomic assignments at the {tax_rank} level")

    # Write out the relative abundances of each linkage group
    write_lg_abund(
        writer,
        lg_abund_df,
        summary_hdf,
    )

    # Write out the mean wald metric by linkage group
    write_wald(
        writer,
        summary_hdf,
        gene_index_df,
    )

def write_gene_annotations(writer, gene_index_df, summary_hdf):
    """Write out the annotations for the genes in each linkage group."""
    
    # Read in the annotations for every gene
    logging.info(f"Reading in /annot/gene/all from {summary_hdf}")
    annot_df = pd.read_hdf(
        summary_hdf,
        "/annot/gene/all",
        columns=['gene', 'length', 'eggNOG_desc', 'tax_id', 'tax_name']
    )

    # Add the annotation for each linkage group
    annot_df = annot_df.assign(
        linkage_group = annot_df['gene'].apply(
            gene_index_df.set_index("gene")["linkage_group"].get
        )
    )

    # Write out the annotation in shards
    logging.info("Writing out annotations for genes in every linkage group")
    write_out_shards(
        writer,
        annot_df,
        "LG/gene-annotations"
    )

def write_out_shards(writer, df, output_folder, ix_col='linkage_group', mod=1000):
    """Write out a DataFrame in shards."""
    
    assert 'group_ix' not in df.columns.values, "Table cannot contain `group_ix`"

    # Make sure that the `ix_col` is formatted as an integer
    df = df.apply(
        lambda c: c.apply(int) if c.name == ix_col else c
    )

    logging.info(f"Writing out a table with {df.shape[0]:,} rows to {output_folder} in {mod} shards")

    # Iterate over the DataFrame, sharding `ix_col` by `mod`
    for group_ix, group_df in df.assign(
        group_ix = df[ix_col].apply(lambda i: i % mod).apply(int)
    ).groupby(
        'group_ix'
    ):
        writer.write(
            f"{output_folder}/{group_ix}",
            group_df.drop(
                columns='group_ix'
            )
        )


def assembly_key_list(detail_hdf):
    """Get the list of specimens which have assembly informtion."""
    with pd.HDFStore(detail_hdf, 'r') as store:
        return [
            p
            for p in store
            if p.startswith("/abund/allele/assembly/")
        ]


def get_cag_size(summary_hdf):
    """Get the size of each CAG."""
    return read_cag_dict(summary_hdf).value_counts()


def read_cag_dict(summary_hdf):
    """Get the dict matching each gene to a CAG."""
    logging.info("Reading the assignment of genes to CAGs")

    df = pd.read_hdf(
        summary_hdf,
        "/annot/gene/cag",
    )

    return df.set_index(
        "gene"
    )["CAG"]


def read_contig_info(summary_hdf, detail_hdf, path_name, remove_edge=True, cache=None):
    """Read the contig information for a single assembly."""

    # Read the table
    logging.info(f"Reading {path_name}")
    with pd.HDFStore(detail_hdf, 'r') as store:
        df = pd.read_hdf(store, path_name)

    # Remove genes which don't match the gene catalog
    df = df.loc[df["catalog_gene"].isnull().apply(lambda v: v is False)]

    # Remove genes which don't have a CAG assigned
    cag_dict = read_cag_dict(summary_hdf)
    df = df.loc[df["catalog_gene"].apply(
        cag_dict.get).isnull().apply(lambda v: v is False)]

    # Remove genes at the edge of the contig
    if remove_edge:
        df = df.query(
            "start_type != 'Edge'"
        )

    # Add the specimen name to the contig name
    df = df.assign(
        contig=df["contig"] + "_" + df["specimen"].apply(str)
    ).reset_index(
        drop=True
    )

    return df


@lru_cache(maxsize=1)
def genome_key_list(summary_hdf):
    """Get the list of genomes which have alignment information."""
    with pd.HDFStore(summary_hdf, 'r') as store:
        return [
            p
            for p in store
            if p.startswith("/genomes/detail/")
        ]


def build_contig_dict(
    summary_hdf, 
    detail_hdf, 
    remove_edge=False, 
    min_genes=2, 
    testing=False,
):
    """Read in a set with the membership of the genes in every contig."""

    # Set up a dict, keyed by contig
    contig_membership = {}

    # Iterate over every assembly
    for path_name in assembly_key_list(detail_hdf):

        # Read in the table of contig membership
        contig_df = read_contig_info(
            summary_hdf,
            detail_hdf,
            path_name,
            remove_edge=remove_edge,
        )

        # Iterate over every contig
        logging.info(
            f"Adding {contig_df.shape[0]:,} genes across {contig_df['contig'].unique().shape[0]:,} contigs from {path_name.split('/')[-1]}")
        start_time = time()
        i = 0
        for contig_name, contig_genes in contig_df.groupby("contig"):

            # Skip contigs with less than `min_genes` genes
            if contig_genes.shape[0] >= min_genes:

                # Add the set to the dict
                contig_membership[contig_name] = set(
                    contig_genes["catalog_gene"].tolist())

            i += 1
            if testing and i == 100:
                logging.info("TESTING -- STOPPING EARLY")
                break

        logging.info(f"Done - {round(time() - start_time, 1):,} seconds elapsed")
        start_time = time()

    logging.info(f"Returning {len(contig_membership):,} sets of gene membership")

    return contig_membership


def index_contig_gene_sets(contig_gene_sets):
    start_time = time()
    gene_index = defaultdict(set)
    for contig_name, gene_set in contig_gene_sets.items():
        for gene_name in list(gene_set):
            gene_index[gene_name].add(contig_name)
    logging.info(
        f"Made index for {len(contig_gene_sets):,} contigs and {len(gene_index):,} genes -- {round(time() - start_time, 1):,} seconds")
    return gene_index


def read_gene_cag_dict(hdf_fp):
    return pd.read_hdf(
        hdf_fp,
        "/annot/gene/all",
        columns=["gene", "CAG"]
    ).set_index(
        "gene"
    )["CAG"].to_dict()


def read_corncob(hdf_fp):
    logging.info(f"Reading in '/stats/cag/corncob' from {hdf_fp}")
    df = pd.read_hdf(
        hdf_fp,
        "/stats/cag/corncob",
    )

    # Make sure that we have a wald metric available
    if "wald" not in df.columns.values:
        logging.info("Computing Wald metric")
        df = df.assign(
            wald=df["estimate"] / df["std_error"]
        )

    return df


def calc_mean_wald(gene_index_df, gene_cag_dict, wald_dict):

    return pd.Series({
        lg_name: lg_df["gene"].apply(
            lambda gene_name: wald_dict.get(gene_cag_dict.get(gene_name))
        ).dropna().mean()
        for lg_name, lg_df in gene_index_df.groupby("linkage_group")
    })


class DataWriter:
    
    def __init__(
        self,
        output_folder=None,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region=os.environ.get("AWS_REGION", "us-west-2"),
    ):
        assert output_folder is not None, "Please specify output_folder="
        self.output_folder = output_folder
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region = region

        # If the output is directed to S3
        if output_folder.startswith("s3://"):

            assert "/" in output_folder[5:], "Must specify prefix within S3 bucket"
            self.bucket = output_folder[5:].split("/", 1)[0]
            self.prefix = output_folder[5:].split("/", 1)[1]

            # Connect to boto3
            self.client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region,
            )
            logging.info("Connected to AWS S3 to write output")

        # If the folder is local
        else:

            self.bucket = None
            self.prefix = None
            self.client = None

    def write(self, title, dat, verbose=False):

        """
        Write out a data object.
        pd.Series: write out feather format (columns are 'index' and 'value')
        pd.DataFrame: write out feather format (index will be dropped)
        """

        assert not title.endswith("/")

        # If the data is a Series
        if isinstance(dat, pd.Series):

            # Convert to a DataFrame
            dat = pd.DataFrame(dict(
                index=dat.index.values,
                value=dat.values,
            ))
            output_format = 'feather'

        # If the data is a DataFrame
        elif isinstance(dat, pd.DataFrame):

            # It must be a DataFrame
            assert isinstance(dat, pd.DataFrame)

            # Drop the index
            dat = dat.reset_index(
                drop=True
            )
            output_format = 'feather'

        # Otherwise
        else:

            # Write out everything else as JSON
            output_format = 'json'
            
        # If the output path is on S3
        if self.bucket is not None:

            # Format the S3 prefix
            prefix = os.path.join(
                self.prefix,
                f"{title}.{output_format}"
            )

            # Open a file object in memory
            with tempfile.NamedTemporaryFile() as f:

                # Save as a file object in memory
                if output_format == 'feather':
                    dat.to_feather(f)

                elif output_format == 'json':
                    json.dump(dat, f)

                # Write the file to S3
                self.client.upload_file(
                    f.name,
                    self.bucket,
                    prefix
                )

            if verbose:
                logging.info(f"Wrote: s3://{self.bucket}/{prefix}")

        # Otherwise, write to the local filesystem
        else:

            # Format the path to write out
            fpo = os.path.join(
                self.output_folder, 
                f"{title}.{output_format}"
            )

            # Get the folder containing this file
            containing_folder = fpo.rsplit("/", 1)[0]

            # Check if the folder does not exist
            if not os.path.exists(containing_folder):

                # Make the folder if needed
                os.makedirs(containing_folder)

            if output_format == 'feather':
                dat.to_feather(fpo)

            else:
                assert output_format == 'json', f"Output format not recognized ({output_format})"

                with open(fpo, 'wt') as handle:
                    json.dump(dat, handle)

            if verbose:
                logging.info(f"Wrote: {fpo}")


def write_lg_abund(writer, lg_abund, hdf_fp):
    """Write out abundance values for linkage groups."""

    abund_manifest = []

    # Write out the abundance for each specimen
    for specimen_name, specimen_abund in lg_abund.iteritems():

        # Assign an index for this object
        ix = len(abund_manifest)

        # Annotate this object in the manifest
        abund_manifest.append({
            "type": "specimen",
            "label": specimen_name,
            "index": ix
        })

        # Write out the abundance vector
        writer.write(
            f"LG/abundance/{ix}",
            specimen_abund
        )

    logging.info(f"Wrote out {lg_abund.shape[1]:,} specimen abundances")

    # Read in the manifest
    manifest = pd.read_hdf(hdf_fp, "/manifest")
    for k in ["R1", "R2"]:
        if k in manifest.columns.values:
            manifest = manifest.drop(columns=k)
    manifest = manifest.drop_duplicates().set_index("specimen")

    # Group together specimens which have the same value
    for col_name in manifest.columns.values:
        for value, d in manifest.groupby(col_name):
            if d.shape[0] > 1:

                # Assign an index for this object
                ix = len(abund_manifest)

                # Annotate this object in the manifest
                abund_manifest.append({
                    "type": "manifest",
                    "column": col_name,
                    "value": value,
                    "index": ix
                })

                # Write out the file object
                writer.write(
                    f"LG/abundance/{ix}",
                    lg_abund.reindex(
                        columns=d.index.values
                    ).sum(
                        axis=1
                    )
                )

    # Write out the specimen manifest
    writer.write(
        "LG/abundance/index",
        abund_manifest
    )

    logging.info(
        f"Wrote out {len(abund_manifest):,} abundances of specimens grouped by metadata")


def write_wald(writer, hdf_fp, gene_index_df):
    """Write out all of the annotations available for each linkage group."""

    # Read in the table linking each gene to a CAG
    gene_cag_dict = read_gene_cag_dict(hdf_fp)

    # Keep track of the wald values written out
    wald_manifest = []

    # Summarize the linkage groups on the basis of their average Wald for each parameter
    for parameter, parameter_df in read_corncob(hdf_fp).groupby("parameter"):

        # Skip the intercept
        if parameter == "(Intercept)":
            continue

        # Get the index for this parameter
        ix = len(wald_manifest)

        # Add this to the manifest
        wald_manifest.append({
            "type": "wald",
            "parameter": parameter,
            "index": ix
        })

        # Write out the data object
        writer.write(
            f"LG/wald/{ix}",
            calc_mean_wald(
                gene_index_df,
                gene_cag_dict,
                parameter_df.set_index("CAG")["wald"].to_dict()
            )
        )
        logging.info(f"Wrote out wald summary for {parameter}")

    # Write out the wald manifest
    logging.info("Writing out wald parameter manifest")
    writer.write(
        "LG/wald/index",
        wald_manifest
    )

    logging.info(
        f"Wrote out average Wald metrics for {len(wald_manifest):,} parameters in this dataset"
    )


def pick_top_taxon(
    lg_taxa,         # Proportional taxonomic assignment of genes per subnetwork
    node_groupings,  # Subnetwork grouping of linkage groups
    lg_size,         # Number of genes in each linkage group
    tax_df,             # Taxonomy DataFrame
    tax_rank,        # Specific taxonomic rank
    min_prop=0.5,    # Minimum threshold for assignment
    min_size=20      # Size threshold of linkage group for assignment
):
    """Pick the top hit for each linkage group at a given rank"""

    # Subset down to the assignments at this rank
    rank_assignments = lg_taxa.reindex(
        columns=[
            tax_id
            for tax_id in lg_taxa.columns.values
            if tax_df.loc[tax_id, "rank"] == tax_rank
        ]
    )

    # Scale to the maximum number of genes assigned at this rank
    rank_assignments = rank_assignments.apply(
        lambda r: r / r.max() if r.max() > 0 else r,
        axis=1
    )

    # Pick the top hit per linkage group
    rank_assignments = pd.DataFrame(
        [
            {
                "linkage_group": lg_name,
                "top_hit": lg_assignments.sort_values().index.values[-1],
                "proportion": lg_assignments.sort_values().values[-1],
            }
            for subnetwork_name, lg_assignments in rank_assignments.iterrows()
            for lg_name in node_groupings[subnetwork_name]
            if lg_size[lg_name] >= min_size
        ]
    ).query(
        f"proportion >= {min_prop}"
    )

    return rank_assignments.assign(
        top_hit_name=rank_assignments["top_hit"].apply(tax_df["name"].get)
    ).set_index(
        "linkage_group"
    )[
        "top_hit_name"
    ]


def combine_overlapping_contigs(contig_gene_sets, min_overlap_prop=0.75, max_missing_n=1):
    merged_contig_sets = contig_gene_sets.copy()

    # Iterate under there are no remaining contigs which satisfy the overlap
    found_overlap = 1
    while found_overlap:
        found_overlap = 0

        # Freeze the list of contig names
        contig_names = list(merged_contig_sets.keys())

        # Make an index grouping contigs by which genes they contain
        gene_index = index_contig_gene_sets(merged_contig_sets)

        # Iterate over each contig
        start_time = time()
        for contigA in contig_names:

            # Check all of the contigs which it shares any genes with
            for contigB in list(set([
                contig_name
                for gene_name in list(merged_contig_sets[contigA])
                for contig_name in list(gene_index[gene_name])
                if contig_name != contigA
            ])):

                # Skip this contig if it's already been removed
                if merged_contig_sets.get(contigB) is None:
                    continue

                # Calculate the number and proportion of genes which overlap
                contig_size = len(merged_contig_sets[contigA])
                overlap_size = len(
                    merged_contig_sets[contigA] & merged_contig_sets[contigB])
                missing_n = contig_size - overlap_size
                overlap_prop = overlap_size / float(contig_size)

                # If either threshold is met
                if overlap_size > 0 and (overlap_prop >= min_overlap_prop or missing_n <= max_missing_n):

                    # Mark that we found an overlapping pair of contigs to merge
                    found_overlap += 1

                    # Merge the two contigs into the former contigB
                    merged_contig_sets[contigB] = merged_contig_sets[contigA] | merged_contig_sets[contigB]

                    # Remove the former contigA
                    del merged_contig_sets[contigA]

                    # Don't check contigA against any additional contigs
                    break

        logging.info(
            f"Merged {found_overlap:,} contigs -- {round(time() - start_time, 1):,} seconds")

    # Return the final set of merged contigs
    logging.info(f"Returning a set of {len(merged_contig_sets):,} contigs")
    return merged_contig_sets


def find_linkage_groups(merged_contig_sets):

    # Transform the contig sets so that we have the list of contigs which each gene is found in
    gene_index = index_contig_gene_sets(merged_contig_sets)

    # Bin each gene based on the set of contigs it is found in
    linkage_groups = defaultdict(set)

    # Iterate over each gene
    start_time = time()
    for gene_name, contig_set in gene_index.items():

        # Make a string from the set of contigs
        lg_name = " - ".join(sorted(list(contig_set)))

        # Add to the linkage group
        linkage_groups[lg_name].add(gene_name)

    logging.info(f"Found {len(linkage_groups):,} linkage groups")
    return linkage_groups


class LinkageNetwork:

    def __init__(self, summary_hdf, merged_contig_sets, max_cag_size=10000, testing=False):

        # The data we will track in this object are:

        # The genes which belong to each linkage group
        self.group_genes = defaultdict(set)
        # The CAG(s) for each linkage group
        self.group_cag = defaultdict(set)
        # The contig(s) for each linkage group, if any
        self.group_contigs = defaultdict(set)

        # As part of building the linkage groups, we need to know the CAG assignment for each gene
        cag_dict = read_cag_dict(summary_hdf)

        # We will also need to know how large each CAG is
        cag_size = cag_dict.value_counts()

        # We also need to know which linkage groups contain each CAG or contig
        self.groups_with_cag = defaultdict(set)
        self.groups_with_contig = defaultdict(set)

        # Keep track of which genes and LGs have already been assigned
        genes_in_lgs = set()
        existing_lgs = set()

        # Build the linkage groups from the merged contig sets
        for contig_list, gene_name_list in find_linkage_groups(merged_contig_sets).items():

            # If there is only a single contig
            if " - " not in contig_list:

                # And all of the genes belong to a single CAG
                if len(set([cag_dict.loc[gene_name] for gene_name in gene_name_list])) == 1:

                    # Then don't add this linkage group, because these genes
                    # will be linked on the basis of CAG, with this LG not adding
                    # any additional information
                    continue

            # Iterate over every gene which aligns to this group of contigs
            for gene_name in gene_name_list:

                # Is the CAG below the size threshold?
                if cag_size[cag_dict.loc[gene_name]] <= max_cag_size:

                    # If so, format the group name using the CAG label and the contig list
                    lg_name = f"{contig_list} :: CAG{cag_dict.loc[gene_name]}"

                # Otherwise,
                else:

                    # Format the group name using just the contig list
                    lg_name = f"{contig_list}"

                # Add the gene to this LG
                self.group_genes[lg_name].add(gene_name)

                # Record that this gene has been added
                genes_in_lgs.add(gene_name)

                # If this LG is new, record the CAG and contigs
                if lg_name not in existing_lgs:

                    # Is the CAG below the size threshold?
                    if cag_size[cag_dict.loc[gene_name]] <= max_cag_size:

                        # Then group this gene on the basis of belonging to that CAG
                        self.group_cag[lg_name] = set(
                            [cag_dict.loc[gene_name]])

                    # if the CAG is above the size threshold
                    else:

                        # Then ignore the CAG assignment, but do fill in a dummy value (the linkage group name)
                        self.group_cag[lg_name] = set([contig_list])

                    # Record the contigs that this linkage group belongs to
                    self.group_contigs[lg_name] = set(contig_list.split(" - "))

                    existing_lgs.add(lg_name)

                    # Index the linkage groups which are joined by CAG
                    self.groups_with_cag[
                        list(self.group_cag[lg_name])[0]
                    ].add(lg_name)

                    # And by contig
                    for contig_name in contig_list.split(" - "):
                        self.groups_with_contig[contig_name].add(lg_name)

        logging.info(
            f"Added genes from contigs to create a network with {len(self.group_genes):,} linkage groups and {sum(map(len, self.group_genes.values())):,} genes")

        # Now we need to add LGs which contain genes that do not align to any contigs
        i = 0
        for gene_name, cag_id in cag_dict.items():

            # If the gene has not yet been grouped
            if gene_name not in genes_in_lgs:

                # Is the CAG above the size threshold?
                if cag_size[cag_dict.loc[gene_name]] > max_cag_size:

                    # Then skip it
                    continue

                # Otherwise

                # Set up the name of the linkage group as the name of the CAG
                lg_name = f"CAG{cag_id}"

                # Add the gene to this LG
                self.group_genes[lg_name].add(gene_name)

                # If this LG is new, record the CAG and contigs
                if lg_name not in existing_lgs:

                    # Record the CAG
                    self.group_cag[lg_name] = set([cag_dict.loc[gene_name]])

                    # There is no contig to record for this LG

                    existing_lgs.add(lg_name)

                    self.groups_with_cag[cag_dict.loc[gene_name]].add(lg_name)

                    # Increment the counter
                    i += 1
                    if testing and i == 100:
                        logging.info("TESTING -- STOPPING EARLY")
                        break

        logging.info(
            f"Created a network with {len(self.group_genes):,} linkage groups and {sum(map(len, self.group_genes.values())):,} genes")

        # Cache the group size and number of connections
        self.nconns_cache = dict()
        self.ngenes_cache = dict()

        # Useful Functions

    def group_list(self):
        return list(self.group_genes.keys())

    def group_sizes(self):
        return pd.Series({
            lg_name: len(gene_list)
            for lg_name, gene_list in self.group_genes.items()
        })

    def ngenes(self, lg_name):
        if self.ngenes_cache.get(lg_name) is None:
            self.ngenes_cache[lg_name] = len(self.group_genes.get(lg_name))
        return self.ngenes_cache[lg_name]

    def nconn(self, lg_name):
        if self.nconns_cache.get(lg_name) is None:
            self.nconns_cache[lg_name] = len(self.connections(lg_name))
        return self.nconns_cache[lg_name]

    def connections(self, lg_name):
        return list(set([
            other_lg
            for cag_id in self.group_cag[lg_name]
            for other_lg in self.groups_with_cag[cag_id]
            if other_lg != lg_name
        ] + [
            other_lg
            for contig_name in self.group_contigs.get(lg_name, [])
            for other_lg in self.groups_with_contig[contig_name]
            if other_lg != lg_name
        ]))

    def remove(self, lg_to_remove):

        # Invalidate the nconn cache for all LGs that this is connected to
        for other_lg in self.connections(lg_to_remove):
            if self.nconns_cache.get(other_lg) is not None:

                # Increment the value downwards by 1
                new_value = self.nconns_cache[other_lg] - 1
                if new_value < 0:
                    del self.nconns_cache[other_lg]
                else:
                    self.nconns_cache[other_lg] = new_value

        # Remove this contig from `groups_with_cag`
        for cag_id in self.group_cag[lg_to_remove]:
            if lg_to_remove in self.groups_with_cag[cag_id]:
                self.groups_with_cag[cag_id].remove(lg_to_remove)

        # For each contig that this LG belongs to
        for contig_name in self.group_contigs.get(lg_to_remove, []):
            # Remove this LG from that contig's entry in `groups_with_contig`
            if lg_to_remove in self.groups_with_contig[contig_name]:
                self.groups_with_contig[contig_name].remove(lg_to_remove)

        # Remove the entries listing this LG
        del self.group_genes[lg_to_remove]
        del self.group_cag[lg_to_remove]
        if self.group_contigs.get(lg_to_remove) is not None:
            del self.group_contigs[lg_to_remove]

        # Remove `lg_to_remove` from both the nconn and ngenes cache
        if self.nconns_cache.get(lg_to_remove) is not None:
            del self.nconns_cache[lg_to_remove]
        if self.ngenes_cache.get(lg_to_remove) is not None:
            del self.ngenes_cache[lg_to_remove]

    def merge(self, lg_to_remove, lg_to_join):
        """
        Combine the contents (genes) of one linkage group with another.
        """

        assert lg_to_remove != lg_to_join, f"Cannot join LG to itself ({lg_to_remove})"

        # Add the genes from `lg_to_remove` to the set for `lg_to_join`
        self.group_genes[lg_to_join] = self.group_genes.get(
            lg_to_join, set([])) | self.group_genes[lg_to_remove]

        # Remove the remaining references to `lg_to_remove`
        self.remove(lg_to_remove)

        # Update the ngenes cache for lg_to_join
        self.ngenes_cache[lg_to_join] = len(self.group_genes[lg_to_join])


def group_summary_table(LN):
    """Summarize the size and connectivity of all nodes in the network."""
    plot_df = pd.DataFrame(dict(
        ngenes=LN.group_sizes()
    ))
    return plot_df.assign(
        nconn=pd.Series(
            [
                LN.nconn(n) + 1
                for n in plot_df.index.values
            ],
            index=plot_df.index.values
        )
    ).assign(
        connection_size_ratio=lambda d: d["ngenes"] / d["nconn"],
    ).assign(
        connection_size_ratio_log10=lambda d: d["connection_size_ratio"].apply(
            np.log10)
    )


def merge_nodes_by_ratio(
    LN,
    max_ratio=0,  # Merge all nodes at or below this threshold
):

    """
    Merge nodes with less information content (fewer genes, more connections) into
    more information-dense neighbors.
    """

    logging.info(
        f"Merging nodes which fall under the connectivity ratio of 10^{max_ratio}")

    # Make a table with the number of genes, and the number of connections, for all groups
    summary_df = group_summary_table(LN)

    # Filter down to the gene groups which fall into this ratio
    summary_df = summary_df.query(
        f"connection_size_ratio_log10 <= {max_ratio}"
    )
    logging.info(f"Screening a batch of {summary_df.shape[0]:,} groups")

    # Keep track of how many groups were trimmed
    ntrimmed = 0

    # Iterate over every group in this preliminary list
    for group_name, group_stats in summary_df.sort_values(
        by='connection_size_ratio_log10',
        ascending=True,
    ).iterrows():

        # Skip any groups which were above the threshold at the beginning of the loop
        if group_stats["connection_size_ratio_log10"] > max_ratio:
            continue

        # Skip any groups which are not connected
        if LN.nconn(group_name) == 0:
            continue

        # Double check the connection score
        group_score = np.log10(
            LN.ngenes(group_name) / (LN.nconn(group_name) + 1)
        )

        # If the score is above the threshold
        if group_score > max_ratio:

            # Skip it
            continue

        # Get the list of groups which are connected to it
        connected_groups = LN.connections(group_name)
        assert len(connected_groups) > 0, group_name

        # Implicitly, this node meets the specified requirements

        # Find the connection score(s) of the node(s) that this is connected to
        connected_group_scores = pd.Series({
            connected_group_name: LN.ngenes(
                connected_group_name) / (LN.nconn(connected_group_name) + 1)
            for connected_group_name in connected_groups
        })

        # Merge this group into the highest scoring one it is connected to
        adjacent_group_name = connected_group_scores.sort_values(
            ascending=False
        ).index.values[0]

        # This action will also remove the node
        LN.merge(
            group_name,
            adjacent_group_name,
        )

        # Either way, increment the counter
        ntrimmed += 1

        if ntrimmed % 10000 == 0:
            logging.info(f"Trimmed {ntrimmed:,} groups")

    # Get the vector of group sizes
    vc = LN.group_sizes()
    logging.info(
        f"Trimmed {ntrimmed:,} groups, resulting in {vc.shape[0]:,} groups for {vc.sum():,} genes"
    )


def merge_terminal_nodes(
    LN,
    max_size=10,  # Merge all terminal nodes which container fewer than this number of genes
):
    """Merge nodes which are terminal (have only 1 non-self connection)."""

    logging.info(f"Merging terminal nodes smaller than {max_size}")

    # Keep track of how many groups were trimmed
    ntrimmed = 1

    # Iterate until they are all merged
    while ntrimmed > 0:

        # Reset the counter
        ntrimmed = 0

        # Make a table with the number of genes, and the number of connections, for all groups
        summary_df = group_summary_table(LN)

        # Iterate over every group which meets this threshold
        for group_name in summary_df.query(
            "nconn == 2"
        ).query(
            f"ngenes < {max_size}"
        ).sort_values(
            by="ngenes"
        ).index.values:

            # Get the list of groups which are connected to it
            connected_groups = LN.connections(group_name)

            # If there is more (or less) than one connected group
            if len(connected_groups) != 1:

                # Skip it
                pass

            # If the group size meets the threshold
            elif LN.ngenes(group_name) >= max_size:

                # Skip it
                pass

            # If both thresholds are met
            else:

                # Merge the node with its neighbor
                LN.merge(
                    group_name,
                    connected_groups[0]
                )

                # Increment the counter
                ntrimmed += 1

        # Get the vector of group sizes
        vc = LN.group_sizes()
        logging.info(
            f"Merged {ntrimmed:,} groups, resulting in {vc.shape[0]:,} groups for {vc.sum():,} genes"
        )

def remove_unconnected_nodes(
    LN,
    max_size_unconnected=5,
):
    """Remove all unconnected nodes which container fewer than this number of genes."""

    logging.info(f"Removing unconnected nodes smaller than {max_size_unconnected}")

    # Make a table with the number of genes, and the number of connections, for all groups
    summary_df = group_summary_table(LN)

    # Keep track of how many groups were trimmed
    ntrimmed = 0

    # Iterate over every group which meets this threshold
    for group_name in summary_df.query(
        "nconn == 1"
    ).query(
        f"ngenes < {max_size_unconnected}"
    ).index.values:

        # Remove the node
        LN.remove(group_name)

        # Increment the counter
        ntrimmed += 1

    # Get the vector of group sizes
    vc = LN.group_sizes()
    logging.info(
        f"Trimmed {ntrimmed:,} groups, resulting in {vc.shape[0]:,} groups for {vc.sum():,} genes")


def rename_nodes(LN):
    """Rename nodes with integers counting up from 0."""

    logging.info("Renaming linkage groups with ordinal numbers")

    mapping = {
        old_name: new_name
        for new_name, old_name in enumerate(LN.group_sizes().index.values)
    }

    logging.info(f"Found {len(mapping):,} linkage groups to rename")

    # Function will use the mapping to rename the keys of a dict
    rename_dict_keys = lambda d: {mapping[k]: v for k, v in d.items()}

    # Rename each of the componant elements of the LinkageNetwork
    LN.group_genes = rename_dict_keys(LN.group_genes)
    LN.group_cag = rename_dict_keys(LN.group_cag)
    LN.group_contigs = rename_dict_keys(LN.group_contigs)

    # Function will use the mapping to rename the elements of a set, 
    # which are the values of a dict
    rename_dict_set = lambda d: {k: set([mapping[i] for i in v]) for k, v in d.items()}
    LN.groups_with_contig = rename_dict_set(LN.groups_with_contig)
    LN.groups_with_cag = rename_dict_set(LN.groups_with_cag)

    # Make sure that all nodes have been renamed
    new_name_set = set(list(mapping.values()))

    # Check the keys of each of the dictionaries
    for d in [LN.group_genes, LN.group_cag, LN.group_contigs]:
        for k in d.keys():
            assert k in new_name_set, k

    # Check the members of each of the sets
    for d in [LN.groups_with_cag, LN.groups_with_contig]:
        for s in d.values():
            for k in s:
                assert k in new_name_set, k

    logging.info("All linkage groups have been renamed")

def read_genome_gene_sets(summary_hdf):

    # Iterate over every genome in the HDF
    static_genome_key_list = genome_key_list(summary_hdf)
    logging.info(
        f"Preparing to read data for {len(static_genome_key_list):,} genomes")

    # Format the output as a dict of sets
    output = dict()

    # Open a connection to the HDF
    with pd.HDFStore(summary_hdf, "r") as store:

        # Iterate over every key with alignment details
        for key in static_genome_key_list:

            # Parse the genome ID from the key
            genome_id = key.split("/")[-1]
            assert len(genome_id) > 0

            # Add the genes for this genome
            output[genome_id] = set(pd.read_hdf(store, key)["gene"].tolist())

    logging.info(f"Read in data for {len(output):,} genomes")
    return output


def count_genome_overlap(LN, genome_gene_sets):
    """For each genome, count the number of linkage groups it overlaps with."""

    # Format the output as a list
    # Each item will be a dict, and it will be formatted as a DataFrame at the end
    output = []

    # Iterate over each genome
    for genome_id, gene_set in genome_gene_sets.items():

        # Iterate over every linkage group
        for group_name, group_set in LN.group_genes.items():

            # Count the number of genes which overlap
            n = len(gene_set & group_set)

            # If there is no overlap
            if n == 0:

                # Skip it
                continue

            # Add data to the output
            prop_genome = n / len(gene_set)
            prop_lg = n / len(group_set)
            containment = max(prop_genome, prop_lg)
            output.append(dict(
                genome=genome_id,
                lg=group_name,
                n=n,
                prop_genome=prop_genome,
                prop_lg=prop_lg,
                containment=containment,
            ))

    return pd.DataFrame(output)


def make_network(
    LN
):

    logging.info(f"Building a network with {len(LN.group_genes):,} linkage groups")

    # Set up the graph
    G = nx.Graph()

    # Add all of this information to the network

    # First, add the nodes
    G.add_nodes_from([
        (lg_name, {"type": "metagenome", "size": len(gene_set)})
        for lg_name, gene_set in LN.group_genes.items()
    ])

    # Now add the links between linkage groups
    G.add_edges_from([
        (group_a, group_b)
        for group_a in LN.group_genes.keys()
        for group_b in LN.connections(group_a)
        if group_a < group_b
    ])
    logging.info("Done")

    return G


def add_genomes_to_graph(
    LN,
    G,
    genome_overlap_df,
    min_genome_containment=0.5
):

    # Filter the genome data by containment
    filtered_genome_df = genome_overlap_df.query(
        f"containment > {min_genome_containment}"
    )
    n_genomes = filtered_genome_df["genome"].unique().shape[0]

    logging.info(f"Adding {n_genomes:,} genomes to the graph")

    # Add the genomes
    G.add_nodes_from([
        (genome_id, {"type": "genome"})
        for genome_id in filtered_genome_df["genome"].unique()
    ])

    # Now add the links to genomes
    G.add_edges_from([
        (
            r["genome"],
            r["lg"],
            {
                k: r[k]
                for k in [
                    "prop_genome",
                    "containment",
                    "prop_lg",
                    "n"
                ]
            }
        )
        for _, r in filtered_genome_df.iterrows()
    ])
    logging.info("Done")


class Taxonomy:

    def __init__(self, summary_hdf):
        """Read the taxonomy structure."""

        # Save the taxonomy
        self.tax = pd.read_hdf(summary_hdf, "/ref/taxonomy").apply(
            lambda c: c.apply(lambda v: 'none' if pd.isnull(v) else str(
                int(float(v)))) if c.name in ['parent', 'tax_id'] else c
        ).reset_index(
            drop=True
        ).set_index(
            "tax_id"
        )

    @lru_cache(maxsize=None)
    def path_to_root(self, tax_id):
        # Format as a string
        tax_id = str(tax_id)

        # Keep track of the taxa we've visited thus far
        visited = set([])

        while tax_id not in visited and tax_id in self.tax.index.values:

            # Add this tax ID to the set of visited
            visited.add(tax_id)

            # Get the parent taxon
            tax_id = self.tax.loc[
                tax_id,
                "parent"
            ]

            # If this is null
            if pd.isnull(tax_id):
                # Stop
                break

        return list(visited)

def read_taxonomy_df(summary_hdf):
    """Read in the taxonomic assignment for all genes."""

    with pd.HDFStore(summary_hdf, "r") as store:
        return pd.read_hdf(
            store,
            "/annot/gene/all",
            columns=["gene", "tax_id", "tax_name"]
        ).set_index(
            "gene"
        ).apply(
            lambda c: c.apply(float).apply(int).apply(str) if c.name == "tax_id" else c
        )


def count_tax_assignments(LN, taxonomy_df, tax):
    """Count up the number of genes assigned to each taxon."""

    # The final DataFrame will have the columns linkage_group, tax_id, parent, name, count

    # Build the DataFrame from a list
    output = []

    # Iterate over each linkage group
    for linkage_group, gene_name_set in LN.group_genes.items():

        # Get the list of tax IDs, and get the total counts for each
        tax_counts = pd.Series([
            taxonomy_df.loc[gene_name, "tax_id"]
            for gene_name in gene_name_set
        ]).value_counts()

        # Add this set of counts to the DataFrame
        output.extend([
            {
                'linkage_group': linkage_group,
                'tax_id': tax_id,
                'count': count
            }
            for tax_id, count in tax_counts.items()
        ])

    # Format as a DataFrame
    output = pd.DataFrame(output)

    # Add the parent and name for each
    return output.assign(
        parent = output.tax_id.apply(tax.tax["parent"].get),
        name = output.tax_id.apply(tax.tax["name"].get),
    )
    

def make_tax_spectrum_df(LN, G, taxonomy_df, tax):
    """Make a DataFrame with the taxonomy spectrum for every connected group."""

    # Make a DataFrame with the taxonomy spectrum for every connected group
    tax_spectrum_df = {}

    # Make a Series with the number of genes in each connected group
    node_size = {}

    # Keep track of the group that each node belongs to
    node_groupings = {}

    # Iterate over each of the connected groups in the graph
    for node_set in nx.connected_components(G):

        # Link the taxon spectrum to the largest group of genes in this set of nodes
        largest_name = largest_node(node_set, LN)
        node_groupings[largest_name] = list(node_set)
        node_size[largest_name] = len(node_set)

        # Get the spectrum of taxonomic assignments for this set of nodes
        tax_spectrum_df[largest_name] = get_tax_spectra(
            node_set,
            LN,
            taxonomy_df,
            tax,
        )

        if len(tax_spectrum_df) % 2000 == 0:
            logging.info(f"Found taxa for {len(tax_spectrum_df)} connected groups")

    # Format as a DataFrame
    node_size = pd.Series(node_size)
    tax_spectrum_df = pd.DataFrame(tax_spectrum_df).fillna(0).T
    assert tax_spectrum_df.shape[0] > 0
    assert tax_spectrum_df.shape[1] > 0
    assert node_size.shape[0] == tax_spectrum_df.shape[0]
    logging.info(
        f"Found {tax_spectrum_df.shape[1]:,} taxa for {tax_spectrum_df.shape[0]:,} connected groups"
    )

    return tax_spectrum_df, node_groupings


def get_tax_spectra(node_set, LN, taxonomy_df, tax):

    # Keep a running total of each tax ID and the number of genes
    # which were assigned to it
    tax_counts = defaultdict(int)

    # Keep a running total of the number of genes in the spectrum
    ngenes = 0

    # Iterate over every linkage group
    for group_name in list(node_set):
        # Iterate over every gene in this linkage group
        for gene_name in list(LN.group_genes.get(group_name, {})):
            # Get the tax ID for this gene
            assigned_tax_id = taxonomy_df.loc[gene_name, "tax_id"]

            # Iterate over the path to root
            for tax_id in tax.path_to_root(assigned_tax_id):

                # Increment the counter
                tax_counts[tax_id] += 1

            # Increment the gene counter
            ngenes += 1

    tax_counts = pd.Series(tax_counts, dtype='float64')

    # Divide by the total number of genes to get the spectrum
    if ngenes > 0:
        tax_counts = tax_counts / ngenes

    return tax_counts


def largest_node(node_set, LN):
    """Find the node which contains the most genes in a set of nodes."""
    max_size = None
    max_label = None

    for group_name in node_set:
        group_size = len(LN.group_genes.get(group_name, {}))
        if group_size > 0:

            if max_size is None or group_size > max_size:
                max_size = group_size
                max_label = group_name

    return max_label


def name_tax_spectra(tax_spectra, tax):
    df = pd.DataFrame(dict(
        prop=tax_spectra.loc[tax_spectra > 0]
    ))
    df = df.assign(
        name=tax.tax["name"],
        rank=tax.tax["rank"],
    ).sort_values(
        by="prop",
        ascending=False
    ).groupby(
        "rank"
    ).head(
        1
    ).set_index(
        "rank"
    ).reindex(
        index=["species", "genus", "family", "order", "class", "phylum"]
    ).dropna(
    ).query(
        "prop > 0.25"
    )
    if df.shape[0] > 0:
        return df["name"].values[0]
    else:
        return "Unclassified"


def get_linkage_group_abundances(LN, detail_hdf):
    """Get the CAG-level abundances for each linkage group."""

    # Make a Series assigning each gene to its linkage group
    gene_group_assignment = pd.Series({
        gene_name: group_name
        for group_name, gene_name_list in LN.group_genes.items()
        for gene_name in gene_name_list
    })

    # Make a dict, keyed by specimen, with the abundances of each linkage group
    abund_dict = {}

    # Open the HDF5 file
    with pd.HDFStore(detail_hdf, "r") as store:

        # Iterate over each table in the store
        for table_name in store:

            # Only consider the tables with abundance information
            if not table_name.startswith("/abund/gene/long/"):
                continue

            # Parse the name of the specimen
            specimen_name = table_name.replace("/abund/gene/long/", "")

            # Read in the observed abundances for this specimen
            specimen_df = pd.read_hdf(store, table_name)

            # Compute the relative abundance based on the depth of sequencing per gene
            specimen_abund = specimen_df.set_index(
                "id")["depth"] / specimen_df["depth"].sum()

            # Calculate the aggregate relative abundance per linkage group
            abund_dict[
                specimen_name
            ] = pd.DataFrame(  # Make a DataFrame
                dict(  # With the linkage group and abundance of all genes
                    linkage_group=gene_group_assignment,
                    abund=specimen_abund
                )
            ).dropna(  # Drop any gene which is missing either
                # Group together all of the genes from the same linkage group
            ).groupby(
                "linkage_group"
            ).sum()["abund"]  # Calculate the aggregate abundance per group

            if len(abund_dict) % 10 == 0:
                logging.info(
                    f"Computed relative abundance of linkage groups across {len(abund_dict):,} specimens")

    logging.info(
        f"Done computing relative abundance of linkage groups across {len(abund_dict):,} specimens"
    )

    # Make a DataFrame
    return pd.DataFrame(abund_dict).fillna(0)


def write_edges(writer, G, output_folder):
    """Reformat the edges from the graph object as a table in feather format."""
    
    # Make a table of edges
    edge_df = pd.DataFrame([
        {
            "from": edge[0],
            "to": edge[1]
        }
        for edge in G.edges
    ])

    # Save to the specified file
    writer.write(
        "LG/edges",
        edge_df
    )

def expand_subnetworks(node_groupings, G, coords, q=0.25):
    """Given a set of coordinates, expand to include the members of each subnetwork."""

    # Find the first quartile (if q=0.25) distance to the nearest neighbor for each
    median_nearest_neighbor = np.quantile(
        [
            np.delete(r, i).min()
            for i, r in enumerate(
                squareform(
                    pdist(
                        coords,
                        metric="euclidean"
                    )
                )
            )
        ],
        q
    )


    # Read in the graph structure of the subnetworks
    logging.info(f"Processing network layout for {len(G.nodes):,} linkage groups")
    logging.info(f"Read in {sum(map(len, node_groupings.values())):,} linkage groups in {len(node_groupings):,} subnetworks")

    # Expand each subnetwork using that value as the maximum size in either dimension
    return pd.concat(
        [
            expand_single_subnetwork(
                lg_name_list,
                G,
                coords.loc[n],
                median_nearest_neighbor,
            )
            for n, lg_name_list in node_groupings.items()
            if n in coords.index.values
        ]
    )


def expand_single_subnetwork(lg_name_list, G, tsne_coords, final_size):
    # Get the coordinate column names
    col_names = tsne_coords.index.values

    # If there is just a single linkage group in this subnetwork
    if len(lg_name_list) == 1:
        return pd.DataFrame([{
            "linkage_group": list(lg_name_list)[0],
            col_names[0]: tsne_coords[0],
            col_names[1]: tsne_coords[1],
        }])

    # Otherwise, this subnetwork contains multiple linkage groups

    # Get the subnetwork containing these genes
    H = G.subgraph(
        n for n in G.nodes
        if n in lg_name_list
    )

    # Get the positions for these nodes
    pos = pd.DataFrame(
        nx.spring_layout(H),
        index=col_names
    ).T

    # Scale both axes to range from -`final_size` to +`final_size`
    pos = pos - pos.min()
    pos = (2 * final_size * pos / pos.max()) - final_size

    # Now move the entire frame of reference to the indicated coordinates
    pos = pos + tsne_coords

    # Make the column names conform
    return pos.reset_index(
    ).rename(
        columns={
            "index": "linkage_group",
        }
    )


def subnetwork_size(gene_index_df, node_groupings):
    # Get the number of genes per linkage group
    lg_size = gene_index_df[
        "linkage_group"
    ].value_counts()

    # Make sure that there is a size for all linkage groups
    n_missing = np.sum(list(map(
        lambda lg_name: int(lg_size.get(lg_name) is None),
        [
            lg_name
            for lg_name_list in node_groupings.values()
            for lg_name in lg_name_list
        ]
    )))

    assert n_missing == 0, f"Missing sizes for {n_missing:,} linkage groups"

    # Return the sum of the sizes of the linkage groups in each subnetwork
    return {
        n: np.sum(list(map(lg_size.get, lg_name_list)))
        for n, lg_name_list in node_groupings.items()
    }


def taxonomic_linkage_clustering(
    tax_spectra_df,
    method="complete",
    metric="euclidean",
):

    # Perform linkage clustering
    Z = linkage(
        tax_spectra_df,
        method=method,
        metric=metric,
    )

    # Compute the optimal leaf ordering
    Z = optimal_leaf_ordering(
        Z,
        tax_spectra_df,
        metric
    )

    # Return the linkage groups, as well as the labels (which should all be integers)
    return Z, list(map(int, tax_spectra_df.index.values))


class Partition:

    def __init__(
        self, 
        name,
        size,
        theta_min=0, 
        theta_max=2*np.pi, 
        radius=0, 
        is_leaf=False
    ):
        self.name = name
        self.size = size
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.radius = radius
        self.is_leaf = is_leaf
        
    def show(self):
        print(json.dumps({
            k: str(self.__getattribute__(k))
            for k in dir(self)
            if not (k.startswith("__") or callable(self.__getattribute__(k)))
        }, indent=4))
        
    def split(self, childA, childB, radius_delta):
        """Split a node, weighted by child size."""
        # Make sure that the child size adds up to the node size
        assert childA['size'] + childB['size'] == self.size, (childA['size'], childB['size'], self.size)
        assert childA['size'] > 0
        assert childB['size'] > 0
        
        # Get the arc length of the parent
        theta_range = self.theta_max - self.theta_min
        
        # Compute the proportion of that arc which is assigned to each child
        thetaA = theta_range * childA['size'] / self.size
        thetaB = theta_range * childB['size'] / self.size
        
        nodeA = Partition(
            childA['name'],
            childA['size'],
            theta_min = self.theta_min,
            theta_max = self.theta_min + thetaA,
            radius = self.radius + radius_delta,
            is_leaf = childA['is_leaf']
        )

        nodeB = Partition(
            childB['name'],
            childB['size'],
            theta_min = self.theta_max - thetaB,
            theta_max = self.theta_max,
            radius = self.radius + radius_delta,
            is_leaf = childB['is_leaf']
        )

        return nodeA, nodeB
    
    def pos(self):
        """Calculate the position of the node in cartesian coordinates."""
        theta = np.mean([self.theta_min, self.theta_max])
        
        return {
            "name": self.name,
            "x": self.radius * math.cos(theta),
            "y": self.radius * math.sin(theta),
        }
    
        
class PartitionMap:
    
    def __init__(self, Z, leaf_names, size_dict):
        """Input is a linkage matrix, and a list of the names of all leaves"""
        
        # Save the dictionary with the number of genes in each subnetwork
        self.size_dict = size_dict

        # Make sure that there is a size value for all listed leaves
        n_missing = np.sum(list(map(lambda n: size_dict.get(n) is None, leaf_names)))
        assert n_missing == 0, f"Missing {n_missing:,}/{len(leaf_names):,} node sizes"

        logging.info(f"Number of leaves: {len(leaf_names):,}")
        
        # Make a DataFrame with the nodes which were joined at each iteration
        Z = pd.DataFrame(
            Z,
            columns = ["childA", "childB", "distance", "size"]
        )

        # Save the number of leaves as a shortcut
        self.nleaves = len(leaf_names)

        # Number internal nodes starting from `nleaves`
        Z = Z.assign(
            node_ix = list(map(
                lambda i: i + self.nleaves,
                np.arange(Z.shape[0])
            ))
        ).apply(
            lambda c: c.apply(int if c.name != 'distance' else float)
        ).set_index(
            "node_ix"
        )

        # Function to get the size of a node / leaf
        # based on the indexing system used by `linkage`
        self.node_size = lambda i: Z.loc[int(i), 'size'] if i >= self.nleaves else size_dict[leaf_names[int(i)]]
        
        # Use the number of genes per subnetwork to update the size of each node
        for node_ix, r in Z.iterrows():

            # Update the table
            try:
                Z.loc[node_ix, 'size'] = self.node_size(r.childA) + self.node_size(r.childB)
            except:
                logging.info(f"Problem updating size: {node_ix} / {r.childA} / {r.childB}")
                logging.info(f"Unexpected error: {sys.exc_info()[0]}")
                raise

        # Reverse the order
        Z = Z[::-1]
        
        # Create the map, assigning the entire range to the final node which was created
        root_node = Z.index.values[0]
        logging.info(f"Root node {root_node} contains {self.node_size(root_node):,} genes")
        self.partition_map = {
            root_node: Partition(root_node, self.node_size(root_node))
        }

        # Save the entire linkage matrix
        self.Z = Z
        
        # Save the leaf names
        self.leaf_names = leaf_names
        
        # Expand each node in turn
        for parent_node_ix, r in Z.iterrows():
            self.add_children(parent_node_ix, r)
            
    def annotate(self, node_ix):
        
        return {
            "size": self.node_size(node_ix),
            "name": node_ix,
            "is_leaf": node_ix < self.nleaves
        }
            
        
    def add_children(self, parent_node_ix, r):
        
        # Make sure that the parent node has a partition assigned
        assert parent_node_ix in self.partition_map
        
        # Check if both children are leaves
        # If they are not leaves, then their size is encoded in the linkage matrix
        childA = self.annotate(r.childA)
        childB = self.annotate(r.childB)
        
        # Split the parent node, weighted by size
        nodeA, nodeB = self.partition_map[
            parent_node_ix
        ].split(
            childA,
            childB,
            r.distance
        )
        
        self.partition_map[childA['name']] = nodeA
        self.partition_map[childB['name']] = nodeB
        
    def get_coords(self):
        """Return the x-y coordinates of all leaf nodes."""
        
        return pd.DataFrame(
            [
                node.pos()
                for node_name, node in self.partition_map.items()
                if node.is_leaf
            ]
        ).set_index(
            "name"
        )


def linkage_partition(
    writer, 
    tax_spectrum_df,
    gene_index_df,
    node_groupings,
    G,
    method="complete",
    metric="euclidean",
):

    # Make sure that each index of the taxonomic spectra
    # corresponds directly to a grouping of nodes

    
    # Get the linkage clustering based on taxonomic assignments
    logging.info("Performing linkage clustering")
    Z, Z_index = taxonomic_linkage_clustering(
        tax_spectrum_df,
        method=method,
        metric=metric
    )
    
    # Get the sizes of each subnetwork
    size_dict = subnetwork_size(
        gene_index_df,
        node_groupings,
    )

    # Use that linkage matrix to build a set of coordinates
    logging.info("Mapping linkage clusters to x-y coordinates")
    pm = PartitionMap(Z, Z_index, size_dict)
    coords = pm.get_coords()
    
    # Replace each subnetwork with its members
    logging.info("Expanding subnetworks")
    df = expand_subnetworks(
        node_groupings, 
        G,
        coords,
    ).reset_index(
        drop=True
    )
    
    # Save in feather format
    logging.info("Done formatting network display")
    writer.write(
        "LG/coords",
        df,
        verbose=True
    )

if __name__ == "__main__":

    log_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [GLAM Metagenome Map] %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Write logs to STDOUT
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(
        description="""
        Build a metagenome map for visualization in the GLAM Browser.

        Example Usage:

        metagenome_map.py \ 
            --summary-hdf <SUMMARY_HDF> \ 
            --detail-hdf <DETAIL_HDF> \ 
            --output-folder <OUTPUT_FOLDER>

        """
    )

    parser.add_argument(
        "--summary-hdf",
        type=str,
        help="Path to results HDF5 file generated by geneshot"
    )

    parser.add_argument(
        "--detail-hdf",
        type=str,
        help="Path to detailed results from geneshot containing assembly information"
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        help="Folder to write output files"
    )

    parser.add_argument(
        "--testing",
        action="store_true",
        help="If specified, use a random subset of the data for testing purposes"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Make sure the input files exist
    for fp in [args.summary_hdf, args.detail_hdf]:
        assert os.path.exists(fp), f"Cannot find {fp}"

    glam_network(
        args.summary_hdf,
        args.detail_hdf,
        args.output_folder,
        method='complete',
        metric='euclidean',
        testing=args.testing,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region=os.environ.get("AWS_REGION", "us-west-2"),
        min_node_size=10
    )
