#!/usr/bin/env python3

import argparse
from functools import lru_cache
import logging
import os
import pandas as pd


def read_gene_cag_dict(hdf_fp):
    return pd.read_hdf(
        hdf_fp,
        "/annot/gene/all",
        columns=["gene", "CAG"]
    ).set_index(
        "gene"
    )["CAG"].to_dict()


def read_corncob(hdf_fp):
    return pd.read_hdf(
        hdf_fp,
        "/stats/cag/corncob",
    )


def calc_mean_wald(lg_membership, gene_cag_dict, wald_dict):

    return pd.Series({
        lg_name: lg_df["gene"].apply(
            lambda gene_name: wald_dict.get(gene_cag_dict.get(gene_name))
        ).dropna().mean()
        for lg_name, lg_df in lg_membership.groupby("linkage_group")
    })


def write_out(folder, title, dat, verbose=False):
    """Write out a single vector in feather format (columns are 'index' and 'value')."""

    fpo = os.path.join(folder, f"{title}.feather")

    if os.path.exists(fpo):
        if verbose:
            logging.info(f"Exists: {fpo}")
        return

    # Make sure the containing folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Write out to the file
    pd.DataFrame(dict(
        index=dat.index.values,
        value=dat.values,
    )).to_feather(
        fpo
    )
    if verbose:
        logging.info(f"Wrote: {fpo}")


def annotate_linkage_groups(input_prefix, hdf_fp):
    """Write out all of the annotations available for each linkage group."""

    # Skip taxonomic assignments if they are already done
    if all([
        os.path.exists(f"{input_prefix}/taxonomic/{tax_rank}.feather")
        for tax_rank in ["phylum", "class", "order", "family", "genus", "species"]
    ]):
        logging.info("All taxonomic labels have been computed already")
    else:
        logging.info("Computing taxonomic labels")

        # Read in the table with the taxonomic assignments for genes in each linkage group
        lg_taxa = pd.read_feather(
            f"{input_prefix}.taxSpectrum.feather"
        ).set_index("index")

        # Read in the table with the names and ranks for each taxon
        tax_df = pd.read_hdf(hdf_fp, "/ref/taxonomy").set_index("tax_id")

        # For each taxonomic level, write out the best hit for each linkage group
        for tax_rank in ["phylum", "class", "order", "family", "genus", "species"]:
            if os.path.exists(
                f"{input_prefix}/taxonomic/{tax_rank}.feather"
            ):

                logging.info(f"Output already exists for {tax_rank} level summaries, skipping")

            else:

                write_out(
                    f"{input_prefix}/taxonomic/",
                    tax_rank,
                    pick_top_taxon(lg_taxa, tax_df, tax_rank)
                )

                logging.info(f"Wrote out taxonomic assignments at the {tax_rank} level")

    # Read in the table listing which genes are grouped into which linkage groups
    lg_membership = pd.read_feather(
        f"{input_prefix}.linkage-group-gene-index.feather"
    )

    # Read in the table listing the relative abundance of each linkage group in each specimen
    lg_abund = pd.read_feather(
        f"{input_prefix}.linkage-group-abundance.feather"
    ).set_index(
        "index"
    )

    # Write out the abundance for each specimen
    for specimen_name, specimen_abund in lg_abund.iteritems():
        write_out(
            f"{input_prefix}/abundance/specimen/",
            specimen_name, 
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
    i = 0
    for col_name in manifest.columns.values:
        for value, d in manifest.groupby(col_name):
            if d.shape[0] > 1:
                write_out(
                    f"{input_prefix}/abundance/manifest/{col_name}",
                    value,
                    lg_abund.reindex(
                        columns=d.index.values
                    ).sum(
                        axis=1
                    )
                )
                i += 1
    logging.info(f"Wrote out {i:,} abundances of specimens grouped by metadata")

    # Read in the table linking each gene to a CAG
    gene_cag_dict = read_gene_cag_dict(hdf_fp)

    # Summarize the linkage groups on the basis of their average Wald for each parameter
    i = 0
    for parameter, parameter_df in read_corncob(hdf_fp).groupby("parameter"):

        # Skip the intercept
        if parameter == "(Intercept)":
            continue

        write_out(
            f"{input_prefix}/wald/parameter/",
            parameter,
            calc_mean_wald(
                lg_membership,
                gene_cag_dict,
                parameter_df.set_index("CAG")["wald"].to_dict()
            )
        )
        i += 1
    logging.info(
        f"Wrote out average Wald metrics for {i:,} parameters in this dataset"
    )


def pick_top_taxon(lg_taxa, tax_df, tax_rank, min_prop=0.5):
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

    logging.info(rank_assignments.head())

    # Pick the top hit per linkage group
    rank_assignments = pd.DataFrame(
        [
            {
                "linkage_group": lg_name,
                "top_hit": lg_assignments.sort_values().index.values[-1],
                "proportion": lg_assignments.sort_values().values[-1],
            }
            for lg_name, lg_assignments in rank_assignments.iterrows()
        ]
    )

    return rank_assignments.assign(
        top_hit_name=rank_assignments.apply(
            lambda r: tax_df.loc[r["top_hit"],
                                 "name"] if r["proportion"] >= min_prop else "Unassigned",
            axis=1
        )
    ).set_index(
        "linkage_group"
    )[
        "top_hit_name"
    ]


if __name__ == "__main__":

    log_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [Annotate GLAM Metagenome Map] %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Write logs to STDOUT
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(
        description="""
        Generate annotations for a metagenome map to be used in the GLAM Browser.

        Example Usage:

        annotate_metagenome_map.py \ 
            --input-prefix <INPUT_PREFIX> \ 
            --summary-hdf <SUMMARY_HDF> \ 
            --output-folder <OUTPUT_FOLDER> \ 
            --output-prefix <OUTPUT_PREFIX>

        """
    )

    parser.add_argument(
        "--input-prefix",
        type=str,
        help="Prefix for files generated by metagenome_map.py"
    )

    parser.add_argument(
        "--summary-hdf",
        type=str,
        help="Path to results from geneshot (HDF5)"
    )

    # Parse the arguments
    args = parser.parse_args()

    annotate_linkage_groups(
        args.input_prefix, 
        args.summary_hdf
    )
