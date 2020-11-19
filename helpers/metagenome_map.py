#!/usr/bin/env python3

import argparse
from collections import defaultdict
from functools import lru_cache
import logging
import networkx as nx
import os
import pandas as pd
import numpy as np
from time import time
from copy import deepcopy


def glam_network(
    summary_hdf,
    detail_hdf,
    output_folder,
    output_prefix,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region=os.environ.get("AWS_REGION", "us-west-2"),
    min_node_size=10
):
    """Use co-abundance and co-assembly to render genes in a network layout."""

    # 1 - Build a set of gene membership for all contigs
    contig_gene_sets = build_contig_dict(summary_hdf, detail_hdf, remove_edge=False)

    # 2 - Combine contigs which satisfy a conservative overlap threshold
    merged_contig_sets = combine_overlapping_contigs(contig_gene_sets)

    # 3 - Build a network of gene linkage groups
    LN = LinkageNetwork(summary_hdf, merged_contig_sets)

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

    ############################
    # FINISHED PRUNING NETWORK #
    ############################

    # Read in the genome information
    genome_gene_sets = read_genome_gene_sets(detail_hdf)
    genome_overlap_df = count_genome_overlap(LN, genome_gene_sets)
    G = make_network(LN)
    add_genomes_to_graph(LN, G, genome_overlap_df)

    # Read the taxonomy
    tax = Taxonomy(summary_hdf)

    # Read the taxonomic assignment per gene
    taxonomy_df = read_taxonomy_df(summary_hdf)

    # Summarize each connected set of linkage groups on the basis of taxonomic spectrum
    # In the process, record the those discrete connected linkage groups
    tax_spectrum_df, node_groupings = make_tax_spectrum_df(LN, G, taxonomy_df, tax)

    # Summarize each linkage group on the basis of abundance
    ln_abund_df = get_linkage_group_abundances(LN, detail_hdf)

    # The path for output files will vary only by suffix
    output_path = lambda suffix: os.path.join(output_folder, f"{output_prefix}.{suffix}")

    # Write out the graph in graphml format
    nx.write_graphml(G, output_path("graphml"))

    # Write out the taxonomic spectrum for each subnetwork
    tax_spectrum_df.T.reset_index().to_feather(output_path("feather"))

    # Write out the table of which genes are part of which linkage group
    pd.DataFrame([
        {
            "gene": gene_name,
            "linkage_group": group_name
        }
        for group_name, gene_name_list in LN.group_genes.items()
        for gene_name in gene_name_list
    ]).to_feather(
        output_path("linkage-group-gene-index.feather")
    )

    # Write out the table of which linkage groups are connected
    pd.DataFrame([
        {
            "linkage_group": group_name,
            "subnetwork": subnetwork_name
        }
        for subnetwork_name, group_name_list in node_groupings.items()
        for group_name in group_name_list
    ]).to_feather(
        output_path("subnetworks.feather")
    )

    # Write out the table of relative abundances for each linkage group
    ln_abund_df.reset_index().to_feather(
        "2020-11-12-linkage-group-abundance.feather"
    )

@lru_cache(maxsize=1)
def assembly_key_list(detail_hdf):
    """Get the list of specimens which have assembly informtion."""
    with pd.HDFStore(detail_hdf, 'r') as store:
        return [
            p
            for p in store
            if p.startswith("/abund/allele/assembly/")
        ]


@lru_cache(maxsize=1)
def get_cag_size(summary_hdf):
    """Get the size of each CAG."""
    return read_cag_dict(summary_hdf).value_counts()


@lru_cache(maxsize=1)
def read_cag_dict(summary_hdf):
    """Get the dict matching each gene to a CAG."""
    return pd.read_hdf(
        summary_hdf,
        "/annot/gene/all",
        columns=["gene", "CAG"]
    ).set_index(
        "gene"
    )["CAG"]


@lru_cache()
def read_contig_info(summary_hdf, detail_hdf, path_name, remove_edge=True):
    """Read the contig information for a single assembly."""
    # Read the table
    print(f"Reading {path_name}")
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
        contig=df["contig"] + "_" + df["specimen"]
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


@lru_cache(maxsize=None)
def read_genome_info(summary_hdf, path_name):
    """Read the alignment information for a single genome,"""

    # Read the table
    with pd.HDFStore(summary_hdf, 'r') as store:
        df = pd.read_hdf(store, path_name)

    # Return the number of genes for each CAG
    return pd.DataFrame(
        dict(
            n_genes=df.groupby(
                ["genome_id", "CAG"]
            ).apply(
                len
            )
        )
    ).reset_index(
    ).assign(
        cag_size=lambda d: d["CAG"].apply(
            int
        ).apply(
            get_cag_size(summary_hdf).get
        ),
        genome_size=df.shape[0]
    )


def build_contig_dict(summary_hdf, detail_hdf, remove_edge=True, min_genes=2):
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
            remove_edge=remove_edge
        )

        # Iterate over every contig
        print(
            f"Adding {contig_df.shape[0]:,} genes across {contig_df['contig'].unique().shape[0]:,} contigs from {path_name.split('/')[-1]}")
        start_time = time()
        for contig_name, contig_genes in contig_df.groupby("contig"):

            # Skip contigs with less than `min_genes` genes
            if contig_genes.shape[0] >= min_genes:

                # Add the set to the dict
                contig_membership[contig_name] = set(
                    contig_genes["catalog_gene"].tolist())

        print(f"Done - {round(time() - start_time, 1):,} seconds elapsed")
        start_time = time()

    print(f"Returning {len(contig_membership):,} sets of gene membership")

    return contig_membership


def index_contig_gene_sets(contig_gene_sets):
    start_time = time()
    gene_index = defaultdict(set)
    for contig_name, gene_set in contig_gene_sets.items():
        for gene_name in list(gene_set):
            gene_index[gene_name].add(contig_name)
    print(
        f"Made index for {len(contig_gene_sets):,} contigs and {len(gene_index):,} genes -- {round(time() - start_time, 1):,} seconds")
    return gene_index


# def calculate_contig_containment_single(contigA, contig_gene_sets, gene_index):
#     """Function to consider a single contig."""

#     # Keep track of the best hit thus far
#     top_hit = None
#     top_hit_n = None

#     # Iterate over the other contigs
#     for contigB in list(set([
#         contig_name
#         for gene_name in list(contig_gene_sets[contigA])
#         for contig_name in list(gene_index[gene_name])
#     ])):

#         # Don't compare to self
#         if contigA == contigB:
#             continue

#         # Check to see if this is a match
#         hit_n = len(contig_gene_sets[contigA] & contig_gene_sets[contigB])

#         # See if this is the new 'best hit'
#         if top_hit_n is None or hit_n > top_hit_n:

#             # Set the top hit
#             top_hit = contigB
#             top_hit_n = hit_n

#     return {
#         "contig": contigA,
#         "contig_size": len(contig_gene_sets[contigA]),
#         "best_match": top_hit,
#         "match_n": top_hit_n,
#     }


# def calculate_contig_containment(contig_gene_sets):
#     """Function to combine results for all contigs."""

#     # Make an index grouping contigs by which genes they contain
#     gene_index = index_contig_gene_sets(contig_gene_sets)

#     start_time = time()
#     final_df = pd.DataFrame([
#         calculate_contig_containment_single(
#             contig_name,
#             contig_gene_sets,
#             gene_index
#         )
#         for contig_name in list(contig_gene_sets.keys())
#     ]).assign(
#         match_prop=lambda df: df["match_n"].fillna(0) / df["contig_size"]
#     )
#     print(
#         f"Made membership table -- {round(time() - start_time, 1):,} seconds"
#     )
    
#     return final_df


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

        print(
            f"Merged {found_overlap:,} contigs -- {round(time() - start_time, 1):,} seconds")

    # Return the final set of merged contigs
    print(f"Returning a set of {len(merged_contig_sets):,} contigs")
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

    print(f"Found {len(linkage_groups):,} linkage groups")
    return linkage_groups


class LinkageNetwork:

    def __init__(self, summary_hdf, merged_contig_sets, max_cag_size=10000):

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

        print(
            f"Added genes from contigs to create a network with {len(self.group_genes):,} linkage groups and {sum(map(len, self.group_genes.values())):,} genes")

        # Now we need to add LGs which contain genes that do not align to any contigs
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

        print(
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

    print(
        f"Merging nodes which fall under the connectivity ratio of 10^{max_ratio}")

    # Make a table with the number of genes, and the number of connections, for all groups
    summary_df = group_summary_table(LN)

    # Filter down to the gene groups which fall into this ratio
    summary_df = summary_df.query(
        f"connection_size_ratio_log10 <= {max_ratio}"
    )
    print(f"Screening a batch of {summary_df.shape[0]:,} groups")

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
            print(f"Trimmed {ntrimmed:,} groups")

    # Get the vector of group sizes
    vc = LN.group_sizes()
    print(
        f"Trimmed {ntrimmed:,} groups, resulting in {vc.shape[0]:,} groups for {vc.sum():,} genes"
    )


def merge_terminal_nodes(
    LN,
    max_size=10,  # Merge all terminal nodes which container fewer than this number of genes
):
    """Merge nodes which are terminal (have only 1 non-self connection)."""

    print(f"Merging terminal nodes smaller than {max_size}")

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
        print(
            f"Merged {ntrimmed:,} groups, resulting in {vc.shape[0]:,} groups for {vc.sum():,} genes"
        )

def remove_unconnected_nodes(
    LN,
    max_size_unconnected=5,
):
    """Remove all unconnected nodes which container fewer than this number of genes."""

    print(f"Removing unconnected nodes smaller than {max_size_unconnected}")

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
    print(
        f"Trimmed {ntrimmed:,} groups, resulting in {vc.shape[0]:,} groups for {vc.sum():,} genes")


def read_genome_gene_sets(summary_hdf):

    # Iterate over every genome in the HDF
    static_genome_key_list = genome_key_list(summary_hdf)
    print(
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

    print(f"Read in data for {len(output):,} genomes")
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

    print(f"Building a network with {len(LN.group_genes):,} linkage groups")

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
    print("Done")

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

    print(f"Adding {n_genomes:,} genomes to the graph")

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
    print("Done")


class Taxonomy:

    def __init__(self, summary_hdf):
        """Read the taxonomy structure."""
        self.tax = pd.read_hdf(summary_hdf, "/ref/taxonomy").set_index(
            "tax_id"
        ).apply(
            lambda c: c.apply(lambda v: 'none' if pd.isnull(
                v) else str(int(float(v)))) if c.name == 'parent' else c
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
        )


# Make a DataFrame with the taxonomy spectrum for every connected group
def make_tax_spectrum_df(LN, G, taxonomy_df, tax):

    # Make a DataFrame with the taxonomy spectrum for every connected group
    tax_spectrum_df = {}

    # Make a Series with the number of genes in each connected group
    node_size = {}

    # Keep track of the group that each node belongs to
    node_groupings = {}

    # Iterate over each of the connected groups in the graph
    for node_set in nx.connected_components(G):

        # Link the taxon spectrum to the largest group of genes in this set of nodes
        largest_name = largest_node(node_set)
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
            print(f"Found taxa for {len(tax_spectrum_df)} connected groups")

    # Format as a DataFrame
    node_size = pd.Series(node_size)
    tax_spectrum_df = pd.DataFrame(tax_spectrum_df).fillna(0)
    assert node_size.shape[0] == tax_spectrum_df.shape[1]
    print(
        f"Found {tax_spectrum_df.shape[0]:,} taxa for {tax_spectrum_df.shape[1]:,} connected groups")

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


def largest_node(node_set):
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


# def name_tax_spectra(tax_spectra, tax):
#     df = pd.DataFrame(dict(
#         prop=tax_spectra.loc[tax_spectra > 0]
#     ))
#     df = df.assign(
#         name=tax.tax["name"],
#         rank=tax.tax["rank"],
#     ).sort_values(
#         by="prop",
#         ascending=False
#     ).groupby(
#         "rank"
#     ).head(
#         1
#     ).set_index(
#         "rank"
#     ).reindex(
#         index=["species", "genus", "family", "order", "class", "phylum"]
#     ).dropna(
#     ).query(
#         "prop > 0.25"
#     )
#     if df.shape[0] > 0:
#         return df["name"].values[0]
#     else:
#         return "Unclassified"


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
                print(
                    f"Computed relative abundance of linkage groups across {len(abund_dict):,} specimens")

    print(
        f"Computed relative abundance of linkage groups across {len(abund_dict):,} specimens"
    )

    # Make a DataFrame
    return pd.DataFrame(abund_dict).fillna(0)


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
            --output-folder <OUTPUT_FOLDER> \ 
            --output-prefix <OUTPUT_PREFIX>

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
        "--output-prefix",
        type=str,
        help="Prefix for all output files"
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
        args.output_prefix,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region=os.environ.get("AWS_REGION", "us-west-2"),
        min_node_size=10
    )
