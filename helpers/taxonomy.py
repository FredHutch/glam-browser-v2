from functools import lru_cache
import logging
import pandas as pd

class Taxonomy:

    def __init__(self, store):
        """Read in the taxonomy table."""

        # Read the taxonomy table
        logging.info("Reading in /ref/taxonomy")

        self.taxonomy_df = pd.read_hdf(
            store,
            "/ref/taxonomy"
        ).apply(
            lambda c: c.fillna(0).apply(float).apply(
                int) if c.name in ["parent", "tax_id"] else c,
        ).set_index(
            "tax_id"
        )

        self.all_taxids = set(self.taxonomy_df.index.values)

    @lru_cache(maxsize=None)
    def path_to_root(self, tax_id, max_steps=100):
        """Parse the taxonomy to yield a list with all of the taxa above this one."""

        visited = []

        for _ in range(max_steps):

            # Skip taxa which are missing
            if tax_id not in self.all_taxids:
                break

            # Add to the path we have visited
            visited.append(tax_id)

            # Get the parent of this taxon
            parent_id = self.taxonomy_df.loc[tax_id, "parent"]

            # If the chain has ended, stop
            if parent_id in visited or parent_id == 0:
                break

            # Otherwise, keep walking up
            tax_id = parent_id

        return visited

    @lru_cache(maxsize=None)
    def anc_at_rank(self, tax_id, rank):
        for anc_tax_id in self.path_to_root(tax_id):
            if self.taxonomy_df.loc[anc_tax_id, "rank"] == rank:
                return anc_tax_id

    def name(self, tax_id):
        if tax_id in self.all_taxids:
            return self.taxonomy_df.loc[tax_id, "name"]

    def parent(self, tax_id):
        if tax_id in self.all_taxids:
            return self.taxonomy_df.loc[tax_id, "parent"]

    def make_cag_tax_df(
        self,
        taxa_vc,
    ):
        """Return a nicely formatted taxonomy table from a list of tax IDs and the number of assignments for each."""

        # We will construct a table with all of the taxa in the tree, containing
        # The ID of that taxon
        # The name of that taxon
        # The number of genes assigned to that taxon (or its children)
        # The rank of that taxon
        # The ID of the parent of that taxon

        # The number of genes found at that taxon or in its decendents
        counts = defaultdict(int)

        # Keep track of the total number of genes with a valid tax ID
        total_genes_assigned = 0

        # Iterate over each terminal leaf
        for tax_id, n_genes in taxa_vc.items():

            # Skip taxa which aren't in the taxonomy
            if tax_id not in self.taxonomy_df.index.values:
                continue

            # Count all genes part of this analysis
            total_genes_assigned += n_genes

            # Walk up the tree from the leaf to the root
            for anc_tax_id in self.path_to_root(tax_id):

                # Add to the sum for every node we visit along the way
                counts[anc_tax_id] += n_genes

        if len(counts) == 0:
            return pd.DataFrame([{
                "tax_id": None
            }])

        # Make a DataFrame
        df = pd.DataFrame({
            "count": counts,
        })

        # Add the name, parent, rank
        df = df.assign(
            tax_id=df.index.values,
            parent=self.taxonomy_df["parent"],
            rank=self.taxonomy_df["rank"],
            name=self.taxonomy_df["name"],
            total=total_genes_assigned,
        )

        return df
