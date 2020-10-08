# Gene-Level Association of Microbiomes (GLAM) - Browser

The GLAM Browser is intended to make it easier to visualize and understand the
results of a microbiome experiment which has been analyzed by whole-genome
shotgun (WGS) sequencing and analyzed with gene-level metagenomic analysis.
The analytical tool used to process raw data from FASTQ format which is
compatible with visualization using the GLAM Browser is the `geneshot`
analysis tool ([link](https://github.org/golob-minot/geneshot)).

## Running the GLAM Browser

### Prerequisites

In order to run the GLAM Browser, you need to first have access to:

- Docker
- MariaDB database (required details: hostname, port, database name, username, and password)
- Data storage bucket in AWS S3 (required details: AWS credentials with access)
- Files output by `geneshot`, summarizing gene-level metagenomic analysis results

### GLAM Configuration as Environment Variables

The easiest way to specify the configuration needed to run the GLAM Browser is to
create a small file which includes the details for each of the prerequisites listed
above. There is an empty example of such a file provided in this repository as
`glam-browser-v2.env`. To set the environment variables listed in this file, simply
run `source glam-browser-v2.env`.

#### AWS Credentials

Note that there are two options for specifying AWS credentials in the example
`glam-browser-v2.env`. The preferred option is to set the `AWS_PROFILE`, which
specifies a profile name used with `aws --profile NAME configure`. An alternate
approach is to set each of the individual componants of that profile, including
the ID, secret key, and region. Using the `AWS_PROFILE` is preferred as it
limits the number of places in your filesystem where these sensitive credentials
are saved.

### Installing the GLAM Browser CLI

We have made a small command-line interface with the functions needed to set up
the GLAM Browser. This Python utility is most easily installed directly from
GitHub:

```#!/bin/bash
python3 -m pip install git+https://github.com/FredHutch/glam-browser-v2.git
```

Once installed, the GLAM Browser CLI can be run using the command `glam-cli`,
with basic options listed with the `--help` flag.

```#!/bin/bash
glam-cli --help
```

### Setting Up the GLAM Browser

The GLAM CLI provides a number of functions which can be used to add users and
datasets to the GLAM Browser. This results in data being written to the S3 bucket
and MariaDB database that you have specified. 

**Please be aware that writing data to AWS S3 will incur costs to your account.**

Here is an example of how you might want to set up a single account with a single
dataset:

```#!/bin/bash
# Stop execution if any errors are encountered
set -e

# Set the environment variables used by GLAM
source glam-browser-v2.env

# Set up the database itself
glam-cli setup

# Add a user
glam-cli add-user --username userName --email user@domain.xyz

# Set the password for that user
glam-cli set-password --username userName --password "<PASSWORD>"

# Index a dataset (writing to AWS S3)
SOURCE="/path/to/geneshot/output/results.hdf5"
DESTINATION="s3://${S3_BUCKET}/prefix/for/this/dataset"
glam-cli index-dataset --fp "$SOURCE" --uri "$DESTINATION"

# Add that dataset to the GLAM database
glam-cli add-dataset --uri "${DESTINATION}" --dataset-id short_dataset_id --name "Long Dataset Name"

# Grant access for that user to the dataset
glam-cli grant-access --user-name userName --dataset-name short_dataset_id
```

### Launching the GLAM Browser

Once you have set up the database (MariaDB) and all data objects (AWS S3) as described above,
you are ready to actually launch the GLAM Browser pointing to those datasets. In this example
we will show how you can run the GLAM Browser locally using Docker. To deploy the browser in
a way that is available to multiple users, you will have to follow the specifications of your
hosting provider.

#### Docker

Before you start, make sure that Docker is running. You can test this quickly by pulling the
GLAM image with `docker pull quay.io/fhcrc-microbiome/glam-browser-v2:latest`.

#### Configuration

If you are starting at this step, make sure that the GLAM configuration is set up with the
approriate environment variables as described above (e.g. `source glam-browser-v2.env`).

#### Starting Up

With all of the data and configuration in place, you can launch the GLAM Browser with the
command:

```#!/bin/bash
docker-compose -f public-docker-compose.yml up
```

This command will download the latest GLAM Browser image and start it, while passing in the
needed environment variables and opening a port for viewing in the browser. Once the
launching process has completed, you can access it using your web browser (e.g. Chrome) at
[0.0.0.0:8050](0.0.0.0:8050).

#### Notes on Security

The example shown in this documentation has the GLAM Browser deployed on a single isolated
computer with no network access provided to external users.
When deploying the GLAM Browser in any environment other than a single isolated computer,
it is essential that TLS security is used to encrypt the credentials used to access data.
Furthermore, the container hosting the browser must be protected from any unauthorized
access, as the AWS credentials used to access the raw data are stored as environment
variables. To mitigate any possibility of data loss, we strongly recommend that a dedicated
IAM be used for the GLAM Browser which has read-only access to the bucket used for hosting.
With that deployment approach, the AWS credentials used to write to the S3 bucket (in the
`glam-cli index-dataset` step) should be distinct from those used to launch the browser.
To be clear, the former will have read-write access, while the latter will be read-only.

### Details on GLAM Browser Displays

Scientific details of the visualizations presented by the GLAM Browser are found in [the wiki for this repository](https://github.com/FredHutch/glam-browser-v2/wiki).
