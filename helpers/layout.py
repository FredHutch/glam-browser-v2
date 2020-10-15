#!/usr/bin/env python3
from collections import OrderedDict
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_table
import pandas as pd
import numpy as np
from .common import encode_search_string
from .common import decode_search_string

class GLAM_LAYOUT:

    def __init__(
        self,
        gtm_container=None,
        glam_db=None,
        glam_io=None,
        brand="GLAM Browser",
        **kwargs
    ):

        # Store the Google Tag Manager container ID, if any
        self.gtm_container = gtm_container

        # Store the 'brand' of the page
        self.brand = brand

        # Store the GLAM_DB object
        self.glam_db = glam_db
        self.glam_io = glam_io

    def base(self):
        
        return html.Div(
            self.top_navbar(
            ) + self.sub_navbar(
            ) + [
                # Contains the URL
                dcc.Location(id='url', refresh=False),

                # Modal for login with username and password
                self.login_modal(),

                # Modal to change the user's password
                self.change_password_modal(),

                # Page content will be filled in using this element
                dbc.Spinner(id='page-content'),

                # Div with the manifest table used to mask specimens
                self.manifest_div(),

                # Store the browsing history
                html.Div(children=[], id="history-forward", style={"display": "none"}),
                html.Div(children=[], id="history-reverse", style={"display": "none"}),

            ],
            className="container"
        )

    def manifest_div(self):
        return  html.Div(
            [
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Row([
                            dbc.Col(
                                dcc.Markdown("Specimen Manifest", style={"marginTop": "10px"})
                            ),
                            dbc.Col(
                                dbc.Button(
                                    html.Span(html.I(className="fas fa-search")),
                                    id="open-manifest",
                                    n_clicks=0,
                                    style={"margin": "10px"}
                                ),
                                style={"textAlign": "right", "marginTop": "5px"}
                            )
                        ]),
                    ),
                    dbc.Collapse(
                        dbc.CardBody([
                            dash_table.DataTable(
                                id="manifest-table",
                                columns=[{"name": "specimen", "id": "specimen"}],
                                data=[{'specimen': 'none'}],
                                row_selectable='multi',
                                style_table={'minWidth': '90%'},
                                style_header={
                                    "backgroundColor": "rgb(2,21,70)",
                                    "color": "white",
                                    "textAlign": "center",
                                },
                                page_action='native',
                                page_size=20,
                                filter_action='native',
                                sort_action='native',
                                hidden_columns=[],
                                selected_rows=[],
                                css=[{"selector": ".show-hide", "rule": "display: none"}],
                            ),
                            html.Br(),
                            html.Label("Show / Hide Columns"),
                            dcc.Dropdown(
                                id="manifest-table-select-columns",
                                options=[{"label": n, "value": n} for n in ["specimen"]],
                                value=["specimen"],
                                multi=True,
                            ),
                            html.Br(),
                            html.Label("Bulk Show / Hide Specimens"),
                            dbc.Row([
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="select-specimens-show-hide",
                                        options=[{"label": n, "value": n} for n in ["Show", "Hide"]],
                                        value="Hide",
                                        multi=False,
                                    )
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="select-specimens-col",
                                        options=[{"label": n, "value": n} for n in ["specimen"]],
                                        value="specimen",
                                        multi=False,
                                    )
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="select-specimens-comparitor",
                                        options=[{"label": n, "value": n} for n in [
                                            "Equals",
                                            "Does not equal",
                                            "Is greater than",
                                            "Is less than",
                                            "Is greater than or equal to",
                                            "Is less than or equal to",
                                        ]],
                                        value="Equals",
                                        multi=False,
                                    )
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="select-specimens-value",
                                        options=[{"label": "none", "value": "none"}],
                                        value="none",
                                        multi=False,
                                    )
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Apply",
                                        id="select-specimens-apply",
                                        disabled=True,
                                        href="/datasets",
                                    )
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Reset",
                                        id="select-specimens-reset",
                                        href="/datasets",
                                    )
                                ),
                            ])
                        ]),
                    id="manifest-table-body",
                    is_open=False
                    )
                ]),
            ],
            id="manifest-table-div",
            style={"marginTop": "20px", "marginBottom": "20px", "display": "none"},
        )

    def login_modal(self):
        return dbc.Modal(
            [
                dbc.ModalHeader(
                    "Log In"
                ),
                dbc.ModalBody(
                    [
                        dbc.FormGroup(
                            [
                                dbc.Label("GLAM Browser Username"),
                                dbc.Input(
                                    type="text", 
                                    id="username", 
                                    placeholder="Enter username",
                                    debounce=True,
                                ),
                                dbc.FormText(
                                    "Please provide the username for your GLAM Browser account",
                                    color="secondary",
                                ),
                            ]
                        ),
                        dbc.FormGroup(
                            [
                                dbc.Label(
                                    "GLAM Browser Password"),
                                dbc.Input(
                                    type="password", 
                                    id="password", 
                                    placeholder="Enter password",
                                    debounce=True,
                                ),
                                dbc.FormText(
                                    "Please provide the password for your GLAM Browser account",
                                    color="secondary",
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Apply",
                            id="login-modal-apply-button",
                            n_clicks=0,
                        ),
                    ]
                ),
            ],
            id="login-modal",
            centered=True,
            keyboard=False,
            backdrop="static",
        )

    def change_password_modal(self):
        return dbc.Modal(
            [
                dbc.ModalHeader(
                    "Change Password"
                ),
                dbc.ModalBody(
                    [
                        dbc.FormGroup(
                            [
                                dbc.Label("Username"),
                                dbc.Input(
                                    type="text", 
                                    id="change-password-username", 
                                    placeholder="Enter username",
                                    debounce=True,
                                ),
                                dbc.FormText(
                                    "Please provide the username for your GLAM Browser account",
                                    color="secondary",
                                ),
                            ]
                        ),
                        dbc.FormGroup(
                            [
                                dbc.Label("Current Password"),
                                dbc.Input(
                                    type="password", 
                                    id="change-password-old", 
                                    placeholder="Enter current password",
                                    debounce=True,
                                ),
                                dbc.FormText(
                                    "Please provide the current password for your GLAM Browser account",
                                    color="secondary",
                                ),
                            ]
                        ),
                        dbc.FormGroup(
                            [
                                dbc.Label("New Password"),
                                dbc.Input(
                                    type="password", 
                                    id="change-password-new", 
                                    placeholder="Enter new password",
                                    debounce=True,
                                ),
                                dbc.FormText(
                                    "Please provide the new password for your GLAM Browser account",
                                    color="secondary",
                                ),
                            ]
                        ),
                        dbc.FormGroup(
                            [
                                dbc.Label("Confirm New Password"),
                                dbc.Input(
                                    type="password", 
                                    id="change-password-confirm", 
                                    placeholder="Re-enter new password",
                                    debounce=True,
                                ),
                                dbc.FormText(
                                    "Please provide the new password for your GLAM Browser account",
                                    color="secondary",
                                ),
                            ]
                        ),
                        html.Div(
                            children=None,
                            id="change-password-response-text"
                        )
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Apply",
                            id="change-password-apply-button",
                            n_clicks=0,
                        ),
                        dbc.Button(
                            "Close",
                            id="change-password-close-button",
                            n_clicks=0,
                        ),
                    ]
                ),
            ],
            id="change-password-modal",
            centered=True,
            keyboard=False,
            backdrop="static",
        )

    def list_bookmarks_modal(self):
        return dbc.Modal(
            [
                dbc.ModalHeader(
                    "Bookmarks"
                ),
                dbc.ModalBody(
                    html.Div(id="bookmarks-modal-body")
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Close",
                            id={"type": "close-modal", "name": "list-bookmarks"},
                        ),
                    ]
                ),
            ],
            id={"type": "modal", "name": "list-bookmarks"},
            centered=True,
            is_open=False,
        )

    def save_bookmark_modal(self):
        return dbc.Modal(
            [
                dbc.ModalHeader(
                    "Save Bookmark"
                ),
                dbc.ModalBody(
                    dbc.FormGroup(
                        [
                            dbc.Label("Bookmark Name"),
                            dbc.Input(
                                type="text", 
                                id="save-bookmark-name", 
                                placeholder="Page name",
                                debounce=True,
                            ),
                            dbc.FormText(
                                "Please provide a name describing this page",
                                color="secondary",
                            ),
                        ]
                    )
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Close",
                            id={"type": "close-modal", "name": "save-bookmark"},
                        ),
                        dbc.Button(
                            "Save",
                            id="save-bookmark-button",
                        ),
                    ]
                ),
            ],
            id={"type": "modal", "name": "save-bookmark"},
            centered=True,
            is_open=False,
        )

    def bookmark_list(self, bookmarks):
        
        if len(bookmarks) == 0:
            return dcc.Markdown("No bookmarks saved")
        else:
            return html.Div([
                dbc.Row(
                    [
                        dbc.Col(dcc.Markdown(i["bookmark_name"]), width=8),
                        dbc.Col(
                            [
                                dbc.Button(
                                    html.Span(html.I(className="fas fa-link")),
                                    id={"name": "link-in-bookmarks-modal", "index": ix},
                                    href=i["bookmark"],
                                    style={"margin": "5px", "textAlign": "right"}
                                ),

                                dbc.Button(
                                    html.Span(html.I(className="fas fa-trash-alt")),
                                    id={"name": "delete-bookmark", "index": ix},
                                    style={"margin": "5px", "textAlign": "right"}
                                ), 
                            ]
                        ),
                    ]
                )
                for ix, i in enumerate(bookmarks)
            ])

    def jumbotron_page(self, children):
        return dbc.Jumbotron(
            [
                dbc.Container(
                    children,
                    fluid=True,
                )
            ],
            fluid=True,
            style={"background-color": "white"}
        )

    # BACKGROUND OF LOGIN MODAL
    def login_page_background(self):
        return self.jumbotron_page([])

    # NOT LOGGED IN PAGE
    def logged_out_page(self):
        return self.jumbotron_page([
            dcc.Markdown("### Welcome to the GLAM Browser\n\nTo get started, log in to your account or browse some publicly available datasets."),
            html.Br(),
            dbc.Button("Log In", style={"margin": "10px"}, id={"type": "login-button", "parent": "logged-out-page"}),
            dcc.Link(dbc.Button("Public Datasets", style={"margin": "10px"}), href="/public"),
        ])

    # PAGE NOT FOUND
    def page_not_found(self):
        return self.jumbotron_page([
            dcc.Markdown("### Welcome to the GLAM Browser\n\nThe page you have requested is not available."),
        ])

    # Formats the navbar at the top of the page
    def top_navbar(self):
        return [dbc.NavbarSimple(
            brand=self.brand,
            color="primary",
            dark=True,
            children=[
                dbc.DropdownMenu(
                    label="User",
                    children=[
                        dbc.DropdownMenuItem(
                            'Log In',
                            id={"type": "login-button", "parent": "navbar"}, 
                        ),
                        dbc.DropdownMenuItem(
                            'Log Out',
                            href="/",
                            external_link=True,
                        ),
                        dbc.DropdownMenuItem(
                            'Change Password',
                            id="change-password-button",
                        )
                    ],
                    right=True,
                    in_navbar=True,
                    caret=True,
                )

            ],
            id="navbar"
        )]
    
    # Navbar with the username, back, forward, and bookmark buttons
    def sub_navbar(self):
        return [
            dbc.NavbarSimple(
                brand="Anonymous",
                dark=False,
                color="secondary",
                children=[
                    dcc.Link(
                        dbc.Button(
                            html.Span(
                                html.I(className="fas fa-arrow-left"), 
                                id="back-button"
                            )
                        ),
                        href="/back",
                        id="back-button-link",
                    ),
                    dcc.Link(
                        dbc.Button(
                            html.Span(
                                html.I(className="fas fa-arrow-right"),
                                id="forward-button"
                            )                            
                        ),
                        href="/forward",
                        id="forward-button-link",
                    ),
                    dbc.Button(
                        html.Span(html.I(className="fas fa-bookmark"), id="save-bookmark-modal-button"),
                        id={"type": "open-modal", "name": "save-bookmark"}
                    ),
                    dbc.Button(
                        html.Span(html.I(className="fas fa-bars"), id="open-bookmarks-button"),
                        id={"type": "open-modal", "name": "list-bookmarks"}
                    ),
                    dcc.Link(
                        dbc.Button(
                            html.Span(
                                html.I(className="fas fa-home"),
                                id="home-button"
                            )
                        ),
                        href="/datasets"
                    ),
                ],
                id="sub-navbar"
            ),
            self.save_bookmark_modal(),
            self.list_bookmarks_modal(),
        ] + [
            dbc.Tooltip(text, target=target)
            for text, target in [
                ("Back", "back-button"),
                ("Forward", "forward-button"),
                ("Save Bookmark", "save-bookmark-modal-button"),
                ("Open Bookmarks", "open-bookmarks-button"),
                ("Home", "home-button")
            ]
        ]


    # LIST OF DATASETS
    def dataset_list(self, dataset_list, username):
        """This is the page which will display a list of datasets."""

        return [
            self.dataset_button(dataset)
            for dataset in dataset_list
        ]

    # BUTTON FOR A SINGLE DATASET
    def dataset_button(self, dataset_id, action="open", search_string=None):
        
        assert action in ["open", "close"], "Options for action are open/close"

        # Get the human readable name for the dataset
        dataset_name = self.glam_db.get_dataset_name(dataset_id)

        # Check to see if a search string was passed in
        if search_string is not None:
            args = decode_search_string(search_string)
            if "n" in args:
                try:
                    n = int(args["n"])
                except:
                    n = None
        else:
            n = None
            args = {}

        # If we don't have the dataset size in the search string
        if n is None:
            # Let's read in the manifest to get the number of specimens
            dataset_uri = self.glam_db.get_dataset_uri(dataset_id)
            n = self.glam_io.get_manifest(dataset_uri).shape[0]

        # Pass the dataset size in with the search string
        if action == "open":
            href = "/d/{}/analysis{}".format(
                dataset_id,
                encode_search_string({
                    "n": n,
                    "mask": args.get("mask"),
                })
            )
        else:
            href = "/datasets"

        return dbc.Card(
            dbc.CardBody(
                dbc.Row([
                    dbc.Col(
                        dbc.Label(
                            "{}: {:,} specimens".format(
                                dataset_name,
                                n
                            )
                        ),
                        style={"textAlign": "left"}
                    ),
                    dbc.Col(
                        dcc.Link(
                            dbc.Button(
                                "Open Dataset" if action == "open" else "Close Dataset", 
                                color="secondary"
                            ),
                            href=href,
                        ),
                        style={"textAlign": "right"}
                    )
                ])
            ),
            style={"marginTop": "10px", "marginBottom": "10px"}
        )


    # DISPLAY ALL ANALYSES AVAILABLE FOR A SINGLE DATASET
    def dataset_display(self, username, dataset_id, search_string):

        # Unpack arguments from the search string
        args = decode_search_string(search_string)

        # If we don't have the argument "n=" included, return an error
        if args.get("n") is None:
            return self.jumbotron_page("Missing argument n=")
        try:
            args["n"] = int(args.get("n"))
        except:
            return self.jumbotron_page("Could not decode 'n=' as an integer")

        output_layout = [
            # At the top of the dataset display is a banner with the dataset name
            self.dataset_button(
                dataset_id,
                action="close"
            )
        ]

        # Below that is a card with the list of each analysis that is available
        for analysis in self.analysis_list(self.glam_db.get_dataset_uri(dataset_id)):

            # Skip cards that are disabled for this dataset
            if analysis is None:
                continue

            # Append to the layout
            output_layout.append(
                dbc.Card(
                    [
                        dbc.CardHeader(dbc.Row([
                            dbc.Col(
                                html.Div(
                                    dcc.Markdown(
                                        f"**{analysis.long_name}**:\t{analysis.description}"
                                    ), 
                                    style={"marginTop": "10px"}
                                ),
                                width=10,
                            ),
                            dbc.Col(
                                dcc.Link(
                                    dbc.Button("Open Analysis", color="secondary"),
                                    href="/d/{}/a/{}{}".format(
                                        dataset_id,
                                        analysis.short_name,
                                        encode_search_string({
                                            "n": args["n"],
                                            "mask": args["mask"],
                                            **{
                                                k: v(
                                                    self.glam_io, 
                                                    self.glam_db.get_dataset_uri(dataset_id)
                                                )
                                                for k, v in analysis.dynamic_defaults.items()
                                            },
                                            **analysis.defaults
                                        })
                                    )
                                ),
                                style={"textAlign": "right", "marginTop": "5px"},
                                width=2,
                            )
                        ]))
                    ],
                    style={"marginTop": "10px", "marginBottom": "10px"}
                )
            )

        return output_layout

    # Return the list of analyses available for a dataset
    # notably, this list is ordered
    def analysis_list(self, dataset_uri):
        return [
            ExperimentSummaryCard(),
            RichnessCard(),
            SingleSampleCard(),
            OrdinationCard(),
            CAGSummaryCard(),
            CagAbundanceHeatmap(),
            AnnotationHeatmapCard(),
            VolcanoCard() if self.glam_io.has_parameters(dataset_uri) else None,
            PlotCagCard(),
            TaxonomySunburstCard(),
            GenomeContainmentHeatmap() if self.glam_io.has_genomes(dataset_uri) and self.glam_io.has_genome_parameters(dataset_uri) else None,
            GenomeAssociationCard() if self.glam_io.has_genomes(dataset_uri) and self.glam_io.has_genome_parameters(dataset_uri) else None,
            GenomeAlignmentsCard() if self.glam_io.has_genomes(dataset_uri) else None,
        ]

    # Key the analysis by `short_name`
    # notably, dicts are not ordered
    def analysis_dict(self, dataset_uri):
        return {
            analysis.short_name: analysis
            for analysis in self.analysis_list(dataset_uri)
            if analysis is not None
        }

    # Get the default values for a particular AnalysisCard
    def get_defaults(self, dataset_id, analysis_id):
        return self.analysis_dict(
            self.glam_db.get_dataset_uri(dataset_id)
        )[
            analysis_id
        ].defaults

    # Return the page for a spcific analysis
    def analysis_page(self, username, password, dataset_id, analysis_name, search_string):

        # Make sure the user is logged in
        if self.glam_db.valid_username_password(username, password) is False:
            return self.page_not_found()

        # Make sure that the user is allowed to access this dataset
        if self.glam_db.user_can_access_dataset(dataset_id, username, password) is False:
            return self.page_not_found()

        # Make sure we have this analysis
        if analysis_name not in self.analysis_dict(self.glam_db.get_dataset_uri(dataset_id)):
            return self.page_not_found()

        # Get the URI for the dataset
        dataset_uri = self.glam_db.get_dataset_uri(dataset_id)

        return [
            # At the top of the dataset display is a banner with the dataset name
            self.dataset_button(
                dataset_id,
                action="close"
            ),
            # Below that is a card with the analysis itself            
            self.analysis_dict(
                self.glam_db.get_dataset_uri(dataset_id)
            )[analysis_name].card(
                # ID for this dataset
                dataset_id, 
                # Location to read data from
                dataset_uri,
                # Search string contains all parameters needed to render the card
                search_string,
                # Pass in the GLAM_IO object so that needed data can be read
                self.glam_io
            )
        ]

    def exp_table_row(self, header1, value1, header2, value2, header_bg="#F4F6F6", value_bg="white", spacer_bg="white"):
        return [     # Table Body
            html.Tr([    # Row
                html.Td(
                    header1,
                    style={"backgroundColor": header_bg}
                ),
                html.Td(
                    value1,
                    style={"backgroundColor": value_bg}
                ),
                html.Td(
                    "",
                    style={"backgroundColor": spacer_bg}
                ),
                html.Td(
                    header2,
                    style={"backgroundColor": header_bg}
                ),
                html.Td(
                    value2,
                    style={"backgroundColor": value_bg}
                ),
            ]
            )]
            
    def format_experiment_summary_table(self, dataset_metrics):
        return html.Div([
            dbc.Table([
                html.Tbody(
                    self.exp_table_row(  # Wrapper for each row
                        "Total Reads",
                        "{:,}".format(
                            int(dataset_metrics.loc["total_reads"])
                        ),
                        "Aligned Reads",
                        "{}%".format(round(
                            100 * \
                            float(dataset_metrics.loc["aligned_reads"]) / \
                            float(dataset_metrics.loc["total_reads"]),
                            1
                        ))
                    ) + self.exp_table_row(
                        "Genes (#)",
                        "{:,}".format(
                            int(dataset_metrics.loc["num_genes"])
                        ),
                        "CAGs (#)",
                        "{:,}".format(
                            int(dataset_metrics.loc["num_cags"])
                        )
                    ) + self.exp_table_row(
                        "Specimens (#)",
                        "{:,}".format(
                            int(dataset_metrics.loc["num_samples"])
                        ),
                        "Formula",
                        "{}".format(dataset_metrics.loc["formula"])
                    )
                )
            ], bordered=True, hover=True, responsive=True),
        ])

#####################################
# CARD CONTAINING ANALYSIS DISPLAYS #
#####################################

class AnalysisCard:

    def __init__(self):

        # Placeholders which will be replaced in the __init__
        # method used by each of the AnalysisCards below
        self.long_name = "Generic Card"
        self.description = "Description of the card"
        self.short_name = "generic_card"
        self.args = {}

        # Each analysis card has a list of plot elements.
        # This is used to link the default arguments defined
        # on each page to the list of plot elements displayed
        self.plot_list = []

        # The help_text string will be filled into the help modal
        self.help_text = None

        # This dict provides a way to set defaults dynamically
        # using a function which takes a GLAM_IO and dataset_uri as inputs
        self.dynamic_defaults = dict()

    def format_href(self, args, dataset_id, **addl_args):

        # Update the arguments
        for k, v in addl_args.items():
            args[k] = v

        # Format the link with these settings
        return "/d/{}/a/{}{}".format(
            dataset_id,
            self.short_name,
            encode_search_string(args)
        )

    def card_wrapper(
        self,
        dataset_id,
        card_body,
        custom_id=None,
        custom_style=None,
    ):
        # Must provide help text
        assert self.help_text is not None, "Must provide help text for every card"

        # Make a link (href) which will be applied to the pencil icon
        # The functionality will be to add or subtract the 'editable' function from self.plot_div()
        pencil_args = self.args.copy()
        if pencil_args.get("editable", "false") == "true":
            pencil_args["editable"] = "false"
        else:
            pencil_args["editable"] = "true"
        pencil_href = self.format_href(pencil_args, dataset_id)

        return html.Div([
            html.Br(),
            dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col(
                            html.Div(
                                self.long_name,
                                style={"marginTop": "10px"}
                            ),
                        ),
                        dbc.Col(
                            html.Div([
                                dbc.Button(
                                    html.Span(html.I(className="fas fa-pen-square")),
                                    id="make-plot-editable",
                                    style={"margin": "10px"},
                                    href=pencil_href
                                ),
                                dbc.Button(
                                    html.Span(html.I(className="far fa-question-circle")),
                                    id="open-help-text",
                                    n_clicks=0,
                                    style={"margin": "10px"}
                                ),
                                dcc.Link(
                                    dbc.Button(
                                        html.Span(html.I(className="fas fa-times")),
                                        style={"margin": "10px"}
                                    ),
                                    href="/d/{}/analysis{}".format(
                                        dataset_id,
                                        encode_search_string({
                                            k: v
                                            for k, v in self.args.items()
                                            if k in ["n", "mask"]
                                        })
                                    )
                                ),
                                dbc.Modal(
                                    [
                                        dbc.ModalHeader(
                                            self.long_name
                                        ),
                                        dbc.ModalBody(
                                            dcc.Markdown(self.help_text)
                                        ),
                                        dbc.ModalFooter(
                                            dbc.Button(
                                                "Close",
                                                id={
                                                    "type": "close-help-text",
                                                    "parent": self.short_name
                                                },
                                                className = "ml-auto"
                                            )
                                        )
                                    ],
                                    id = {
                                        "type": "help-text-modal",
                                        "parent": self.short_name
                                    }
                                )],
                                style={"textAlign": "right"}
                            ),
                        )
                    ])
                ),
                dbc.CardBody([
                    dbc.Collapse(
                        card_body,
                        is_open=True,
                        id={"type": "collapsable-card-body", "parent": self.short_name}
                    )
                ])
            ]),
            dbc.Tooltip(
                "Help Text", 
                target="open-help-text"
            ),
            dbc.Tooltip(
                "Toggle Manual Editing of Axis Labels", 
                target="make-plot-editable"
            ),
        ],
            id=self.short_name if custom_id is None else custom_id,
            style=custom_style
        )
        
    def multiselector(
        self, 
        label="Label", 
        options={},
        key=""
    ):
        # Show a label which also includes the current selection,
        # and present a dropdown menu of links
        # By default, no values are selected.
        # Clicking on an entry in the list will toggle it as being on/off

        # Get the list of currently selected values
        curr = [
            i
            for i in self.args.get(key, "").split(",")
            if len(i) > 0
        ]

        # The link for each item in the list will return a different
        # href which adds or subtracts the item from the list

        # Add or subtract the value from the list of selected values
        new_selection = lambda short_id: [i for i  in curr if i != short_id] if short_id in curr else curr + [short_id]

        # Format the new list of selected values as a string
        new_key = lambda short_id: ",".join(new_selection(short_id)) if len(new_selection(short_id)) > 0 else ""

        # Make a new link with the value either added or subtracted
        new_href = lambda short_id: self.format_href(self.args.copy(),  self.dataset_id,  **{key: new_key(short_id)})

        return [
            dbc.Button(
                label, 
                disabled=True, 
                color="light",
                style={'marginTop': '15px'},
                block=True,
            ),
            dbc.DropdownMenu(
                label="Add / Remove",
                children=[
                    dbc.DropdownMenuItem(
                        long_name,
                        href=new_href(short_id),
                        active=short_id in curr
                    )
                    for short_id, long_name in options.items()
                ],
                style={'marginTop': '5px'}
            )
        ]

    def dropdown_menu(
        self, 
        label="Label", 
        options={},
        key=""
    ):
        # Show a label which also includes the current selection,
        # and present a dropdown menu of links
        # Clicking each link changes which value is selected
        return [
            dbc.Button(
                label, 
                disabled=True, 
                color="light",
                style={'marginTop': '15px'},
                block=True,
            ),
            dbc.DropdownMenu(
                label=options.get(self.args.get(key), "None Selected"),
                children=[
                    dbc.DropdownMenuItem(
                        long_name,
                        href=self.format_href(self.args.copy(), self.dataset_id, **{key: short_id})
                    )
                    for short_id, long_name in options.items()
                ],
                style={'marginTop': '5px'}
            )
        ]

    def cag_metric_dropdown(
        self,
        label="Metric",
        key="cag_metric",
        options=dict(
            size="Size",
            entropy="Entropy",
            mean_abundance="Mean Abund.",
            prevalence="Prevalence",
            std_abundance="Std. Abund."
        )
    ):
        return self.dropdown_menu(
            label=label, options=options, key=key
        )

    def on_off_dropdown(
        self,
        label=None,
        key=None,
        options=dict(
            on="On",
            off="Off"
        )
    ):
        assert label is not None
        assert key is not None
        return self.dropdown_menu(
            label=label, options=options, key=key
        )

    def input_field(
        self,
        label="Label",
        key="NAME_OF_KEY",
        default_value="value",
        variable_type=None,
        input_type="text",
        placeholder="Please enter value",
        description="Description of input field",
        min_value=None,
        max_value=None,
    ):
        # Figure out what value to display
        # If the value is set in self.args, use that. Otherwise, use the default
        value = self.args.get(key, default_value)
        
        # If the variable type is set, apply that
        if variable_type is not None:
            value = variable_type(value)


        # Provide a text input field with an "Apply" button
        # that links to the href with this field added to the search string
        return [
            dbc.FormGroup(
                [
                    dbc.Button(
                        label, 
                        disabled=True, 
                        color="light",
                        style={'marginTop': '15px'},
                        block=True,
                    ),
                    dbc.Input(
                        type=input_type,
                        id={"name": "input_field", "key": key},
                        placeholder=placeholder,
                        debounce=False,
                        value=value,
                        min=min_value,
                        max=max_value,
                        style={'marginTop': '5px'},
                    ),
                    dbc.Button(
                        "Apply",
                        id={"name": "apply_button", "key": key},
                    ),
                    dbc.FormText(
                        description,
                        color="secondary",
                        style={'marginTop': '2px'}
                    )
                ]
            )
        ]

    def format_plot_id(self, plot_name):
        """Given a plot_name, return an ID which is compatible with the plotting callback."""

        # Add this plot name to the list of plots for this particular card
        self.plot_list.append(plot_name)

        return {"name": plot_name, "type": "analysis-card-plot"}

    def plot_div(self, plot_name):
        """Given a plot name, wrap around a spinner and Graph object."""
        # Also, the editable flag on the graph object will be set by the arguments from the search string
        return dbc.Spinner(
            html.Div(
                    dcc.Graph(
                        self.format_plot_id(plot_name),
                        config=dict(editable = self.args.get("editable", "false") == "true")
                    )
                )
            )


#################
# RICHNESS CARD #
#################
class RichnessCard(AnalysisCard):

    def __init__(self):

        self.long_name = "Richness"
        self.description = "Number of genes detected in each specimen"
        self.short_name = "specimen_summary"
        self.plot_list = []
        self.defaults = dict(
            metric="n_genes_assembled",
            plot_type="scatter",
            metadata="none",
            log_x="on"
        )
        self.dynamic_defaults = dict()
        self.help_text = """
In order to perform gene-level metagenomic analysis, the first step is to 
estimate the abundance of every microbial gene in every sample.

The analytical steps needed to perform this analysis include:

- Removal of host sequences by subtractive alignment
- _De novo_ assembly of every individual sample
- Deduplication of protein coding sequences across all samples to form a _de novo_ gene catalog
- Alignment of WGS reads from every sample against that gene catalog
- Estimation of the relative abundance of every gene as the proportion of reads which align uniquely (normalized by gene length)

To visualize how genes were quantified in each sample, you may select:

- The proportion of reads from each sample which align uniquely to a single protein-coding gene from the catalog
- The number of genes which were identified by _de novo_ assembly in each sample
- The number of genes which were identified by alignment against the non-redundant gene catalog generated by _de novo_ assembly

Data may be summarized either by a single point per sample or with a frequency histogram.

To mask any sample from this plot, deselect it in the manifest at the bottom of the page.

Note: Click on the camera icon at the top of this plot (or any on this page) to save an image to your computer.
        """

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Read the manifest in order to render the card
        manifest_df = glam_io.get_manifest(dataset_uri)

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            dbc.Row([
                dbc.Col(
                    # The plot name used below corresponds to a plot defined in GLAMPlotting                        
                    self.plot_div("richness-graph"),
                    align="center",
                    width=8,
                ),
                dbc.Col(
                    self.dropdown_menu(
                        label="Display Values",
                        options={
                            "n_genes_assembled": "Genes Assembled (#)",
                            "n_genes_aligned": "Genes Aligned (#)",
                            "pct_reads_aligned": "Reads Aligned (%)",
                        },
                        key="metric",
                    ) + self.dropdown_menu(
                        label="Plot Type",
                        options={
                            "scatter": "Scatter Plot",
                            "hist": "Histogram",
                        },
                        key="plot_type",
                    ) + self.dropdown_menu(
                        label="Color By",
                        options={
                            f: f
                            for f in ["none"] + [f for f in manifest_df.columns.values if f != "Unnamed: 0"]
                        },
                        key="metadata",
                    ) + self.dropdown_menu(
                        label="Number of Reads - Log Scale",
                        options={
                            "on": "On",
                            "off": "Off",
                        },
                        key="log_x",
                    )
                )
            ]),
        )


###########################
# EXPERIMENT SUMMARY CARD #
###########################
class ExperimentSummaryCard(AnalysisCard):

    def __init__(self):

        self.long_name = "Experiment Overview"
        self.description = "Summary metrics for entire analysis"
        self.short_name = "experiment_summary"
        self.plot_list = []
        self.defaults = dict()
        self.dynamic_defaults = dict()
        self.help_text = """
### Experiment Summary

An experiment consists of a collection of metagenomic WGS files which
each represent a physical specimen containing a mixture of microbes.
The experimental design is described using a manifest or sample sheet
which annotates the specimens which are, for example, treatment vs.
controls.

The metrics displayed in this panel are as follows:

* Total Reads: The number of reads across all FASTQ input files (after filtering host)
* Aligned Reads: The proportion of reads which were aligned uniquely to any gene in the catalog
* Genes: The number of unique genes assembled _de novo_ across all specimens
* CAGs: The number of gene groups identified by clustering the gene catalog by co-abundance across this collection of specimens
* Formula: If specified, a formula capturing the comparison of specimens intended with this experimental design
"""

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Parse the search string, while setting default arguments for this card
        # (while there are no arguments here, we need this to keep n={})
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            html.Div(id={"name": "experiment-summary", "index": 0}),
        )

######################
# SINGLE SAMPLE CARD #
######################
class SingleSampleCard(AnalysisCard):

    def __init__(self):

        self.long_name = "Single Sample"
        self.description = "Abundance of CAGs in a single sample"
        self.short_name = "single_sample"
        self.plot_list = []
        self.defaults = dict(
            display_metric="prop",
            compare_to="cag_size",
            sample="none",
        )
        self.dynamic_defaults = dict()
        self.help_text = """
**Summary of the CAGs detected in a single sample.**

You may select any sample to display the relative abundance of every CAG which
was detected, comparing by default against the size (number of genes) in each CAG. 

Optionally, you may choose to compare the abundance of CAGs in a pair of samples.

Abundance metrics:

- Relative Abundance (default): The proportion of gene copies detected in a sample which are assigned to each CAG
- Centered Log-Ratio: The log10 transformed relative abundance of each CAG divided by the geometric mean of relative abundance of all CAGs detected in that sample

Note: Click on the camera icon at the top of this plot to save an image to your computer.
            """

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Read the manifest in order to render the card
        manifest_df = glam_io.get_manifest(dataset_uri)

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # If there is no default sample, use the first one in the list
        if self.args["sample"] == "none":
            self.args["sample"] = manifest_df.index.values[0]

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            dbc.Row([
                dbc.Col(
                    # The plot name used below corresponds to a plot defined in GLAMPlotting                        
                    self.plot_div("single-sample-graph"),
                    align="center",
                    width=8,
                ),
                dbc.Col(
                    self.dropdown_menu(
                        label='Select Sample',
                        options={
                            specimen_name: specimen_name
                            for specimen_name in manifest_df.index.values
                        },
                        key="sample"
                    ) + self.dropdown_menu(
                        label="Compare To",
                        options={
                            **{
                                specimen_name: specimen_name
                                for specimen_name in manifest_df.index.values
                            },
                            **{
                                "cag_size": "CAG Size"
                            }
                        },
                        key="compare_to"
                    ) + self.dropdown_menu(
                        label="Display Metric",
                        options={
                            "prop": "Relative Abundance",
                            "clr": "Centered Log-Ratio"
                        },
                        key="display_metric"
                    )
                )
            ]),
        )


###################
# ORDINATION CARD #
###################
class OrdinationCard(AnalysisCard):

    def __init__(self):

        self.long_name = "Community Ordination"
        self.description = "Comparison of specimens by overall composition (e.g. PCA)"
        self.short_name = "ordination"
        self.plot_list = []
        self.defaults = dict(
            distance_metric = "braycurtis",
            ordination_algorithm = "pca",
            primary_pc = "1",
            secondary_pc = "2",
            perplexity = 30
        )
        self.dynamic_defaults = dict()
        self.help_text = """
**Beta-diversity summary of the similarity of community composition across samples.**

One way to understand the composition of a microbial community is to compare every pair of samples
on the basis of what microbes were detected. This approach is referred to as 'beta-diversity' analysis.

You may select a distance metric which is used to calculate the similarity of every pair of samples.
Based on that distance matrix, PCA or t-SNE may be used to summarize the groups of samples which
are most similar to each other.

By moving the sliders, you may select different composite indices to display on the plot.

The upper plot shows the frequency histogram of samples across the first composite axis.
The lower scatter plot shows two axes, and metadata from the manifest can be overlaid as a color on each point.

To mask any sample from this plot, deselect it in the manifest at the bottom of the page.

Note: Click on the camera icon at the top of this plot (or any on this page) to save an image to your computer.
    """

    def ordination_settings(self):
        """Depending on whether PCA or t-SNE is selected, provide the appropriate controls."""
        if self.args["ordination_algorithm"] == "pca":
            return self.dropdown_menu(
                label = "Primary Axis",
                options = {
                    str(v): str(v)
                    for v in range(1, 11)
                },
                key = "primary_pc"
            ) + self.dropdown_menu(
                label = "Secondary Axis",
                options = {
                    str(v): str(v)
                    for v in range(1, 11)
                },
                key = "secondary_pc"
            )
            
        else:
            assert self.args["ordination_algorithm"] == "tsne", self.args

            return self.input_field(
                label="Perplexity",
                key="perplexity",
                default_value=self.defaults["perplexity"],
                input_type="number",
                max_value=100,
                min_value=0,
                description="Adjust the degree of clustering by t-SNE"
            )

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Read the manifest in order to render the card
        manifest_df = glam_io.get_manifest(dataset_uri)

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            dbc.Row([
                dbc.Col(
                    # The plot name used below corresponds to a plot defined in GLAMPlotting                            
                    self.plot_div("ordination-graph"),
                    width=8,
                    align="center"
                ),
                dbc.Col(
                    self.dropdown_menu(
                        label='Distance Metric',
                        options = {
                            'euclidean': 'Euclidean',
                            'aitchison': 'Aitchison',
                            'braycurtis': 'Bray-Curtis',
                        },
                        key = "distance_metric"
                    ) + self.dropdown_menu(
                        label = "Ordination Method",
                        options = {
                            'pca': 'PCA',
                            'tsne': 't-SNE',
                        },
                        key = "ordination_algorithm"
                    ) + self.ordination_settings(
                    ) + self.dropdown_menu(
                        label = "Color By",
                        options = {
                            "none": "None",
                            **{
                                k: k
                                for k in manifest_df.columns.values
                            }
                        },
                        key="color_by",
                    ) + [
                        html.Div(
                            id={"name": "ordination-anosim-results", "index": 0}
                        )
                    ],
                    width=4,
                    align="center"
                )
            ]),
)

####################
# CAG SUMMARY CARD #
####################


class CAGSummaryCard(AnalysisCard):

    def __init__(self):

        self.long_name = "CAG Characteristics"
        self.description = "Summary metrics describing all CAGs (e.g. size)"
        self.short_name = "cag_summary"
        self.plot_list = []
        self.defaults = dict(
            cag_metric = "size",
            histogram_metric = "genes",
            histogram_log = "on",
            histogram_nbins = 50,
        )
        self.dynamic_defaults = dict()
        self.help_text = """
A key factor in performing efficient gene-level metagenomic analysis is the grouping of genes by co-abundance.
The term 'co-abundance' is used to describe the degree to which any pair of genes are found at similar relative
abundances in similar samples. The core concept is that any pair of genes which are always found on the same
molecule of DNA are expected to have similar relative abundances as measured by WGS metagenomic sequencing.

In this analysis, genes were grouped into Co-Abundant Groups (CAGs) by average linkage clustering using the
cosine measure of co-abundance across every pair of samples. After constructing CAGs for this dataset, each
of those CAGs can be summarized on the basis of the aggregate abundance of all genes contained in that CAG.
(note: every gene can only belong to a single CAG in this approach).

The biological interpretation of CAGs is that they are expected to correspond to groups of genes which are
consistently found on the same piece of genetic material (chromosome, plasmid, etc.), or that they are found
in organismsm which are highly co-abundant in this dataset.

This panel summarizes that set of CAGs on the basis of:

- Size: Number of genes contained in each CAG
- Mean Abundance across all samples (as the sum of the relative abundance of every gene in that CAG)
- Entropy: The evenness of abundance for a given CAG across all samples
- Prevalence: The proportion of samples in which a CAG was detected at all
- Standard Deviation of relative abundance values across all samples

Note: Masking a sample from the manifest at the bottom of the page does _not_ update the summary CAG metrics displayed here.

By default, this frequency histogram displays the number of _genes_ found in the group of CAGs that fall
within a given range. You may instead choose to display the number of CAGs which fall into each bin.

Note: Click on the camera icon at the top of this plot (or any on this page) to save an image to your computer.
    """

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            [
                dbc.Row([
                    dbc.Col(
                        # The plot name used below corresponds to a plot defined in GLAMPlotting
                        self.plot_div("cag-descriptive-stats-graph"),
                        width=8,
                        align="center"
                    ),
                    dbc.Col(
                        self.cag_metric_dropdown(
                        ) + self.dropdown_menu(
                            label = "Histogram Display",
                            options = dict(
                                genes = 'Number of genes',
                                cags = 'Number of CAGs',
                            ),
                            key = "histogram_metric"
                        ) + self.on_off_dropdown(
                            label = "Histogram Log Scale",
                            key = "histogram_log"
                        ) + self.input_field(
                            label = "Number of Bins",
                            key = "histogram_nbins",
                            default_value = self.defaults["histogram_nbins"],
                            input_type = "number",
                            placeholder = "Enter number of bins",
                            description = "",
                            min_value = 10,
                            max_value = 100,
                        ),
                        width=4,
                        align="center"
                    )
                ])
            ],

        )
######################
# / CAG SUMMARY CARD #
######################

##############################
# CAG ABUNDANCE HEATMAP CARD #
##############################
class CagAbundanceHeatmap(AnalysisCard):

    def __init__(self):

        self.long_name = "CAG Abundance Heatmap"
        self.description = "Relative abundance of multiple CAGs across all specimens"
        self.short_name = "cag_abund_heatmap"
        self.plot_list = []
        self.defaults = dict(
            select_cags_by = "abundance",
            ncags = 20,
            min_cag_size = 5,
            max_cag_size = 0,
            metadata_field = "",
            group_by = "cag",
            metric = "log10",
            annot_tax = "none",
        )
        self.dynamic_defaults = dict()
        self.help_text="""
This display lets you compare the relative abundance of a group of CAGs across all samples.
You may choose to view those CAGs which are most highly abundant, those CAGs containing the
largest number of genes, or those CAGs which are most consistently associated with a parameter
in your formula (if provided).

If you decide to display those CAGs which are most associated with a parameter in the formula,
then you will see the estimated coefficient of association for each CAG against that parameter
displayed to the right of the heatmap.

You may also choose to display the taxonomic annotation of each CAG at a particular taxonomic
level (e.g. species). That will add a color label to each row in the heatmap, and you can see
the name of the organism that represents by moving your mouse over that part of the plot.

The controls at the top of the display help you customize this heatmap. You may choose to include
a greater or smaller number of CAGs; you may choose to filter CAGs based on their size (the
number of genes in each CAG); and you may choose to annotate the samples based on user-defined
metadata from the manifest.

By default, the columns in the heatmap are ordered based on the similarity of CAG abundances
(average linkage clustering), but you may also choose to set the order according to the sorted
metadata for each sample.

Note: Click on the camera icon at the top of this plot (or any on this page) to save an image to your computer.
    """

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Read the manifest in order to render the card
        manifest_df = glam_io.get_manifest(dataset_uri)

        # Get the list of parameters available
        parameter_list = glam_io.get_parameter_list(dataset_uri)

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            [
                dbc.Row([
                    dbc.Col(
                        # Wrapper around the spinner and 'editable' functionality for a given graph
                        self.plot_div(
                            # The plot ID used below corresponds to a plot defined in GLAMPlotting
                            "cag-abund-heatmap"
                        )
                    )
                ]),
                dbc.Row([
                    dbc.Col(
                        self.dropdown_menu(
                            label = "Select CAGs By",
                            options = {
                                "abundance": "Average Relative Abundance",
                                "size": "Size (Number of Genes)",
                                **{
                                    f"parameter::{p}": f"Association with {p}"
                                    for p in parameter_list
                                }
                            },
                            key = "select_cags_by"
                        ) + self.input_field(
                            label = "Number of CAGs to Display",
                            key = "ncags",
                            input_type = "number",
                            variable_type = int,
                            min_value = 5,
                            max_value = 100,
                            description = ""
                        ),
                        width = 4,
                    ),
                    dbc.Col(
                        self.input_field(
                            label = "Minimum CAG size",
                            description = "Only display CAGs which contain at least this number of genes",
                            key = "min_cag_size",
                            input_type = "number",
                            variable_type = int,
                            min_value = 1,
                            max_value = 100000,
                        ) + self.input_field(
                            label = "Maximum CAG size",
                            description = "Only display CAGs which contain at most this number of genes (0 for no limit)",
                            key = "max_cag_size",
                            input_type = "number",
                            variable_type = int,
                            min_value = 0,
                            max_value = 100000,
                        ),
                        width = 4,
                    ),
                    dbc.Col(
                        self.multiselector(
                            label = "Display Metadata",
                            key = "metadata_field",
                            options = {
                                **{
                                    col_name: col_name
                                    for col_name in manifest_df.columns.values
                                }
                            }
                        ) + self.dropdown_menu(
                            label = "Group Specimens",
                            key = "group_by",
                            options = dict(
                                cag = "By CAG Abundances",
                                metadata = "By Metadata"
                            )
                        ) + self.dropdown_menu(
                            label = "Abundance Metric",
                            key = "metric",
                            options = dict(
                                log10 = "Prop. Abund. (log10)",
                                zscore = "Prop. Abund. (log10) (z-score)",
                                raw = "Prop. Abund."
                            )
                        ) + self.dropdown_menu(
                            label = "Taxonomic Annotation",
                            key = "annot_tax",
                            options = {
                                "none": "None",
                                "species": "Species",
                                "genus": "Genus",
                                "family": "Family",
                                "class": "Class",
                                "phylum": "Phylum",
                            }
                        ),
                        width = 4,
                        align = "center",
                    )
                ])
            ],
)
################################
# / CAG ABUNDANCE HEATMAP CARD #
################################

###############################
# CAG ANNOTATION HEATMAP CARD #
###############################
class AnnotationHeatmapCard(AnalysisCard):

    def __init__(self):

        self.long_name = "CAG Annotation Heatmap"
        self.description = "Annotations of genes across multiple CAGs (e.g. taxonomy)"
        self.short_name = "cag_annot_heatmap"
        self.plot_list = []
        self.defaults = dict(
            select_cags_by="abundance",
            ncags=20,
            min_cag_size=5,
            max_cag_size=0,
            annot_type="taxonomic",
            nannots=40,
        )
        self.dynamic_defaults = dict()
        self.help_text = """
This display lets you compare the taxonomic or functional annotations of a group of CAGs.

You may choose to view those CAGs which are most highly abundant, those CAGs containing the
largest number of genes, or those CAGs which are most consistently associated with a parameter
in your formula (if provided).

If you decide to display those CAGs which are most associated with a parameter in the formula,
then you will see the estimated coefficient of association for each CAG against that parameter
displayed to the right of the heatmap. In addition, you will see the aggregate association of
each selected annotation against that same parameter from the formula.

The controls at the top of the display help you customize this heatmap. You may choose to include
a greater or smaller number of CAGs; you may choose to filter CAGs based on their size (the
number of genes in each CAG); and you may choose to display either taxonomic or functional annotations.

Note: Click on the camera icon at the top of this plot (or any on this page) to save an image to your computer.
        """

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Get the list of parameters available
        parameter_list = glam_io.get_parameter_list(dataset_uri)

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        # Set up the annotation options available for this dataset
        annot_type_options = {
            "taxonomic": "Taxonomic",
            "species": "Species",
            "genus": "Genus",
            "family": "Family",
        }

        # If this dataset has genome annotations, add that option
        if glam_io.has_genomes(dataset_uri):
            annot_type_options["genomes"] = "Genomes"

        # If this dataset has functional annotations, add that option
        if glam_io.has_functional_annotations(dataset_uri):
            annot_type_options["eggNOG_desc"] = "Functional"

        return self.card_wrapper(
            dataset_id,
            [
                dbc.Row([
                    dbc.Col(
                        # Wrapper around the spinner and 'editable' functionality for a given graph
                        self.plot_div(
                            # The plot ID used below corresponds to a plot defined in GLAMPlotting
                            "cag-annot-heatmap"
                        )
                    )
                ]),
                dbc.Row([
                    dbc.Col(
                        self.dropdown_menu(
                            label="Select CAGs By",
                            options={
                                "abundance": "Average Relative Abundance",
                                "size": "Size (Number of Genes)",
                                **{
                                    f"parameter::{p}": f"Association with {p}"
                                    for p in parameter_list
                                }
                            },
                            key="select_cags_by"
                        ) + self.input_field(
                            label="Number of CAGs to Display",
                            description="",
                            key="ncags",
                            input_type="number",
                            variable_type=int,
                            min_value=5,
                            max_value=100,
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        self.dropdown_menu(
                            label="Annotation Type",
                            options=annot_type_options,
                            key="annot_type"
                        ) + self.input_field(
                            label="Number of Annotations to Display",
                            description="",
                            key="nannots",
                            input_type="number",
                            variable_type=int,
                            min_value=5,
                            max_value=100,
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        self.input_field(
                            label="Minimum CAG size",
                            description="Only display CAGs which contain at least this number of genes",
                            key="min_cag_size",
                            input_type="number",
                            variable_type=int,
                            min_value=1,
                            max_value=100000,
                        ) + self.input_field(
                            label="Maximum CAG size",
                            description="Only display CAGs which contain at most this number of genes (0 for no limit)",
                            key="max_cag_size",
                            input_type="number",
                            variable_type=int,
                            min_value=0,
                            max_value=100000,
                        ),
                        width=4,
                        align="center",
                    )
                ])
            ],
        )

################
# VOLCANO PLOT #
################
class VolcanoCard(AnalysisCard):

    def __init__(self):

        self.long_name = "Volcano Plot"
        self.description = "Association of CAGs with experimental parameters"
        self.short_name = "volcano"
        self.plot_list = []
        self.defaults = dict(
            min_cag_size=5,
            max_cag_size=0,
            max_pvalue=0.95,
            fdr="on",
            compare_against="coef",
        )

        # The default parameter name needs to be set using a function
        # that takes a GLAM_IO and dataset_uri as inputs
        self.dynamic_defaults = dict(
            parameter=lambda glam_io, dataset_uri: glam_io.get_parameter_list(dataset_uri)[0]
        )

        self.help_text = """
The estimated association of each CAG with a specified metadata feature is displayed
as a volcano plot. The values shown in this display must be pre-computed by selecting
the `--formula` flag when running _geneshot_.

Note: Click on the camera icon at the top of this plot (or any on this page) to save an image to your computer.
        """

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Get the list of parameters available
        parameter_list = glam_io.get_parameter_list(dataset_uri)

        # Set the default parameter to the first one in the list
        if self.defaults.get("parameter") is None:
            self.defaults["parameter"] = parameter_list[0]

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            [
                dbc.Row([
                    # Left-hand column
                    dbc.Col(
                        # Just contains the plot
                        self.plot_div(
                            # The plot ID used below corresponds to a plot defined in GLAMPlotting
                            "volcano-plot"
                        ),
                        width = 8,
                    ),
                    dbc.Col(
                        self.dropdown_menu(
                            label="Select Parameter",
                            options={
                                parameter: parameter
                                for parameter in parameter_list
                            },
                            key="parameter"
                        ) + self.input_field(
                            label="Minimum CAG size",
                            description="Only display CAGs which contain at least this number of genes",
                            key="min_cag_size",
                            input_type="number",
                            variable_type=int,
                            min_value=1,
                            max_value=100000,
                        ) + self.input_field(
                            label="Maximum CAG size",
                            description="Only display CAGs which contain at most this number of genes (0 for no limit)",
                            key="max_cag_size",
                            input_type="number",
                            variable_type=int,
                            min_value=0,
                            max_value=100000,
                        ) + self.input_field(
                            label="Maximum p-value",
                            description="Only display CAGs with p-values below this value",
                            key="max_pvalue",
                            input_type="text",
                            variable_type=float,
                            min_value=0,
                            max_value=1,
                        ) + self.dropdown_menu(
                            label="FDR-BH Adjustment",
                            options={
                                "on": "On",
                                "off": "Off"
                            },
                            key="fdr"
                        ) + self.dropdown_menu(
                            label="Compare Against",
                            options={
                                "coef": "Estimated Coefficient",
                                **{
                                    f"parameter::{parameter}": parameter
                                    for parameter in parameter_list
                                    if parameter != self.args["parameter"]
                                }
                            },
                            key="compare_against"
                        ),
                        width = 4,
                    )
                ])
            ],
        )

#################
# PLOT CAG CARD #
#################
class PlotCagCard(AnalysisCard):

    def __init__(self):

        self.long_name = "Single CAG Plot"
        self.description = "Abundance of a single CAG across all specimens"
        self.short_name = "plot_cag"
        self.plot_list = []
        self.defaults = dict(
            plot_type = "scatter",
            log10 = "on",
            color_by = None,
            facet = None,
        )
        self.dynamic_defaults = dict(
            cag_id = lambda glam_io, dataset_uri: glam_io.get_cag_annotations(
                dataset_uri
            ).sort_values(
                by="mean_abundance"
            ).index.values[-1],
            xaxis = lambda glam_io, dataset_uri: [
                col_name
                for col_name in glam_io.get_manifest(dataset_uri).columns.values
                if col_name not in ["R1", "R2"]
            ][0]
        )

        self.help_text = """
Construct a summary of the abundance of a single CAG in relation to the metadata
assigned to each specimen. By selecting different types of plots, you may flexibly
construct any type of summary display.
        """

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Read the manifest in order to render the card
        manifest_df = glam_io.get_manifest(dataset_uri)
        manifest_options = {
            col_name: col_name
            for col_name in manifest_df.columns.values
            if col_name not in ["R1", "R2"]
        }

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            [
                dbc.Row([
                    dbc.Col(
                        self.plot_div(
                            "single-cag-graph"
                        ),
                        width=8
                    ),
                    dbc.Col(
                        self.input_field(
                            label="CAG ID",
                            description="Select a CAG to display",
                            key="cag_id",
                            input_type="number",
                            variable_type=int,
                            min_value=0,
                            max_value=250000,
                        ) + self.dropdown_menu(
                            label="X-axis",
                            key="xaxis",
                            options=manifest_options
                        ) + self.dropdown_menu(
                            label="Plot Type",
                            key="plot_type",
                            options={
                                "scatter": "Points",
                                "line": "Line",
                                "boxplot": "Boxplot",
                                "strip": "Stripplot",
                            }
                        ) + self.dropdown_menu(
                            label="Color",
                            key="color_by",
                            options=manifest_options
                        ) + self.dropdown_menu(
                            label="Facet",
                            key="facet",
                            options=manifest_options
                        ) + self.dropdown_menu(
                            label="Log Scale",
                            key="log10",
                            options={
                                "on": "On",
                                "off": "Off"
                            }
                        ),
                        width=4
                    )
                ])
            ],
        )


# ###########################
# # GENOME ASSOCIATION CARD #
# ###########################
class GenomeAssociationCard(AnalysisCard):

    def __init__(self):

        self.long_name = "Genome Association Scatterplot"
        self.description = "Summary of genome-level association with experimental parameters"
        self.short_name = "genome_association"
        self.plot_list = []
        self.defaults = dict(
            compare_to="mean_est_coef"
        )
        self.dynamic_defaults = dict(
            parameter=lambda glam_io, dataset_uri: glam_io.get_genome_parameters(dataset_uri)[0]
        )

        self.help_text = """
After aligning genes against a set of genomes, we are able to
annotate genomes on the basis of how many genes align which
belong to CAGs which are associated with any of the parameters
in the associated experiment. 

Operationally, the user sets a cuttoff _alpha_ value when they
execute the genome alignment module which generated this dataset.
That _alpha_ value is used as a cutoff for this genome summary display
in which any CAG with an FDR-BH adjusted p-value below that threshold
are annotated as "highly associated" with that parameter. In order
to summarize the genomes, we can then display the number (or proportion)
of genes in the genome which are "highly associated."

While this approach is somewhat arbitrary, it can be used to quickly
identify those genomes which may contain the genetic elements that
are most strongly associated with a phenotype of interest.

Having identified a genome of interest, the individual gene alignments
can be inspected in the Genome Alignment Details panel of this browser.
        """

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Get the list of parameters available
        parameter_list = glam_io.get_parameter_list(dataset_uri)

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            [
                dbc.Row([
                    dbc.Col(
                        self.plot_div(
                            "genome-association-scatterplot"
                        ),
                        width=8
                    ),
                    dbc.Col(
                        self.dropdown_menu(
                            label="Experiment Parameter",
                            options={
                                **{
                                    parameter: parameter
                                    for parameter in parameter_list
                                }
                            },
                            key="parameter"
                        ) + self.dropdown_menu(
                            label="Compare Against",
                            options={
                                "mean_est_coef": "Mean Estimated Coefficient",
                                **{
                                    parameter: f"Mean Wald: {parameter}"
                                    for parameter in parameter_list
                                }
                            },
                            key="compare_to"
                        ),
                        width=4
                    )
                ])
            ],
        )

##########################
# TAXONOMY SUNBURST CARD #
##########################
class TaxonomySunburstCard(AnalysisCard):

    def __init__(self):

        self.long_name = "Single CAG Taxonomy"
        self.description = "Taxonomic assignment of genes within a single CAG"
        self.short_name = "taxonomy_sunburst"
        self.plot_list = []
        self.defaults = dict()
        self.dynamic_defaults = dict(
            cag_id = lambda glam_io, dataset_uri: glam_io.get_cag_annotations(
                dataset_uri
            ).sort_values(
                by="mean_abundance"
            ).index.values[-1]
        )

        self.help_text = """
The taxonomic annotation of a given CAG is shown as the proportion of
genes which contain a given taxonomic annotation, out of all genes which
were given any taxonomic annotation.
        """

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            [
                dbc.Row([
                    dbc.Col(
                        self.plot_div(
                            "tax-sunburst"
                        ),
                        width=8
                    ),
                    dbc.Col(
                        self.input_field(
                            label="CAG ID",
                            description="Select a CAG to display",
                            key="cag_id",
                            input_type="number",
                            variable_type=int,
                            min_value=0,
                            max_value=250000,
                        ),
                        width=4
                    )
                ])
            ],
        )

###################################
# GENOME CONTAINMENT HEATMAP CARD #
###################################
class GenomeContainmentHeatmap(AnalysisCard):

    def __init__(self):

        self.long_name = "Genome Containment Heatmap"
        self.description = "Genomes containing genes from a single CAG"
        self.short_name = "genome_containment_heatmap"
        self.plot_list = []
        self.defaults = dict(
            genome_n = 20,
            cag_n = 20,
        )
        self.dynamic_defaults = dict(
            cag_id = lambda glam_io, dataset_uri: glam_io.get_top_genome_containment(
                dataset_uri
            ).index.values[0]
        )

        self.help_text = """
Every CAG was aligned against a collection of reference microbial
genomes. To summarize the similarity of each CAG to a given genome,
we can calculate the proportion of genes from that CAG which align
to a given genome (given the stringency of alignment parameters
required by the user).

In order to show those genomes which most closely resemble a selected
CAG, we will display the genomes with the largest number of aligning
genes. For context, we will also display those CAGs which have similarity
to that set of selected genomes.
        """

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            [
                dbc.Row([
                    dbc.Col(
                        self.plot_div(
                            "genome-containment-heatmap"
                        ),
                    )
                ]),
                dbc.Row([
                    dbc.Col(
                        self.input_field(
                            label="CAG ID",
                            description="Select a CAG to display",
                            key="cag_id",
                            input_type="number",
                            variable_type=int,
                            min_value=0,
                            max_value=250000,
                        ),
                        width=4
                    ),
                    dbc.Col(
                        self.input_field(
                            label="Number of Genomes",
                            description="Number of genomes to display",
                            key="genome_n",
                            input_type="number",
                            variable_type=int,
                            min_value=1,
                            max_value=150,
                        ),
                        width=4
                    ),
                    dbc.Col(
                        self.input_field(
                            label="Number of CAGs",
                            description="Number of CAGs to display",
                            key="cag_n",
                            input_type="number",
                            variable_type=int,
                            min_value=1,
                            max_value=150,
                        ),
                        width=4
                    )
                ])
            ],
        )


###########################
# GENOME ALIGNMENTS CARD #
###########################
class GenomeAlignmentsCard(AnalysisCard):

    def __init__(self):

        self.long_name = "Genome Alignments"
        self.description = "Map of gene alignments to a single genome"
        self.short_name = "genome_alignments"
        self.plot_list = []
        self.defaults = dict(
            parameter="",
            window_size="50000"
        )
        self.dynamic_defaults = dict(
            # By default, show the genome with the most genes aligning to a single CAG
            genome_id=lambda glam_io, dataset_uri: glam_io.get_genomes_with_details(dataset_uri)[0]
        )

        self.help_text = """
Detailed gene-level alignments are shown here for individual genomes,
including details on the estimated association of genes with any parameter
of interest. 

After selecting a genome from the drop-down menu, the detailed list of
alignments for that genome will be shown in the table. Clicking on any
row in that table will then center the display on that gene. The amount
of the genome shown in the display can be adjusted with the radio button
below the plot, but individual gene alignments will not be shown above
a certain window size (due to overplotting).

When a parameter is selected the estimated coefficients of association
for each CAG are shown in the table and a rolling window median Wald
statistic is added in the plot.
        """

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Get the list of genomes available for plotting
        genome_id_list = glam_io.get_genomes_with_details(dataset_uri)

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Get the details of all aligned genes for the selected genome
        genome_aln_df = glam_io.get_genome_details(
            dataset_uri,
            self.args["genome_id"]
        )

        # Get the name for each genome from the manifest
        genome_manifest = glam_io.get_genome_manifest(dataset_uri).set_index("id")
        # Sort the genome options alphabetically by name
        genome_options = OrderedDict()
        for genome_id, genome_name in genome_manifest.reindex(
            index=genome_id_list
        ).sort_values(
            by="name"
        )["name"].iteritems():
            genome_options[genome_id] = genome_name

        # Get the options for the parameters which can be displayed
        parameter_options = OrderedDict()
        if glam_io.has_genome_parameters(dataset_uri):
            for p in glam_io.get_genome_parameters(dataset_uri):
                parameter_options[p] = p
        else:
            parameter_options["none"] = "None"

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            [
                dbc.Row([
                    dbc.Col([], width=3),
                    dbc.Col(
                        self.dropdown_menu(
                            label="Genome",
                            options=genome_options,
                            key="genome_id"
                        ),
                        width=6,
                        align="center"
                    ),
                    dbc.Col([], width=3)
                ]),
                dbc.Row([
                    dbc.Col([], width=1),
                    dbc.Col(
                        html.Div(
                            dash_table.DataTable(
                                id='genome-details-table',
                                columns=[
                                    {"name": "Gene Name", "id": "gene"},
                                    {"name": "CAG", "id": "CAG"},
                                    {"name": "Contig", "id": "contig"},
                                    {"name": "Percent Identity", "id": "pident"},
                                    {"name": "Start Position", "id": "contig_start"},
                                    {"name": "End Position", "id": "contig_end"},
                                ],
                                data=genome_aln_df.to_dict(orient="records"),
                                row_selectable='single',
                                style_table={
                                    'minWidth': '100%',
                                },
                                style_header={
                                    "backgroundColor": "rgb(2,21,70)",
                                    "color": "white",
                                    "textAlign": "center",
                                },
                                page_action='native',
                                page_size=20,
                                filter_action='native',
                                sort_action='native',
                                hidden_columns=[],
                                css=[{"selector": ".show-hide",
                                        "rule": "display: none"}],
                            ),
                            style={"marginTop": "20px"}
                        ),
                        width=10,
                        align="center"
                    ),
                    dbc.Col([], width=1)
                ]),
                dbc.Row([
                    dbc.Col(
                        dbc.Spinner(
                            dcc.Graph(
                                id="genome-alignment-plot"
                            )
                        ),
                        width=12,
                        align="center"
                    )
                ]),
                dbc.Row([
                    dbc.Col([], width=2),
                    dbc.Col(
                        self.multiselector(
                            label="Show CAG Association By:",
                            options=parameter_options,
                            key="parameter"
                        ),
                        width=4,
                        align="center"
                    ),
                    dbc.Col(
                        self.dropdown_menu(
                            label="Window Size",
                            options=OrderedDict([
                                ("10000", '10kb'),
                                ("25000", '25kb'),
                                ("50000", '50kb'),
                                ("150000", '150kb'),
                                ("500000", '500kb'),
                                ("-1", 'All'),
                            ]),
                            key="window_size"
                        ),
                        width=4,
                        align="center"
                    ),
                    dbc.Col([], width=2),
                ])
            ],
        )


#################
# CARD TEMPLATE #
#################
class CardTemplate(AnalysisCard):

    def __init__(self):

        self.long_name = ""
        self.description = ""
        self.short_name = ""
        self.plot_list = []
        self.defaults = dict(

        )
        self.dynamic_defaults = dict()

        self.help_text = """
        """

    def card(self, dataset_id, dataset_uri, search_string, glam_io):

        # Read the manifest in order to render the card
        manifest_df = glam_io.get_manifest(dataset_uri)

        # Parse the search string, while setting default arguments for this card
        self.args = decode_search_string(
            search_string, **self.defaults
        )

        # Set the dataset ID (used to render card components)
        self.dataset_id = dataset_id

        return self.card_wrapper(
            dataset_id,
            [],
        )


