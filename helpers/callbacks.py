#!/usr/bin/env python3
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import dash
import json
from time import sleep
from .common import encode_search_string
from .common import decode_search_string

class GLAM_CALLBACKS:

    def __init__(self, glam_db, glam_io, glam_layout, glam_plotting):
        # Attach the GLAM_DB and GLAM_IO objects to this object
        self.glam_db = glam_db
        self.glam_io = glam_io
        self.glam_layout = glam_layout
        self.glam_plotting = glam_plotting

    def page_contents_children(
        self, 
        pathname, 
        search_string, 
        username, 
        password,
        login_modal_is_open
    ):
        # If the login modal is open, don't update
        if login_modal_is_open:
            raise PreventUpdate

        show_style = {"display": "block"}

        # Direct to the login page
        if pathname == "/login":

            # Page loaded in the background of the login modal
            return self.glam_layout.login_page_background(), show_style

        # Show public datasets to any user
        elif pathname == "/public":

            return self.glam_layout.dataset_list(
                self.glam_db.public_datasets(),
                username if self.glam_db.valid_username_password(username, password) else "Anonymous User"
            ), show_style

        # Check to see if the user is NOT logged in
        elif self.glam_db.valid_username_password(username, password) is False:

            # Page for users who are not logged in
            return self.glam_layout.logged_out_page(), show_style

        # User is logged in
        else:

            # Datasets available to this user
            if pathname == "/datasets" or pathname == "/":

                return self.glam_layout.dataset_list(
                    self.glam_db.user_datasets(username, password),
                    username
                ), show_style

            # Open a specific analysis
            elif pathname is not None and pathname.startswith("/d/") and "/a/" in pathname:

                # Parse the dataset name from the path
                dataset_id = pathname.split("/")[2]

                # Parse the analysis name
                analysis_name = pathname.rsplit("/", 1)[1]

                return self.glam_layout.analysis_page(
                    username, 
                    password, 
                    dataset_id, 
                    analysis_name, 
                    search_string
                ), show_style

            # Access to specific dataset
            elif pathname is not None and pathname.startswith("/d/") and pathname.endswith("/analysis"):

                # Parse the dataset name from the path
                dataset_id = pathname.split("/")[2]

                # Check to see if this user has permission to view this dataset
                if self.glam_db.user_can_access_dataset(
                    dataset_id,
                    username,
                    password
                ):
                    return self.glam_layout.dataset_display(username, dataset_id, search_string), show_style

        # catch-all else
        return self.glam_layout.page_not_found(), show_style

    # Function to toggle a modal
    def toggle_modal(self, n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

    # Function to get the ID of the element which triggered a callback
    def parse_callback_trigger(self, ctx):
        # Make sure that the callback was triggered
        assert ctx.triggered, ctx

        # Make sure that we have a trigger listed
        assert len(ctx.triggered) >= 1, ctx.triggered

        # Get the prop_id for the triggered element
        element_id = ctx.triggered[0]["prop_id"]

        # Return the full data, parsed from JSON
        # (note this only works for element IDs which are dicts)
        return json.loads(element_id.rsplit(".", 1)[0])

    # Function which decorates all of the callbacks needed by the app
    def decorate(self, app):

        # Open and close the login modal
        @app.callback(
            Output("login-modal", "is_open"),
            [
                Input({"type": "login-button", "parent": ALL}, "n_clicks"),
                Input("login-modal-apply-button", "n_clicks")
            ]
        )  # pylint: disable=unused-variable
        def login_modal_is_open(login_buttons, apply_button):
            # If the login button has never been pressed, keep the modal closed
            if all([v is None for v in login_buttons]):
                return False

            # Get the context which triggered the callback
            ctx = dash.callback_context

            # If a login button was the trigger
            if "login-button" in ctx.triggered[0]["prop_id"]:
                return True
            # Otherwise, we can assume that the apply button was pressed
            else:
                return False

        # Open and close the change-password modal
        @app.callback(
            Output("change-password-modal", "is_open"),
            [
                Input("change-password-button", "n_clicks"),
                Input("change-password-close-button", "n_clicks")
            ]
        )  # pylint: disable=unused-variable
        def change_password_modal_is_open(open_button, close_button):
            # If the change password button has never been pressed, keep the modal closed
            if open_button is None:
                return False

            # Get the context which triggered the callback
            ctx = dash.callback_context

            # Open the modal if the open button was pressed
            return ctx.triggered[0]["prop_id"] == "change-password-button.n_clicks"

        # Try to change the user's password
        @app.callback(
            Output("change-password-response-text", "children"),
            [
                Input("change-password-apply-button", "n_clicks")
            ],
            [
                State("change-password-username", "value"),
                State("change-password-old", "value"),
                State("change-password-new", "value"),
            ]
        )  # pylint: disable=unused-variable
        def change_password_response(apply_button, username, old_password, new_password):
            # If the apply password button has never been pressed, don't update
            print(apply_button)
            if apply_button is None or apply_button == 0:
                return ""

            # If the username and password is incorrect, stop
            if not self.glam_db.valid_username_password(username, old_password):
                return "Incorrect username / password"

            # Make sure that the password is of sufficient length
            if new_password is None or len(new_password) < 8:
                return "Please provide a new password with at least 8 characters"

            # Otherwise, try to change the password
            self.glam_db.set_password(
                username=username,
                password=new_password
            )
            return "Successfully changed password"


        # Fill in the page-contents while also controlling show/hide of the sub-navbar
        @app.callback(
            [
                Output("page-content", "children"),
                Output("sub-navbar", "style")
            ],
            [
                Input("url", "pathname"),
                Input("url", "search"),
                Input("username", "value"),
                Input("password", "value"),
                Input("login-modal", "is_open"),
            ],
        )  # pylint: disable=unused-variable
        def g(url, search_string, username, password, login_modal_is_open):
            return self.page_contents_children(
                url, 
                search_string, 
                username, 
                password, 
                login_modal_is_open
            )

        # Update the 'brand' of the sub-navbar with the username, if valid
        @app.callback(
            Output("sub-navbar", "brand"),
            [
                Input("username", "value"),
                Input("password", "value"),
            ],
        )  # pylint: disable=unused-variable
        def h(username, password):
            if self.glam_db.valid_username_password(username, password):
                return username
            else:
                return "Anonymous"

        ####################################
        # OPEN / CLOSE HELP MODAL CALLBACK #
        ####################################
        @app.callback(
            Output({"type": "help-text-modal", "parent": MATCH},"is_open"),
            [
                Input("open-help-text", "n_clicks"),
                Input({"type": "close-help-text","parent": MATCH}, "n_clicks")
            ],
            [State({"type": "help-text-modal","parent": MATCH}, "is_open")],
        )
        def j(n1, n2, is_open):
            return self.toggle_modal(n1, n2, is_open)

        ############################
        # COLLAPSE / SHOW MANIFEST #
        ############################
        @app.callback(
            Output("manifest-table-body", "is_open"),
            [
                Input("open-manifest", "n_clicks"),
            ],
            [
                State("manifest-table-body", "is_open"),
            ],
        )
        def k(n_clicks, is_open):
            if n_clicks is None or n_clicks == 0:
                return is_open
            else:
                return not is_open

        ############################
        # SHOW / HIDE MANIFEST DIV #
        ############################
        @app.callback(
            Output("manifest-table-div", "style"),
            [
                Input("url", "pathname"),
            ],
            [
                State("manifest-table-div", "style"),
            ],
        )
        def l(pathname, starting_style):

            # Close the manifest table if we are not looking at an analysis page            
            if pathname is None or "/d/" not in pathname:
                starting_style["display"] = "none"
            # Otherwise, open it
            else:
                starting_style["display"] = "block"

            return starting_style

        ###########################
        # FILTER MANIFEST COLUMNS #
        ###########################
        @app.callback(
            Output("manifest-table", "hidden_columns"),
            [
                Input("manifest-table-select-columns", "value")
            ],
            [
                State("manifest-table", "columns")
            ]
        )
        def hide_manifest_columns(columns_selected, all_columns):
            # Return any columns which aren't selected
            return [
                col["id"]
                for col in all_columns
                if col["id"] not in columns_selected
            ]        

        ###################################
        # GREY OUT FILTERED MANIFEST ROWS #
        ###################################
        @app.callback(
            Output("manifest-table", "style_data_conditional"),
            [
                Input("url", "search"),
            ],
            [
                State("manifest-table", "columns")
            ]
        )
        def grey_out_manifest_rows(
            search_string, columns
        ):
            # Parse the search string
            args = decode_search_string(search_string)

            # Grey out individual rows marked by either `mask` or `filter`
            row_ix_list = [
                int(ix)
                for ix in args.get("mask", "").split(",")
                if len(ix) > 0 and ix != 'None'
            ]
            output_style = []
            for ix in row_ix_list:
                output_style.append(
                    {
                        'if': {'row_index': ix},
                        'backgroundColor': 'grey'
                    }
                )

            return output_style
            
        ##################################
        # INTERACTIVE SPECIMEN FILTERING #
        ##################################
        @app.callback(
            Output("url", "search"),
            [
                Input("manifest-table", "selected_rows")
            ],
            [
                State("url", "pathname"),
                State("url", "search")
            ]
        )
        def interactive_specimen_filtering(selected_rows, pathname, search_string):
            """Record the set of masked specimens in the search string of the URL."""
            if pathname is None or "/d/" not in pathname:
                return search_string

            # Parse out the search string
            args = decode_search_string(search_string)

            # If we don't have `n`, then skip it
            if 'n' not in args:
                return search_string

            # If we do, then update the value of `mask`
            args['mask'] = ",".join([
                str(ix)
                for ix in range(int(args['n']))
                if ix not in selected_rows
            ])

            # Now reformat the search string and return it
            return encode_search_string(args)


        ########################################
        # ADD MANIFEST TO FILTER MANIFEST CARD #
        ########################################
        @app.callback(
            [
                Output("manifest-table", "data"),
                Output("manifest-table", "columns"),
                Output("manifest-table", "selected_rows"),
                Output("manifest-table-select-columns", "options"),
                Output("manifest-table-select-columns", "value"),
                Output("select-specimens-col", "options"),
                Output("select-specimens-col", "value"),
            ],
            [
                Input("url", "pathname"),
                Input("username", "value"),
                Input("password", "value"),
            ],
            [
                State("manifest-table", "selected_rows"),
                State("manifest-table-select-columns", "value"),
            ]
        )
        def update_manifest_table(
            pathname, username, password, selected_rows, columns_selected
        ):

            # Set up the empty data object
            empty_data = [{'specimen': 'none'}]
            empty_columns = [{"name": "specimen", "id": "specimen"}]

            if pathname is None or "/d/" not in pathname:
                return empty_data, empty_columns, [], [], [], [], None

            # Parse the dataset name from the path
            dataset_id = pathname.split("/")[2]

            # Check to see if this user has permission to view this dataset
            if self.glam_db.user_can_access_dataset(dataset_id, username, password) is False:
                return empty_data, empty_columns, [], [], [], [], None

            # Get the base path to the dataset
            dataset_uri = self.glam_db.get_dataset_uri(dataset_id)
            
            # Get the manifest for this dataset
            manifest = self.glam_io.get_manifest(dataset_uri).reset_index()

            # Set up the data that is to be returned
            columns = [
                {
                    "name": col_name, "id": col_name
                }
                for col_name in manifest.columns.values
            ]
            data = manifest.to_dict("records")

            # If no rows are selected, select all of them
            if len(selected_rows) == 0:
                selected_rows = list(range(manifest.shape[0]))

            # Now format the options of selected columns
            dropdown_options = [
                {
                    "label": col_name, "value": col_name
                }
                for col_name in manifest.columns.values
            ]

            # By default, select all of the columns
            output_columns_selected = manifest.columns.values

            # However, if the user has already filtered columns, keep that set
            if len(columns_selected) > 1 and all([col_name in manifest.columns.values for col_name in columns_selected]):
                output_columns_selected = columns_selected

            return data, columns, selected_rows, dropdown_options, output_columns_selected, dropdown_options, manifest.columns.values[0]

        ######################################
        # FILL IN VALUES FOR MANIFEST COLUMN #
        ######################################
        @app.callback(
            [
                Output("select-specimens-value", "options"),
                Output("select-specimens-value", "value"),
            ],
            [
                Input("url", "pathname"),
                Input("username", "value"),
                Input("password", "value"),
                Input("select-specimens-col", "value"),
            ]
        )
        def values_for_specimen_filtering(
            pathname,
            username,
            password,
            filter_col,
        ):
            """Fill in the values from a column which are available for filtering."""
            # Make default output
            disabled_output = [{"label": "none", "value": "none"}], "none"
            
            # Stop if no dataset has been selected
            if pathname is None or "/d/" not in pathname:
                return disabled_output

            # Parse the dataset name from the path
            dataset_id = pathname.split("/")[2]

            # Check to see if this user has permission to view this dataset
            if self.glam_db.user_can_access_dataset(dataset_id, username, password) is False:
                return disabled_output

            # Get the base path to the dataset
            dataset_uri = self.glam_db.get_dataset_uri(dataset_id)

            # Get the manifest for this dataset
            manifest = self.glam_io.get_manifest(dataset_uri).reset_index()

            # Make sure that the manifest has the column of interest
            if filter_col not in manifest.columns.values:
                return disabled_output

            # Return all of the available values for a particular row
            options = [
                {"label": v, "value": v}
                for v in manifest[filter_col].unique()
            ]
            return options, manifest[filter_col].unique()[0]

        #####################################
        # BULK MASK SPECIMENS USING FORMULA #
        #####################################
        @app.callback(
            [
                Output("select-specimens-apply", "href"),
                Output("select-specimens-apply", "disabled"),
                Output("select-specimens-apply", "children"),
            ],
            [
                Input("url", "pathname"),
                Input("url", "search"),
                Input("username", "value"),
                Input("password", "value"),
                Input("select-specimens-show-hide", "value"),
                Input("select-specimens-col", "value"),
                Input("select-specimens-comparitor", "value"),
                Input("select-specimens-value", "value"),
            ]
        )
        def bulk_mask_specimens_using_formula(
            pathname,
            search_string,
            username,
            password,
            filter_operation,
            filter_col,
            filter_comparison,
            filter_value
        ):
            """Based on the user's specifications, format an href to show or mask a set of specimens."""
            # Make default output for a disabled button
            disabled_output = f"{pathname}{search_string}", True, "Apply"
            
            # Stop if no dataset has been selected
            if pathname is None or "/d/" not in pathname:
                return disabled_output

            # Parse the dataset name from the path
            dataset_id = pathname.split("/")[2]

            # Check to see if this user has permission to view this dataset
            if self.glam_db.user_can_access_dataset(dataset_id, username, password) is False:
                return disabled_output

            # Get the base path to the dataset
            dataset_uri = self.glam_db.get_dataset_uri(dataset_id)

            # Get the manifest for this dataset
            manifest = self.glam_io.get_manifest(dataset_uri).reset_index()

            # If the selected column is not in the manifest, return an empty link
            if filter_col not in manifest.columns.values:
                return disabled_output

            # Get the set of rows which pass the filter
            if filter_comparison == "Equals":
                passing_filter = manifest[filter_col] == filter_value
            elif filter_comparison == "Does not equal":
                passing_filter = manifest[filter_col] != filter_value
            elif filter_comparison == "Is greater than":
                passing_filter = manifest[filter_col] > filter_value
            elif filter_comparison == "Is less than":
                passing_filter = manifest[filter_col] < filter_value
            elif filter_comparison == "Is greater than or equal to":
                passing_filter = manifest[filter_col] >= filter_value
            elif filter_comparison == "Is less than or equal to":
                passing_filter = manifest[filter_col] <= filter_value
            else:
                assert False, f"Could not parse comparison {filter_comparison}"

            # The new set of masked specimens will include those specimens which were previously masked
            args = decode_search_string(search_string)
            old_masked_rows = set([
                int(ix)
                for ix in args.get("mask", "").split(",")
                if len(ix) > 0
            ])

            # Get the index positions of all specimens matching the pattern
            passing_filter = passing_filter.reset_index(drop=True)
            rows_matching_pattern = set(passing_filter.index.values[passing_filter])

            # Either add or remove specimens based on user input
            if filter_operation == "Show":
                new_masked_rows = old_masked_rows - rows_matching_pattern
            else:
                assert filter_operation == "Hide", f"Could not parse filter operation {filter_operation}"
                new_masked_rows = old_masked_rows | rows_matching_pattern

            # If all of the rows are masked, disable the button
            if len(new_masked_rows) == manifest.shape[0]:
                return f"{pathname}{search_string}", True, "Please Review Formula"
            
            # Update the arguments for the search string
            args["mask"] = ",".join([str(ix) for ix in list(new_masked_rows)])

            # Set the label of the button
            masking_delta = len(new_masked_rows) - len(old_masked_rows)
            if masking_delta >= 0:
                operation = "Mask"
            else:
                operation = "Show"
            if abs(masking_delta) <= 1:
                item = "Specimen"
            else:
                item = "Specimens"
            
            button_label = f"{operation} {masking_delta} {item}"

            return encode_search_string(args), False, button_label


        ##########################
        # RESET SPECIMEN MASKING #
        ##########################
        @app.callback(
            Output("select-specimens-reset", "href"),
            [
                Input("url", "pathname"),
                Input("url", "search"),
            ]
        )
        def reset_specimen_masking(
            pathname,
            search_string,
        ):
            """Format an href without any `mask` or `filter`."""

            args = decode_search_string(search_string)
            args["mask"] = ""
            return f"{pathname}{encode_search_string(args)}"

        ######################
        # EXPERIMENT SUMMARY #
        ######################
        @app.callback(
            Output({"name": "experiment-summary", "index": MATCH}, "children"),
            [
                Input("url", "pathname"),
                Input("url", "search"),
                Input("username", "value"),
                Input("password", "value"),
            ],
            [
                State({"name": "experiment-summary", "index": MATCH}, "style")
            ]
        )
        def update_experiment_summary_card(pathname, search_string, username, password, _):
            # Make sure we're looking at a dataset
            if "/d/" not in pathname:
                raise PreventUpdate

            # Parse the dataset name from the path
            dataset_id = pathname.split("/")[2]

            # Check to see if this user has permission to view this dataset
            if self.glam_db.user_can_access_dataset(
                dataset_id,
                username,
                password
            ):
                # Get the base path to the dataset
                dataset_uri = self.glam_db.get_dataset_uri(dataset_id)

                # Get the experiment summary metrics
                dataset_metrics = self.glam_io.get_dataset_metrics(dataset_uri)

                # Make a table with the basic summary of an experiment
                return self.glam_layout.format_experiment_summary_table(dataset_metrics)

        #################################
        # ORDINATION PLOT - ANOSIM TEXT #
        #################################
        @app.callback(
            Output({"name": "ordination-anosim-results", "index": MATCH}, "children"),
            [
                Input("url", "pathname"),
                Input("url", "search"),
                Input("username", "value"),
                Input("password", "value"),
            ],
            [
                State({"name": "ordination-anosim-results", "index": MATCH}, "style"),
            ]
        )
        def anosim_results(pathname, search_string, username, password, _):

            if "/d/" not in pathname:
                raise PreventUpdate
            # Parse the dataset name from the path
            dataset_id = pathname.split("/")[2]

            # Parse the search string, while setting default arguments for this card
            args = decode_search_string(
                search_string,
                distance_metric="braycurtis",
                ordination_algorithm="pca",
                primary_pc="1",
                secondary_pc="2",
                perplexity=30
            )

            # Only run if a metadata label has been selected
            if args.get("color_by") is None or args.get("color_by") == "none":
                return ""

            # Check to see if this user has permission to view this dataset
            if self.glam_db.user_can_access_dataset(
                dataset_id,
                username,
                password
            ):
                # Get the base path to the dataset
                dataset_uri = self.glam_db.get_dataset_uri(dataset_id)

                return self.glam_plotting.ordination_anosim_results(
                    self.glam_io.get_distances(
                        dataset_uri, 
                        args["distance_metric"]
                    ),
                    self.glam_io.get_manifest(dataset_uri),
                    args,
                )

        ################
        # RENDER PLOTS #
        ################
        # All of the plots are given a standardzed ID which all trigger the same callback
        # For every plot, we check for credentials, decode the input arguments, and call
        # the apppropriate plotting function
        @app.callback(
            Output({"type": "analysis-card-plot", "name": MATCH}, "figure"),
            [
                Input("url", "pathname"),
                Input("url", "search"),
                Input("username", "value"),
                Input("password", "value"),
            ],
            [State({"type": "analysis-card-plot", "name": MATCH}, "style")]
        )
        def render_plots(pathname, search_string, username, password, _):
            # Get the callback context
            ctx = dash.callback_context

            # Read the name of the plot from the ID of the State
            plot_name = list(ctx.states.keys())[0]
            plot_name = json.loads(plot_name.replace(".style", ""))
            plot_name = plot_name["name"]

            # Only render the plot if we are viewing a dataset and an analysis
            if "/d/" not in pathname or "/a/" not in pathname:
                raise PreventUpdate

            # Parse the dataset name from the path
            dataset_id = pathname.split("/d/")[1].split("/", 1)[0]

            # Parse the analysis name from the path
            analysis_id = pathname.split("/a/")[1].split("/", 1)[0]

            # Only render the plot if this user has permission to view this dataset
            if not self.glam_db.user_can_access_dataset(
                dataset_id,
                username,
                password
            ):
                raise PreventUpdate

            # Get the defaults set by this particular card
            page_defaults = self.glam_layout.get_defaults(dataset_id, analysis_id)

            # Parse the search string, while setting default arguments for this card
            args = decode_search_string(
                search_string,
                **page_defaults
            )

            # Get the base path to the dataset
            dataset_uri = self.glam_db.get_dataset_uri(dataset_id)

            # Render the plot
            return self.glam_plotting.plot(
                plot_name=plot_name,
                glam_io=self.glam_io, 
                args=args, 
                dataset_uri=dataset_uri
            )

        ##########################
        # STORE BROWSING HISTORY #
        ##########################
        # Fill in the href for the "Back" button
        @app.callback(
            [
                Output("back-button-link", "href"),
                Output("forward-button-link", "href"),
                Output("history-reverse", "children"),
                Output("history-forward", "children"),
            ],
            [
                Input("url", "pathname"),
                Input("url", "search"),
            ],
            [
                State("back-button-link", "href"),
                State("forward-button-link", "href"),
                State("history-reverse", "children"),
                State("history-forward", "children"),
            ]
        )
        def browsing_history(
            pathname,
            search_string,
            current_back_href,
            current_forward_href,
            history_reverse,
            history_forward,
        ):
            # Get the callback context
            ctx = dash.callback_context

            # If there was no trigger, don't update
            if not ctx.triggered:
                raise PreventUpdate

            # The 'back' and 'forward' links will contain the argument 'nav=back' or 'nav=forward'

            # Parse the search string to see if this navigation change was part of the callback
            args = decode_search_string(search_string)

            # First let's consider the simple case where no history navigation has taken place
            if args.get("nav") is None:

                # Only add to the history if an analysis is being shown
                if "/a/" in pathname:

                    # Format a string with the current location
                    current_location = f"{pathname}{search_string}"

                    # Add this location to the recent history, if it isn't already the most recent history
                    # If the current page is not the end of the reverse history list, add it there
                    if len(history_reverse) == 0 or history_reverse[-1] != current_location:

                        # Add to the history-reverse list
                        history_reverse.append(current_location)

            # Now let's consider the updates where nav=back
            elif args.get("nav") == "back":

                # If we don't have any reverse history to compare against, stop now
                # This is intended to catch an edge case which should not occur
                if len(history_reverse) == 0:
                    pass

                else:

                    # Remove the 'nav' key from the dict, for comparison to values in the history
                    del args["nav"]

                    # Check to see if the last item in the reverse history is the same as the current location
                    if location_matches(history_reverse[-1], pathname, args):

                        # Everything is as it should be
                        pass

                    # Now check to see if we have navigated to the next-to-last item in the list
                    elif len(history_reverse) > 1 and location_matches(history_reverse[-2], pathname, args):

                        # In this case, remove the last item from the reverse history
                        previous_page = history_reverse.pop()

                        # If this is not already the last item in the forward history, add it there
                        if len(history_forward) == 0 or previous_page != history_forward[-1]:

                            history_forward.append(previous_page)

            # Finally, let's conider the updates where nav=forward
            else:
                assert args.get("nav") == "forward", args

                # Remove the 'nav' key from the dict, for comparison to values in the history
                del args["nav"]

                # If the forward history list is empty
                if len(history_forward) == 0:
                    # then we may have already updated the forward history, and can stop
                    pass

                else:
                    # There is at least one item in the forward history

                    # If the last item in the forward history is the current page
                    if location_matches(history_forward[-1], pathname, args):

                        # Then we should remove the last item from the forward history
                        previous_page = history_forward.pop()

                        # If this is not already the last item in the reverse history, add it there
                        if len(history_reverse) == 0 or previous_page != history_reverse[-1]:

                            history_reverse.append(previous_page)

            # The back button will always be the next-to-last item from reverse history
            if len(history_reverse) <= 1:
                back_button_link = ""
            else:
                back_button_link = f"{history_reverse[-2]}&nav=back"

            # The forward button will always be the last item in forward history
            if len(history_forward) < 1:
                fwd_button_link = ""
            else:
                fwd_button_link = f"{history_forward[-1]}&nav=forward"

            return back_button_link, fwd_button_link, history_reverse, history_forward


        ################################
        # OPEN / CLOSE BOOKMARKS MODAL #
        ################################
        @app.callback(
            Output({"type": "modal", "name": MATCH}, "is_open"),
            [
                Input({"type": "open-modal", "name": MATCH}, "n_clicks"),
                Input({"type": "close-modal", "name": MATCH}, "n_clicks"),
            ],
            [
                State({"type": "modal", "name": MATCH}, "is_open")
            ]
        )
        def open_close_bookmarks(open_clicks, close_clicks, is_open):
            if open_clicks is None and close_clicks is None:
                return is_open
            else:
                return is_open is False


        #####################
        # DISPLAY BOOKMARKS #
        #####################
        @app.callback(
            Output("bookmarks-modal-body", "children"),
            [
                Input("username", "value"),
                Input("password", "value"),
                Input({"type": "close-modal", "name": "save-bookmark"}, "n_clicks"),
                Input({"name": "delete-bookmark", "index": ALL}, "n_clicks")
            ]
        )
        def display_bookmarks(username, password, _a, _b):

            # Not the best solution, but this is used to wait for
            # database update to be complete after bookmark deletion
            sleep(1)

            # Display the list of bookmarks for this user            
            return self.glam_layout.bookmark_list(
                self.glam_db.user_bookmarks(username, password)
            )

        ################################################
        # CLOSE BOOKMARKS MODAL AFTER OPENING BOOKMARK #
        ################################################
        @app.callback(
            Output({"type": "close-modal", "name": "list-bookmarks"}, "n_clicks"),
            [
                Input({"name": "link-in-bookmarks-modal", "index": ALL}, "n_clicks")
            ],
            [State({"type": "close-modal", "name": "list-bookmarks"}, "n_clicks")]
        )
        def close_bookmark_list_after_opening_bookmark(clicks_on_link, clicks_on_close_button):
            if all([i is None for i in clicks_on_link]):
                return clicks_on_close_button
            else:
                if clicks_on_close_button is None:
                    return 1
                else:
                    return clicks_on_close_button + 1            

        #################
        # SAVE BOOKMARK #
        #################
        @app.callback(
            Output({"type": "close-modal", "name": "save-bookmark"}, "n_clicks"),
            [
                Input("save-bookmark-button", "n_clicks"),
            ],
            [
                State("username", "value"),
                State("password", "value"),
                State("url", "pathname"),
                State("url", "search"),
                State("save-bookmark-name", "value"),
                State({"type": "close-modal", "name": "save-bookmark"}, "n_clicks")
            ]
        )
        def save_bookmark(save_nclicks, username, password, pathname, search_string, bookmark_name, n_clicks):
            # Save this bookmark
            if save_nclicks is not None:

                self.glam_db.save_bookmark(
                    username, password, f"{pathname}{search_string}", bookmark_name
                )

                # 'click' the close button
                if n_clicks is None:
                    return 1
                else:
                    return n_clicks + 1

        ###################
        # DELETE BOOKMARK #
        ###################
        @app.callback(
            Output({"name": "delete-bookmark", "index": MATCH}, "style"),
            [
                Input("username", "value"),
                Input("password", "value"),
                Input({"name": "delete-bookmark", "index": MATCH}, "n_clicks")
            ],
            [
                State({"name": "delete-bookmark", "index": MATCH}, "style")
            ]
        )
        def delete_bookmark(username, password, n_clicks, style):
            ctx = dash.callback_context
            if ctx.triggered:
                # Get the index of the bookmark which is to be deleted
                ix = self.parse_callback_trigger(ctx)["index"]
                
                # Delete the bookmark
                self.glam_db.delete_bookmark(username, password, ix)

            return style

        ###################################
        # ENABLE INPUT FIELD APPLY BUTTON #
        ###################################
        @app.callback(
            Output({"name": "apply_button", "key": MATCH}, "href"),
            [
                Input({"name": "input_field", "key": MATCH}, "value")
            ],
            [
                State("url", "pathname"),
                State("url", "search"),
            ]
        )
        def update_input_field_apply_button(field_value, pathname, search_string):
            ctx = dash.callback_context
            if ctx.triggered:

                # Parse the variable name from the callback context
                key_name = self.parse_callback_trigger(ctx)["key"]

                # Return a new href which includes this key-value pair
                args = decode_search_string(search_string)
                args[key_name] = str(ctx.triggered[0]["value"])
                search_string = encode_search_string(args)

                return f"{pathname}{search_string}"

            else:
                return f"{pathname}{search_string}"


def location_matches(full_href, pathname, args):
    """Check if a location matches a given pathname and search string arguments."""

    # The beginning should start with the pathname
    if full_href.startswith(pathname) is False:
        return False

    # If there is no '?', then there is no search string
    if "?" not in full_href:

        # If there are no arguments, then everything is fine
        if len(args) == 0:
            return True

        # Otherwise, something doesn't match
        else:
            return False

    else:
        # A search string is present

        # Get the arguments from the full_href
        query_args = decode_search_string(f"?{full_href.split('?', 1)[1]}")

        # Return false if any parameters don't match
        for k, v in query_args.items():
            if args.get(k) is None or args.get(k) != v:
                return False
        for k, v in args.items():
            if query_args.get(k) is None or query_args.get(k) != v:
                return False

        # Otherwise, everything matches
        return True

