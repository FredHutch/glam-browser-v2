#!/usr/bin/env python3

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
from flask_caching import Cache
from helpers.db import GLAM_DB
from helpers.io import GLAM_IO
from helpers.layout import GLAM_LAYOUT
from helpers.plotting import GLAM_PLOTTING
from helpers.callbacks import GLAM_CALLBACKS
import logging
import os


##################
# SET UP LOGGING #
##################

logFormatter = logging.Formatter(
    '%(asctime)s %(levelname)-8s [GLAM Browser] %(message)s'
)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

# Write to STDOUT
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


##################
# SET UP THE APP #
##################

FONT_AWESOME = {
    "src": "https://kit.fontawesome.com/f8b0dec9e6.js",
    "crossorigin": "anonymous"
}

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    external_scripts=[FONT_AWESOME]
)
app.title = "GLAM Browser"
app.config.suppress_callback_exceptions = True


##################
# CONFIGURE GLAM #
##################

# Get the details for connecting to the database which have
# been set in the environment prior to running the app
glam_config = dict(
    db_name=os.getenv("DB_NAME"),
    db_username=os.getenv("DB_USERNAME"),
    db_password=os.getenv("DB_PASSWORD"),
    db_host=os.getenv("DB_HOST"),
    db_port=os.getenv("DB_PORT"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_region=os.getenv("AWS_REGION"),
)

# GLAM_DB will read from and write to the database
glam_db = GLAM_DB(**glam_config)

# GLAM_IO will read from S3 using the provided AWS credentials
glam_io = GLAM_IO(**glam_config)

# GLAM_PLOTTING just contains plotting code
glam_plotting = GLAM_PLOTTING()

# GLAM_LAYOUT contains all of the layout code for the browser
glam_layout = GLAM_LAYOUT(
    glam_db=glam_db, 
    glam_io=glam_io, 
    **glam_config
)

#####################
# SET UP APP LAYOUT #
#####################

# Set up the layout of the app
app.layout = glam_layout.base()


####################
# SET UP CALLBACKS #
####################

# GLAM_CALLBACKS will drive the interactivity of the browser,
# reading both from the database and from AWS S3
glam_callbacks = GLAM_CALLBACKS(glam_db, glam_io, glam_layout, glam_plotting)

# Decorate the callback functions with @app.callback as appropriate
glam_callbacks.decorate(app)

# Used for gunicorn execution
server = app.server

# ADD THE GTM CONTAINER, IF PROVIDED
gtm_container = os.getenv("GTM_CONTAINER")
if gtm_container is not None and isinstance(gtm_container, str) and gtm_container.startswith("GTM-"):
    app.index_string = """<!DOCTYPE html>
<html>
    <head>
    {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);})(window,document,'script','dataLayer','""" + gtm_container + """');</script>
    </head>
    <body>
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=""" + gtm_container + """"
height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# Run the app
if __name__ == '__main__':

    app.run_server(
        host='0.0.0.0',
        port=8050,
    )
