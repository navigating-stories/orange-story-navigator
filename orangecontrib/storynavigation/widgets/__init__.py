"""
=======
Story Navigator
=======

Textual story narrative theory analysis tools for Orange.

"""

# Category description for the widget registry
import sysconfig
NAME = "Story Navigator"

DESCRIPTION = "Textual story narrative theory analysis tools for Orange."

BACKGROUND = "#C0FF97"

ICON = "icons/Category-NavigatingStories.png"

# Location of widget help files.
WIDGET_HELP_PATH = (
    # Used for development.
    # You still need to build help pages using
    # make htmlhelp
    # inside doc folder
    ("{DEVELOP_ROOT}/doc/_build/html/index.html", None),

    # Documentation included in wheel
    # Correct DATA_FILES entry is needed in setup.py and documentation has to be built
    # before the wheel is created.
    ("{}/help/orange3-storynavigator/index.html".format(sysconfig.get_path("data")), None),

    # Online documentation url, used when the local documentation is available.
    # Url should point to a page with a section Widgets. This section should
    # includes links to documentation pages of each widget. Matching is
    # performed by comparing link caption to widget name.
    ("https://orange3-storynavigator.readthedocs.io/en/latest/", "")
)
