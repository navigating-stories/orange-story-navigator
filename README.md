Digital Story Grammar (DSG) Add-on for Orange3
==============================================

This add-on for Orange3 enables digital story grammar (DSG) analysis of Dutch texts. Please see the following paper for more information on DSG: [Andrade and Andersen (2020)](https://www.tandfonline.com/doi/full/10.1080/13645579.2020.1723205)

Requirements
------------
The following versions of Python and Orange3 were used:

+ Python v3.8.3
+ Orange3 v3.32.0
+ Orange3-text v1.7.0

Installation
------------

To install the add-on from source, clone this repo, `cd` into the root directory and run

    pip install -r requirements.txt
    pip install .

To make this add-on known to your local Orange without copying the code it to Python's site-packages directory, run

    pip install -e .

Usage
-----

After the installation, the widget from this add-on is registered with Orange. To run Orange from the terminal,
use

    orange-canvas

or

    python -m Orange.canvas

The DSG widget appears in the toolbox bar in the section `Navigating Stories`:

![screenshot](https://github.com/navigating-stories/test-orange-widget/blob/master/screenshot.png)

Connecting the DSG widget to the Corpus widget from the `Text Mining` section will automatically start a DSG analysis on the first corpus of the Corpus widget (21 letters from the future).

![screenshot](https://github.com/navigating-stories/test-orange-widget/blob/master/screenshot1.png)

The result, a table with a dependency parsing analysis and a semantic role labeling analysis, can be inspected in the Data Table widget of the `Data` section in Orange:

![screenshot](https://github.com/navigating-stories/test-orange-widget/blob/master/screenshot2.png)

