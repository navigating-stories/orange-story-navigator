Dutch Digital Story Grammar (DSG) Example Add-on for Orange3
============================================================

This is a very basic, experimental add-on for Orange3 which enables digital story grammar (DSG) analysis of Dutch text documents. Please see the following paper for more information: [https://www.tandfonline.com/doi/full/10.1080/13645579.2020.1723205](https://www.tandfonline.com/doi/full/10.1080/13645579.2020.1723205)

Installation
------------

To install the add-on from source run

    pip install .

To register this add-on with Orange, but keep the code in the development directory (do not copy it to 
Python's site-packages directory), run

    pip install -e .

Usage
-----

After the installation, the widget from this add-on is registered with Orange. To run Orange from the terminal,
use

    orange-canvas

or

    python -m Orange.canvas

The new widget appears in the toolbox bar under the section `Navigating Stories`.

![screenshot](https://github.com/biolab/orange3-example-addon/blob/master/screenshot.png)
