Orange3 Story Navigator
=======================

Orange3-Story-Navigator is an add-on for the Orange data mining software package. It
provides story textual analysis features based on principles in [narrative psychology](https://web.lemoyne.edu/~hevern/narpsych/nr-basic.html). The aim of the widgets in the Story Navigator Orange module are to support human analysis of stories represented in digital texts with a main focus on identifying and describing the following components of the narrative (see [[1]](http://www.communicationcache.com/uploads/1/0/8/8/10887248/kenneth_burke_-_a_grammar_of_motives_1945.pdf) and [[2]](https://www.semanticscholar.org/paper/Qualitative-Research-in-Clinical-and-Health-Murray-Sools/8db3916fcd1593086f0a62d78d15eacc2d3236e6) for definitions):

1. Setting
2. Main agent
3. Means
4. Acts and events
5. Purpose
6. Breach

[1] Burke, K. (1969). The grammar of motives. Prentice Hall, New York. Originally published in 1945.
[2] M. Murray and A. Sools, P. Rohleder, A.C. Lyons (Eds.), Qualitative research in clinical and health psychology, Palgrave MacMillan, Houndmills Basingstoke (2015), pp. 133-154

[Story Navigator documentation]().

Documentation is found at: http://orange3-storynavigator.readthedocs.org/

Installation:
-------------

This section is for those who just want to install and use the add-on in Orange3.

Install from Orange add-on installer through Options - Add-ons.

To install the add-on with [pip](https://pypi.org/project/pip/) use

    pip install Orange-Story-Navigator

Development:
------------

This section is for those who want to build the add-on from source and extend, customise, or fix bugs in it.

Requirements:

1. A tool for checking out a [Git]() repository
2. Python 3.9.16+

Steps to build and test from source:

1. Get a copy of the code
    
    ```git clone git@github.com:navigating-stories/orange-story-navigator.git```

2. Change into the ```orange-story-navigator/``` directory
    
    ```cd orange-story-navigator```

3. Create and activate a new Python virtual environment using [virtualenv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

4. Run the following command

    ```pip install -r requirements.txt```

5. If Step 4. completed with no errors, run this command

    ```pip install -e .```

This command installs the widget and registers it with your Orange3 installation. If you encountered an error during Step 4, file an [issue](https://github.com/navigating-stories/orange-story-navigator/issues) with the details and skip the remaining steps.

6. Run 

    ```orange-canvas```

or

```python -m Orange.canvas```

The Orange3 application should start up after a few seconds and you can test the ```story-navigator``` widget.