Building from source
=======
![](../../doc/widgets/images/storynavigator_logo_small.png)

This section is for those who want to build the add-on from source and extend, customise, or fix bugs in it.
**Note:** Mac M1 (Apple silicon) users may encounter problems with building ``storynavigator`` using certain Python 3.8.x or 3.9.x versions and / or actually building or running these specific Python versions at all on your Mac. If you encounter such issues, it is recommended to install [Rosetta 2](https://osxdaily.com/2020/12/04/how-install-rosetta-2-apple-silicon-mac/) and always run the terminal using Rosetta 2 (see how to do the latter [here](https://www.courier.com/blog/tips-and-tricks-to-setup-your-apple-m1-for-development/)) for development tasks.

Requirements:

1. A tool for checking out a [Git](https://git-scm.com/) repository
2. Python 3.9.12+

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

> This command installs the widget and registers it with your Orange3 installation. If you encountered an error during Step 4, file an [issue](https://github.com/navigating-stories/orange-story-navigator/issues) with the details and skip the remaining steps.

6. Run ```orange-canvas``` or ```python -m Orange.canvas```

The Orange3 application should shortly start up with a splash screen

## Testing

Run tests with `pytest`.

For coverage:

```python
coverage run -m pytest
coverage report
# or
coverage html
```
