Contribute
=======
![](../../doc/widgets/images/storynavigator_logo_small.png)

This section is for those who want to contribute to the Story Navigator add-on by building from source, extending, customizing, or fixing bugs in it.

# Contributing guidelines

Welcome! *Story Navigator* is an open-source project that helps analyzing textual stories using the lenses of narrative psychology, narrative analysis, and narrative theory. We aim to work with a a wide range of textual  data. If you're trying *Story Navigator* with your data, your experience, questions, bugs you encountered, and suggestions for improvement are important to the success of the project.

We have a [Code of Conduct](https://github.com/navigating-stories/orange-story-navigator/tree/documentation/doc/widgets/CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.

## Questions, feedback, bugs

Use the search function to see if someone else already ran accross the same issue. Feel free to open a new [issue here](https://github.com/navigating-stories/orange-story-navigator/issues) to ask a question, suggest improvements/new features, or report any bugs that you ran into.

## Submitting changes

Even better than a good bug report is a fix for the bug or the implementation of a new feature. We welcome any contributions that help improve the code.

When contributing to this repository, please first discuss the change you wish to make via an [issue](https://github.com/navigating-stories/orange-story-navigator/issues) with the owners of this repository before making a change. You can also contact the developers at developers@storynavigator.com before submitting PRs.

Contributions can come in the form of:

- Bug fixes
- New features
- Improvement of existing code
- Updates to the documentation
- Etc

We use the usual GitHub pull-request flow. For more info see [GitHub's own documentation](https://help.github.com/articles/using-pull-requests/).

Typically this means:

1. [Forking](https://docs.github.com/articles/about-forks) the repository and/or make a [new branch](https://docs.github.com/articles/about-branches)
2. Making your changes
3. Make sure that the tests pass and add your own
4. Make sure the documentation is updated for new features
5. Pushing the code back to Github
6. [Create a new Pull Request](https://help.github.com/articles/creating-a-pull-request/)

One of the code owners will review your code and request changes if needed. Once your changes have been approved, your contributions will become part of *Story Navigator*.

## Getting started with development

### Setup

*Story Navigator* targets Python 3.9.12 or newer.

Clone the repository into the `storynavigator` directory:

```console
git clone https://github.com/navigating-stories/orange-story-navigator
```

Install using `virtualenv`:

```console
cd storynavigator
python3 -m venv env
source env/bin/activate
python3 -m pip install -e .[develop]
```

Alternatively, install using Conda:

```console
cd storynavigator
conda create -n storynavigator python=3.10
conda activate storynavigator
pip install -e .[develop]
```

### Running tests

*Story Navigator* uses [pytest](https://docs.pytest.org/en/latest/) to run the tests. You can run the tests for yourself using:

```console
pytest
```

To check coverage:

```console
coverage run -m pytest
coverage report  # to output to terminal
coverage html    # to generate html report
```

### Building the documentation

The documentation is written in [markdown](https://www.markdownguide.org/basic-syntax/), and uses [mkdocs](https://www.mkdocs.org/) to generate the pages.

To build the documentation for yourself:

```console
pip install -e .[docs]
mkdocs serve
```

You can find the documentation source in the [docs](https://github.com/navigating-stories/orange-story-navigator/tree/master/doc) directory.

### Making a release

1. Make a new [release](https://github.com/navigating-stories/orange-story-navigator/releases).

2. Under 'Choose a tag', set the tag to the new version. The versioning scheme we use is [SemVer](http://semver.org/), so bump the version (*major*/*minor*/*patch*) as needed. 

3. The [upload to pypi](https://pypi.org/project/orange-story-navigator/) is triggered when a release is published and handled by [this workflow](https://github.com/navigating-stories/orange-story-navigator/actions/workflows/publish.yaml).

4. The [upload to zenodo](https://zenodo.org/records/10994947) is triggered when a release is published.
