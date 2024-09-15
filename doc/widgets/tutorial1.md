Tutorial 1
=======
![](../../doc/widgets/images/storynavigator_logo_small.png)

These *tutorials* will provide a list of workflow examples that demonstrate the usage of the StoyNavigator widgets in order to answer some predefined research questions. Specifically, the tutorials will show how to use the widgets in combination with other, pre0existing widgets within Orange, and how to interpret the results. 

# Tutorial 1: Examining Agency

The first tutorial will demonstrate how to use the StoryNavigator widgets to examine the agency of a text. The tutorial will show how to use the widgets to answer the following research questions:

1. What are the most frequent agents across the entire corpus of stories?
2. What are the most frequent agents in a specific story?


## Workflow

The tutorial will use the following workflow:
![](../../doc/widgets/images/agency.jpg)

This workflow can be downloaded [here](../../doc/widgets/workflows/). In addition, we use a set of 35 Dutch fairytales which can be found [here](../../doc/widgets/fairytales/).

## Starting the workflow
Firsrt, load fairytales.tab using the **Corpus** widget and connect it to **Preprocessing text** with default parameters. The Corpus viewer informs us that we have 35 documents. Next, connect Preprocess Text to both the **elements** and **actors** widget. Remember, the elements widget is the 'motor' of the StoryNavigator which provides information on the texts on a token level, while the actors widget is used to extract the agents from the text.

It is good practice to use a **DataTable** widget to inspect the data at each step of the workflow. This will help you to understand the data and the results of each widget. To further zoom in on the data, you can use the **Select Rows** widget to select only the rows in which we are interested in. Here, we make a selection of rows where there is a minimum amount of agency and a minimum amount of frequency.