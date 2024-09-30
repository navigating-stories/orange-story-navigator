# Tutorial 1: Examining Agency in stories

![](../../doc/widgets/images/storynavigator_logo_small.png)

---
This tuturial is part of a series in which several workflows are presented that demonstrate the usage of the StoyNavigator widgets. These tutorials demonstrate show how to use the storyNavigator widgets in combination with other, pre-existing widgets available within the Orange platform, and how to create output via tables or figuresm and interpret the results. At the basis of each tutorial lies a pre-specified research question related to the narrative structure and contents of the stories under analysis. 
---

This tutorial will demonstrate how to use the StoryNavigator widgets to examine the agency of a text. It will show how to use the widgets to answer the following research questions:

1. What are the most frequent agents across the entire corpus of stories?
2. What are the most frequent agents in a specific story?


## Workflow

The tutorial will use the following workflow:
![](../../doc/widgets/images/agency.jpg)

This workflow can be downloaded [here](../../doc/widgets/workflows/). In addition, it uses a dataset of 35 Dutch fairytales which can be found [here](../../doc/widgets/fairytales/).

## Starting the workflow
Firsrt, load fairytales.tab using the **Corpus** widget and connect it to **Preprocessing text** with default parameters. The Corpus viewer informs us that we have 35 documents. Next, connect Preprocess Text to both the **elements** and **actors** widget. Remember, the elements widget is the 'motor' of the StoryNavigator which provides information on the texts on a token level, while the actors widget is used to extract the agents from the text.

It is good practice to use a **DataTable** widget to inspect the data at each step of the workflow. This will help you to understand the data and the results of each widget. To further zoom in on the data, you can use the **Select Rows** widget to select only the rows in which we are interested in. Here, we make a selection of rows where there is a minimum amount of agency and a minimum amount of frequency occurences in the text.

Next, we use the **GroupBy** widget to either group the data based on *storyid* and *token* if we want to know the most frequent agents in a specific story, or based on *token* if we want to know the most frequent agents across the entire corpus. Finally, we use the **DataTable** widget to inspect the results.

To visualize the results, we can use the **Bar Chart** widget. Connect the **GroupBy** widget to the **Bar Chart** widget and select the *frequency* column or for example the *agency* column as the variable to be plotted. The Bar Chart will show the most frequent agents in the corpus or in a specific story. We can control the amount of output in the barchart by using the **Select Rows** widget to select only the rows in which we are interested in, or else select a specific story we are interested in the dataTable directly befroe the barchart widget. The barchart updates automatically when a selection of data is made. 

## Interpreting the results and changing the set of agency words
The results can be used to answer the research questions mentioned above. For example, the results can show that the most frequent agents throughout  the corpus are 'prince', 'king', and 'queen'. The results can also show that the most frequent agents in a specific story are 'prince', 'king', and 'queen'. In case one is interested in a specific type of noun, or else do not want to include words like 'ik' 'zij' or 'hen', one can use the **Select Rows** widget to select only the rows in which the desired noun is present.