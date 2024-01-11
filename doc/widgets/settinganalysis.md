Setting
=======

![](images/setting_analysis_icon.png)

An Orange3 widget to identify clusters of words in individual stories which might indicate the major topics touched upon within the story. Stories must be textual stories written in the Dutch language.

**Inputs**

- Corpus: A dataset of one or more textual documents in Dutch.

**Outputs**

- Word vectors: A data table consisting of embedding vectors for the relevant words in the given corpus of stories. Each row represents a word vector. The first column is the list of relevant words in the collection, and all subsequent columns contain a component (dimension) value of the word vector. The second column contains the first component, the third column contains the second component etc. The number of dimensions for each word vector is N-1 where N is the number of columns in the data table.

