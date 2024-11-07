import pandas as pd
from storynavigation.modules.meansanalysis import MeansAnalyzer

def test_sort_and_filter_results():
    sample_input = {1: {1: {"text": "", "label_": "", "sentence_id": 1},
                        0: {"text": "", "label_": "", "sentence_id": 0}},
                    0: {0: {"text": "", "label_": "", "sentence_id": 0}}}

    expected_dict = [["", "", "ST0", 0, 0],
                     ["", "", "ST1", 0, 0],
                     ["", "", "ST1", 1, 1]]
    expected_df = pd.DataFrame(expected_dict,
                               columns=["text",
                                        "label",
                                        "text_id",
                                        "sentence_id",
                                        "character_id"]).reset_index(drop=True)
    means_analyzer_object = MeansAnalyzer("", None, pd.DataFrame([], columns=["sentence"]), "")
    results_df = means_analyzer_object._MeansAnalyzer__sort_and_filter_results(sample_input)
    pd.testing.assert_frame_equal(results_df, expected_df)
