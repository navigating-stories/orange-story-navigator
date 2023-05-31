import numpy as np
import networkx as nx
import sys

from AnyQt.QtCore import Qt

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.widget import Input
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget, OWWidget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.settings import SettingProvider, Setting
from Orange.widgets.utils.plot import OWPlotGUI
from Orange.data.pandas_compat import table_to_frame
from Orange.widgets.utils.widgetpreview import WidgetPreview

from orangecontrib.storynavigation.netviz.graphview import GraphView

import pandas as pd

""" Copyright 2023, Ji Qi, Netherlands eScience Center, NL, j.qi@esciencecenter.nl'

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

GRAPH_LAYOUT = ('spring', 'multipartite', 'kamada kawai', 'spectral')

class OWArgExplorer(OWDataProjectionWidget):
    name = 'Narrative Network Explorer'
    description = 'Enables visual plotting and exploration of entities anf their relationships in narrative texts'
    icon = 'icons/OWArgExplorer.svg'
    
    class Inputs:
        edge_data = Input('Edge Data', Table)
        node_data = Input('Node Data', Table)
        
    GRAPH_CLASS = GraphView
    graph = SettingProvider(GraphView) 
    
    node_sparsity = Setting(5)
    graph_layout = Setting(GRAPH_LAYOUT[0]) # comboBox widget returns index of the selection
    # sourcenode_column_name = Setting(0)
    # targetnode_column_name = Setting(1)
    # idx = 0

    def __init__(self):
        super().__init__()
        
        self.edge_data = None
        self.node_data = None
        self.positions = None
        
    def _add_controls(self):
        self.gui = OWPlotGUI(self)
        layout = gui.vBox(self.controlArea, box='Layout') 
        gui.comboBox(layout, self, 'graph_layout', 
                    label='Graph layout', 
                    sendSelectedValue=True,
                    items=GRAPH_LAYOUT, 
                    callback=self.relayout)
        self.sparsity_control = gui.hSlider(layout, self, "node_sparsity", 
                    minValue=0, maxValue=10, intOnly=False, 
                    label="Node sparsity", orientation=Qt.Horizontal,
                    callback_finished=self.relayout)
        
            # edge_selection = gui.vBox(self.controlArea, box='Edge selection') 
            
            # # df_edge = None
            # if hasattr(self, 'edge_data'):
            #     print(table_to_frame(self.edge_data).columns)
            #     print(type(table_to_frame(self.edge_data).columns))
            #     print("yes")

            #     gui.comboBox(edge_selection, self, 'sourcenode_column_name', 
            #                 label='Source node column', 
            #                 sendSelectedValue=True,
            #                 items=table_to_frame(self.edge_data).columns, 
            #                 callback=self.relayout)
                
            #     gui.comboBox(edge_selection, self, 'targetnode_column_name', 
            #                 label='Target node column', 
            #                 sendSelectedValue=True,
            #                 items=table_to_frame(self.edge_data).columns, 
            #                 callback=self.relayout)
            # else:
            #     print("here...")
        
    @Inputs.edge_data
    def set_edge_data(self, data):
        self.edge_data = data

    @Inputs.node_data
    def set_node_data(self, data):
        self.node_data = data
        
    def handleNewSignals(self):
        self.AlphaValue = 0
        self.relayout()
        
    def relayout(self):
        """recompute positions of nodes and reset the graph
        """
        if self.node_data is None or self.edge_data is None:
            return
        
        self.sparsity_control.setEnabled(self.graph_layout == GRAPH_LAYOUT[0])
        self.set_positions()
        self.closeContext()
        self.data = self.node_data
        self.valid_data = np.full(len(self.data), True, dtype=bool)
        self.openContext(self.data)
        self.graph.reset_graph()


    # def table_to_frame_custom(self, tab, include_metas=False):
    #     """
    #     Convert Orange.data.Table to pandas.DataFrame
    #     Parameters
    #     ----------
    #     tab : Table
    #     include_metas : bool, (default=False)
    #         Include table metas into dataframe.
    #     Returns
    #     -------
    #     pandas.DataFrame
    #     """

    #     def _column_to_series(col, vals):
    #         result = ()
    #         if col.is_discrete:
    #             codes = pd.Series(vals).fillna(-1).astype(int)
    #             result = (col.name, pd.Categorical.from_codes(
    #                 codes=codes, categories=col.values, ordered=True
    #             ))
    #         elif col.is_time:
    #             result = (col.name, pd.to_datetime(vals, unit='s').to_series().reset_index()[0])
    #         elif col.is_continuous:
    #             dt = float
    #             # np.nan are not compatible with int column
    #             # using pd.isnull since np.isnan fails on array with dtype object
    #             # which can happen when metas contain column with strings
    #             if col.number_of_decimals == 0 and not np.any(pd.isnull(vals)):
    #                 dt = int
    #             result = (col.name, pd.Series(vals).astype(dt))
    #         elif col.is_string:
    #             result = (col.name, pd.Series(vals))
    #         return result

    #     def _columns_to_series(cols, vals):
    #         return [_column_to_series(col, vals[:, i]) for i, col in enumerate(cols)]

    #     x, y, metas = [], [], []
    #     domain = tab.domain
    #     print("domain: ", domain)
    #     print()
    #     if domain.attributes:
    #         print("domain attributes: ", domain.attributes)
    #         print()
    #         x = _columns_to_series(domain.attributes, tab.X)
    #         print("Table X: ", tab.X)
    #         print()
    #     if domain.class_vars:
    #         print("domain class_vars: ", domain.class_vars)
    #         print()
    #         y_values = tab.Y.reshape(tab.Y.shape[0], len(domain.class_vars))
    #         print("Table Y: ", tab.Y)
    #         print()
    #         print("Table Y reshaped: ", y_values)
    #         print()
    #         y = _columns_to_series(domain.class_vars, y_values)
    #     if domain.metas:
    #         print("domain metas: ", domain.metas)
    #         print()
    #         print("Table metas: ", tab.metas)
    #         print()
            
    #         metas = _columns_to_series(domain.metas, tab.metas)

    #     all_series = dict(x + y + metas)
    #     print("x series: ", x)
    #     print()
    #     print("y series: ", y)
    #     print()
            
    #     all_vars = tab.domain.variables
    #     print("Table domain variables: ", tab.domain.variables)
    #     print()
    #     if include_metas:
    #         all_vars += tab.domain.metas
    #     print("all_vars: ", all_vars)
    #     print()
    #     original_column_order = [var.name for var in all_vars]
    #     unsorted_columns_df = pd.DataFrame(all_series)
    #     print("len: ", len(unsorted_columns_df))
    #     print()
    #     return unsorted_columns_df[original_column_order]
        
    def set_positions(self):
        """set coordinates of nodes to self.positions.
        Args:
            layout (str, optional): name of layout. Defaults to "sfdp".
        """
        df_edge = table_to_frame(self.edge_data)
        df_node = table_to_frame(self.node_data, include_metas=True)

        print()
        print()
        print(df_edge)
        print()
        print()

        print()
        print()
        print(df_node)
        print()
        print()

        # df_edge = self.table_to_frame_custom(self.edge_data, include_metas=True)
        # df_node = self.table_to_frame_custom(self.node_data, include_metas=True)
        # df_edge = self.edge_data
        # df_node = self.node_data

        # df_node = {}
        # df_node['label'] = []
        # for item in df_edge:
        #     df_node['label'].append(item['subject'])
        # print("main man")
        # df_edge = pd.DataFrame([[1, 2, 0.5],[2, 3, 0.3],[3, 1, 0.2],[3, 4, 0.7],[1, 4, 0.8],[2, 5, 0.5],[5, 1, 0.5]], columns=['source', 'target', 'weight'])
        # df_node = pd.DataFrame([[1, 1, 'label1'], [2, 2, 'label1'], [3, 3, 'label2'], [4, 1, 'label1'], [5, 5, 'label2']], columns=['field1', 'field2', 'field3'])
        # print(df_edge.columns)
        # print(df_edge.head())
        G = nx.from_pandas_edgelist(
            df_edge, 
            source='subject_id', target='object_id',
            create_using=nx.DiGraph())
        
        # print("klsdajfbaksbdgksdbagjkg")
        # for node in G.nodes:
        #     nx.set_node_attributes(G, name=node, values={"label" : "test"})
        
        node_attrs = {i: {'subset': df_node['label'][i]} for i in G.nodes}

        print()
        print()
        print("node_attrs: ", node_attrs)
        print()
        print()
        nx.set_node_attributes(G, node_attrs)
        
        if len(G.nodes) < df_node.shape[0]:
            remain_nodes = df_node.iloc[~df_node.index.isin(G.nodes)]
            G.add_nodes_from(remain_nodes.index.tolist())
        # # in case arguments not appear in the attacking network
        # if len(G.nodes) < df_node.shape[0]:
        #     remain_nodes = df_node.iloc[~df_node.index.isin(G.nodes)]
        #     G.add_nodes_from(remain_nodes.index.tolist())
    
        if self.graph_layout == GRAPH_LAYOUT[0]:
            print("method1")
            spasity = (self.node_sparsity + 1) / 11.0
            pos_dict = nx.spring_layout(G, k=spasity, seed=10)
        elif self.graph_layout == GRAPH_LAYOUT[1]:
            print("method2")
            pos_dict = nx.multipartite_layout(G) 
        elif self.graph_layout == GRAPH_LAYOUT[2]:
            print("method3")
            pos_dict = nx.kamada_kawai_layout(G)
        elif self.graph_layout == GRAPH_LAYOUT[3]:
            print("method4")
            pos_dict = nx.spectral_layout(G)
       
        print()
        print("dict keys len: ", len(pos_dict))
        print()
        print()
        print("dict keys: ", pos_dict)
        print()


        self.positions = []
        idx = 1
        for i in sorted(pos_dict.keys()):
            if idx <= len(df_edge):
                self.positions.append(pos_dict[i])  
            idx += 1

        self.positions = np.array([*self.positions])

        print()
        print("asdasd: ", len(self.positions))
        print()
        
            
    def get_embedding(self):
        print("got here!!!!!!")
        print(self.positions.ndim)
        print()
        print("ttt: ", len(self.positions))
        print()
        return self.positions # check if the boolean index error stems from here...
    
    def get_edges(self):
        print("ttt2: ", len(self.edge_data))
        print()
        return table_to_frame(self.edge_data)
    
    def get_marked_nodes(self):
        return None
    
    def get_node_labels(self):
        print("ttt3: ", len(self.node_data))
        print()
        print()
        print("node labels!!! ", table_to_frame(self.node_data)['label'])
        print()
        return table_to_frame(self.node_data)['label']
    
    def selection_changed(self):
        super().selection_changed()
        self.graph.update_edges()


def main():
    #network = read_pajek(join(dirname(dirname(__file__)), 'networks', 'leu_by_genesets.net'))
    # network = read_pajek(join(dirname(dirname(__file__)), 'networks', 'lastfm.net'))
    #network = read_pajek(join(dirname(dirname(__file__)), 'networks', 'Erdos02.net'))
    #transform_data_to_orange_table(network)
    # WidgetPreview(OWSNDSGDepParser).run(set_graph=network)
    WidgetPreview(OWArgExplorer).run()

if __name__ == "__main__":
    main()


# def main(argv=sys.argv):
#     from AnyQt.QtWidgets import QApplication
#     app = QApplication(list(argv))
#     args = app.arguments()
#     if len(args) > 1:
#         filename = args[1]
#     else:
#         filename = "iris"

#     ow = OWArgExplorer()
#     ow.show()
#     ow.raise_()

#     dataset = Table(filename)
#     # ow.set_data(dataset)
#     ow.handleNewSignals()
#     app.exec_()
#     # ow.set_data(None)
#     # ow.handleNewSignals()
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())