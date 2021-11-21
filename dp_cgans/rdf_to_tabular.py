from rdfpandas.graph import to_dataframe
import pandas as pd
import numpy as np
import rdflib


class RDF_to_Tabular(object):

    def __init__(self, file_path):

        self._path = file_path

        ### Input TTL file and convert to a plain CSV ###
        graph = rdflib.Graph()
        graph.parse(self._path, format = 'ttl')
        self._rdf_to_unstructured_df = to_dataframe(graph)




    def fit_convert(self, user_define_data_instance, user_define_is_a, user_define_has_value, set_level, as_column, remove_columns_unique_values):

        rdf_to_unstructured_df = self._rdf_to_unstructured_df
        self._user_define_data_instance = user_define_data_instance
        self._user_define_is_a = user_define_is_a #["rdf:type{URIRef}"]
        self._user_define_has_value = user_define_has_value#["http://www.cancerdata.org/roo/P100042", "http://www.cancerdata.org/roo/P100318"]
        self._set_level = set_level # 1
        self._as_column = as_column # 'object' # 'predicate', 'object', None (=='object')
        self._remove_columns_unique_values = remove_columns_unique_values #False

        def get_entity_index(rdf_to_unstructured_df, data_instance, is_a_list, has_value_list):
            data_instance_df = pd.DataFrame()

            for each_is_a in is_a_list:
                single_type_data_instance_df = rdf_to_unstructured_df[rdf_to_unstructured_df[each_is_a]==data_instance].dropna(axis=1)
                if not single_type_data_instance_df.empty:
                    data_instance_df = pd.concat([data_instance_df, single_type_data_instance_df], axis=1)
                    
            # remove is_a and has_value columns in the main data table
            data_instance_df = data_instance_df.drop(list(set(is_a_list+has_value_list) & set(data_instance_df.columns)),axis=1)
            return data_instance_df



        def get_data_value(URI_node_df, entities_in_graph, is_a_list, has_value_list):
            
            data_value_df=pd.DataFrame()
            sub_entities = {}
            relations_in_graph = {}
            
            for col in URI_node_df.columns:
                for entity in entities_in_graph:
                    try:
                        for each_is_a in is_a_list:
                            element_df=rdf_to_unstructured_df[rdf_to_unstructured_df[each_is_a]==entity].dropna(axis=1).loc[URI_node_df[col]]
                            value_col=[element_col for element_col in element_df.columns for each_has_value in has_value_list if each_has_value == element_col]
                     
                            if len(value_col)>0:
                                new_df = element_df[value_col].reset_index(drop=True).set_index(URI_node_df[col].index)
                                for each_value_col in value_col:
                                    if "integer" in each_value_col:
                                        try:
                                            new_df[each_value_col]=new_df[each_value_col].astype(int)
                                        except ValueError:
                                            new_df[each_value_col]=new_df[each_value_col].astype("float64")
                                    elif "double" in each_value_col:
                                        new_df[each_value_col]=new_df[each_value_col].astype("float64")
                                    elif "boolean" in each_value_col:
                                        new_df[each_value_col]=new_df[each_value_col].astype(bool)
                                    
        #                         new_df.columns=[(col,entity)] # predicate, object
                                if self._as_column == 'predicate':
                                    new_df.columns=[col]
                                elif self._as_column == 'object' or self._as_column==None:
                                    new_df.columns=[entity]
                                else:
                                    new_df.columns=[entity]
                                
                                data_value_df = pd.concat([data_value_df,new_df],axis=1)
                                
                            else:
                                sub_entities[entity] = element_df.columns
                                data_value_df[col] = pd.DataFrame(element_df.index).reset_index(drop=True).set_index(URI_node_df[col].index)
                            
                            relations_in_graph[entity] = col
                
                        break
                    except KeyError:
                        pass
                    except:
                        raise
            return data_value_df, sub_entities, relations_in_graph


        def node_to_value(rdf_to_unstructured_df, input_data_instance, entities_in_graph, is_a_list, has_value_list):
            
            data_instance_df = get_entity_index(rdf_to_unstructured_df, input_data_instance, is_a_list, has_value_list)
            data_value_df, subLevel_entities, relations_in_graph = get_data_value(data_instance_df, entities_in_graph, is_a_list, has_value_list)
            
            return data_value_df, subLevel_entities, relations_in_graph


        def basic_relations_entities(rdf_to_unstructured_df):
           
            is_a_list = [col for col in rdf_to_unstructured_df.columns for single_is_a in self._user_define_is_a if single_is_a in col]  #rdf:type{URIRef}[0] #rdf:type{URIRef}[1]
            has_value_list = [col for col in rdf_to_unstructured_df.columns for single_has_value in self._user_define_has_value if single_has_value in col] 
            entities_in_graph = [entity for i in is_a_list for entity in rdf_to_unstructured_df[i].unique() if entity not in [self._user_define_data_instance, np.nan, None, 'nan', 'None']]
            
            return is_a_list, has_value_list, entities_in_graph


        # 1 LEVEL # 
        def structure_table (rdf_to_unstructured_df, entities_in_graph, is_a_list, has_value_list):
            level = 0
            if self._set_level == "full":
                self._set_level = np.inf 

            multiLevel_entities = {0:[]}

            while(level==0 or (self._set_level > level and multiLevel_entities[level])):
                if level == 0:
                    level += 1 
                    input_data_instance = self._user_define_data_instance
                    multiLevel_data_value, multiLevel_entities[level], multiLevel_relations = \
                    node_to_value(rdf_to_unstructured_df, input_data_instance, entities_in_graph, is_a_list, has_value_list)

                else:

                    for i in range(0,len(multiLevel_entities[level].keys())):
                        input_data_instance = list(multiLevel_entities[level].keys())[i]
                        data_value_df, subLevel_entities, relations_in_graph = node_to_value(rdf_to_unstructured_df, input_data_instance, entities_in_graph, is_a_list, has_value_list)

                        multiLevel_data_value = pd.concat([multiLevel_data_value.reset_index().set_index(multiLevel_relations[input_data_instance]), 
                              data_value_df], axis=1).reset_index(drop=True).set_index('index')

                        multiLevel_relations = {**multiLevel_relations, **relations_in_graph}

                        try:
                            multiLevel_entities[level+1].append(subLevel_entities)
                        except:
                            multiLevel_entities[level+1] = (subLevel_entities)

                    level += 1
            
            if self._remove_columns_unique_values:
                for each_col in multiLevel_data_value.columns:
                    if (len(multiLevel_data_value[each_col].unique()) == len(multiLevel_data_value) and multiLevel_data_value[each_col].dtype=='object'):
                        multiLevel_data_value = multiLevel_data_value.drop(each_col,axis=1)
                    
            return multiLevel_data_value, multiLevel_relations


        is_a_list, has_value_list, entities_in_graph = basic_relations_entities(rdf_to_unstructured_df)
        multiLevel_data_value, multiLevel_relations = structure_table (rdf_to_unstructured_df, entities_in_graph, is_a_list, has_value_list)

        return multiLevel_data_value, multiLevel_relations


