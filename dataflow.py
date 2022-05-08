
import os.path
import pandas as pd
import argparse
import requests
import logging
from datetime import datetime
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions




class LeftJoin(beam.PTransform):
    """This PTransform performs a left join given source_pipeline_name, source_data,
     join_pipeline_name, join_data, common_key constructors"""

    def __init__(self, source_pipeline_name, source_data, join_pipeline_name, join_data, common_key):
        self.join_pipeline_name = join_pipeline_name
        self.source_data = source_data
        self.source_pipeline_name = source_pipeline_name
        self.join_data = join_data
        self.common_key = common_key

    def expand(self, pcolls):
        def _format_as_common_key_tuple(data_dict, common_key):
            return data_dict[common_key], data_dict

        """This part here below starts with a python dictionary comprehension in case you 
        get lost in what is happening :-)"""
        return ({pipeline_name: pcoll 
                | 'Convert to ({0}, object) for {1}'
                .format(self.common_key, pipeline_name)
                                >> beam.Map(_format_as_common_key_tuple, self.common_key)
                 for (pipeline_name, pcoll) in pcolls.items()}
                | 'CoGroupByKey {0}'.format(pcolls.keys()) >> beam.CoGroupByKey()
                | 'Unnest Cogrouped' >> beam.ParDo(UnnestCoGrouped(),
                                                   self.source_pipeline_name,
                                                   self.join_pipeline_name)
                )


class UnnestCoGrouped(beam.DoFn):
    """This DoFn class unnests the CogroupBykey output and emits """

    def process(self, input_element, source_pipeline_name, join_pipeline_name):
        group_key, grouped_dict = input_element
        join_dictionary = grouped_dict[join_pipeline_name]
        source_dictionaries = grouped_dict[source_pipeline_name]
        for source_dictionary in source_dictionaries:
            try:
                source_dictionary.update(join_dictionary[0])
                yield source_dictionary
            except IndexError:  # found no join_dictionary
                yield source_dictionary


class LogContents(beam.DoFn):
    """This DoFn class logs the content of that which it receives """

    def process(self, input_element):
        logging.info("Contents: {}".format(input_element))
        logging.info("Contents type: {}".format(type(input_element)))
        logging.info("Contents Access input_element['Country']: {}".format(input_element['Country']))
        return



def run(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help='Input file to read.')
	parser.add_argument('--output', help='Output table')
	known_args, pipeline_args = parser.parse_known_args(argv)


	with beam.Pipeline(options=PipelineOptions()) as p:

		source_pipeline_name = 'Afghanistan'
		source_data = p | 'Read source data' >> beam.dataframe.io.read_csv('Afghanistan.csv')

		join_pipeline_name = 'country-temperature'
		join_data = p | 'Read source data' >> beam.dataframe.io.read_csv('0_country-temperature.csv')

		common_key = {source_pipeline_name: ['country', 'Year'], join_pipeline_name: ['country', 'Year']}
    	pipelines_dictionary = {source_pipeline_name: source_data,
                            join_pipeline_name: join_data}
    	test_pipeline = (pipelines_dictionary
                     | 'Left join' >> LeftJoin(
                     	source_pipeline_name, source_data, join_pipeline_name, join_data, common_key)
                     | 'Log Contents' >> beam.ParDo(LogContents())
                     )

    result = p.run().wait_until_finish()


if __name__ = '__main__':
	logging.getLogger().setLevel(logging.INFO)
	run()
		



