import numpy as np
import sys

import logging
logging.disable(sys.maxsize)

class EnsembleItem:
	"""
	A class which creates the models for the ensemble learning.
	"""

	def __init__(self, key, classifier, score=0.0,cluster_id=0):
		self.key=key
		self.classifier=classifier
		self.score=score
		self.cluster_id=cluster_id
