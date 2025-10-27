"""Top-level package for AutomatedCellularImageAnalysis."""

__author__ = """Johannes Seiffarth"""
__email__ = "j.seiffarth@fz-juelich.de"
__version__ = "0.3.2"


import pint

ureg = pint.get_application_registry()
Q_ = ureg.Quantity
U_ = ureg.Unit
