__author__ = "JUN WEI WANG"
__email__ = "wjw_03@outlook.com"

"""
Helper functions necessary to help us in processing the information in the raw
data files.
"""

def parse_str(string: str) -> list[float]:
  data: list[float] = [ [float(x) for x in n.strip().split()] for n in string.splitlines() ]
  return data