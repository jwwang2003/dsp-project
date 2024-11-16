"""
Helper functions necessary to help us in processing the information in the raw
data files.
"""

def parse_str(string: str) -> list[float]:
  data: list[float] = [ [float(n)] for n in string.splitlines() ]
  return data