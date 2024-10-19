def parse_str(string: str) -> list[float]:
  data: list[float] = [ float(n) for n in string.splitlines() ]
  return data