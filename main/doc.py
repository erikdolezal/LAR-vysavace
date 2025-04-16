import pdoc
from pathlib import Path
pdoc.render.configure(docformat = "google")
pdoc.pdoc("run", "algorithms", "configs", "SimMain", "Sim", output_directory = Path("doc"))