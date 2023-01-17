# Wiggling: a new technique for leveraging and optimizing scale-dependent kernels

Please see the PDF from the most recent release for the abstract.
This readme will only describe how to build and run the experiments, 
figures and the paper.

The Makefile is POSIX-based.
It is assumed to have versions of `pdflatex`, `make`, `python` and
`python-venv` readily available on the system.
Clone the repo and `cd` into the directory.
Then run the following commands to setup the Python environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Afterwards, simply run
```bash
make build/pdf/wiggle.pdf
```
To build from Latex and also run all the experiments to generate the 
figures.