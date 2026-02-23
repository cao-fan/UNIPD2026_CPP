# Instructions

Install [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

Inside the root folder run:

> uv sync
> uv run python path_planning_final.py

## Alternatively:

> The following instructions require Python 3.12(.7) already installed.

Download the repository. Open the terminal in the saved folder and install the requirements:

>  pip install -r requirements.txt

Install the open source solvers, in particular SCIP was used(see https://pypi.org/project/PuLP/):

>  pip install pulp[open_py]

Install the FFMPEG for the animations:

>  pip install imageio[ffmpeg]

To run the script:

>  python path_planning_final.py

At each visualization window to go on with the next step press 'q', check the terminal for progress.
