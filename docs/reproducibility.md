\# Reproducibility Guide



\## Installation



Clone the repository:



`git clone https://github.com/Kasenotcase/ORIE5270-Project.git`



Then enter the project directory:



`cd ORIE5270-Project`



Install the package in editable mode with development dependencies:



`pip install -e ".\[dev]"`



\## Run tests



Run the full test suite:



`pytest`



The project is configured to require at least 80% test coverage.



To show the detailed coverage report, run:



`pytest --cov=regime\_mpc --cov-report=term-missing`



\## Quick demo



Run the lightweight reproducibility demo:



`python -m regime\_mpc.cli quick-demo`



This demo uses synthetic data and does not require internet access.



It verifies that the following components work together:



\- synthetic return generation

\- backtesting

\- Markowitz optimization

\- performance metric generation

\- CSV output writing



The output files are written to:



`outputs/quick\_demo/`



Expected files include:



\- performance.csv

\- strategy\_returns.csv

\- turnover.csv

\- costs.csv

\- equal\_weight\_weights.csv

\- markowitz\_weights.csv



\## Notes on full empirical data



The full empirical project uses Yahoo Finance and FRED data. Since these sources require network access and may change over time, unit tests avoid live data downloads.



The package keeps live-data functions available, but reproducibility checks use synthetic or cached data.

