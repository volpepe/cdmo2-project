# cdmo2-project
Project for the course of Combinatorial Decision Making and Optimization - Module 2 @UniBo.

To run the code, create a virtual environment using the provided requirements file and run `python3 src/main.py`.

The project is about gradient-free optimization. The map contained in this project was created by merging smaller maps downloaded from [Emilia Romagna's official altimetry repository](https://geoportale.regione.emilia-romagna.it/download/download-data?type=raster). The maps were concatenated and a mean filter was applied on the final matrix to obtain a smoother mountain.

The following agents/optimization methods were implemented:
- Nelder-Mead simplex algorithm
- Particle Swarm Optimization
- Backtracking Line Search

The final visualization/animation was created using [Plotly](https://plotly.com/).
