# Chemotaxis Simulation with Information Theory Analysis

This project simulates chemotaxis behavior in biological organisms using a grid or maze environment. It also incorporates information theory analysis to study the decision-making processes of these organisms. Chemotaxis is the movement of cells or organisms in response to a gradient of a chemical substance. In this simulation, we aim to model and analyze how organisms navigate towards certain chemicals in their environment. This project was realized in a group at the summer academy in Disentis 2023 by the German Academic Scholarship Foundation.

## Table of Contents

- [Introduction to Chemotaxis](#introduction-to-chemotaxis)
- [Installation](#installation)
- [Usage](#usage)

## Introduction to Chemotaxis

Chemotaxis is a biological phenomenon where cells, organisms, or particles move in response to chemical gradients in their environment. This behavior is essential for various biological processes such as immune response, bacteria movement, and embryonic development. Organisms can sense changes in the concentration of specific chemicals and adjust their movement accordingly, either moving towards the source (positive chemotaxis) or away from it (negative chemotaxis).

## Installation

To set up this project, we use Poetry, a dependency management and packaging tool for Python. If you don't have Poetry installed, you can follow the installation instructions [here](https://python-poetry.org/docs/#installation).

Once Poetry is installed, follow these steps to install the project dependencies:

1. Clone this repository:

   ```bash
   git clone https://github.com/eisenmsi/Chemotaxis.git

2. Navigate to the project directory:
    ```bash 
    cd Chemotaxis

3. Install the dependencies using Poetry:
    ```bash
    poetry install


## Usage
This project provides several files for running different aspects of the chemotaxis simulation and information theory analysis. Here are the main files you can interact with:

chemotaxis.py: Runs the chemotaxis simulation in a grid.

chemotaxis_maze.py: Runs the chemotaxis simulation in a maze.

chemotaxis_information_theory.py: Calculates interesting results for information theory.

Happy simulating and analyzing! 🌟
