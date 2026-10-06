# Assignment 2: Predicting Cell Growth Rates with Machine Learning

## Background

The specific growth rate of cells is one of the most important parameters in many bioprocesses because it directly
impacts biomass accumulation, product formation, and process productivity. However, the specific growth rate cannot be
manipulated directly. Instead, it is influenced by environmental and metabolic conditions such as temperature, pH,
nutrient availability, and the accumulation of metabolic byproducts.

Machine learning models can be used to identify relationships between these process variables and cell growth. Such
models can support process optimization, experimental design, and decision-making by providing predictions of cell
growth under different operating conditions.

## Task

- Your task for this assignment is to develop a machine learning model that can predict the specific growth rate of
  cells based on the temperature, pH, glucose concentration and lactate concentration of a cell culture.


- You must also include a `README.md` file which has the following structure:
    - Project Name (Come up with a name for your project)
        - One sentence description.
    - Overview
        - What were the goals/objectives of this project?
    - Technologies Used
        - Python + Libraries used, including version numbers.
        - Note: You may have many libraries installed in your environment, but here you should only mention the ones you
          actually used.
    - Code Design
        - Describe what your code does after you run `main.py`.
        - Don't simply state what functions/classes it is calling.
        - Explain why you are using the function or class.
        - Explain the overall workflow of the program and why each major stage exists (e.g., data loading,
          preprocessing, training, evaluation, visualization, etc.).
    - Analysis
        - You must justify your choice of model architecture (i.e., model and hyperparameters).
        - Provide evidence for why this model architecture was better than alternatives for this dataset.
        - Evidence can be in the form of exported tables as CSV files and/or figures as PNG files.
        - Describe advantages of your model over alternatives.
        - Describe the limitations of your model.
        - Describe how temperature, pH, glucose concentration, and lactate concentration affect the specific growth
          rate.
        - Describe cell culture conditions that result in a high specific growth rate.
        - Describe cell culture conditions that result in a low specific growth rate.
        - Describe cell culture conditions where the model's predictions may not be reliable.
        - Describe some applications of the model you developed and how it would be used.


- Additional Information:
    - There is not a single correct model architecture.
    - You are primarily being evaluated on your ability to justify your use of a model rather than developing a model
      with the lowest possible error.
    - You can use this website to convert CSV to Markdown, if needed: https://convertcsv.com/csv-to-markdown.htm
    - You should present your repository like a portfolio project rather than an assignment for a course. I would
      encourage you not to make any references to the course (e.g., course code, course name, assignment number, student
      ID, etc.).

## Setting up your assignment

1. Download the [repository template](https://github.com/Shawn-Chahal/chg4360c-repo-template) for the course.

2. Add [`dataset_growth_4factor.csv`](provided_files/dataset_cell_growth_4factor.csv) to the `datasets` directory in
   your local repository.

3. When you open the project in PyCharm, make sure that you have the course environment selected in the bottom-right
   corner.

4. Make sure to replace any placeholder files from the repository template with the appropriate content.

5. Delete any placeholder files that you are not using.

## Provided files

[`dataset_growth_4factor.csv`](provided_files/dataset_cell_growth_4factor.csv)

- This file contains simulated cell growth rate data with the following columns:
    - `Specific Growth Rate [d^-1]`: The specific growth rate of cells in units of d^-1.
    - `Temperature [C]`: The temperature in units of °C.
    - `pH`: The pH.
    - `Glucose [mM]`: Glucose concentration in units of mM.
    - `Lactate [mM]`: Lactate concentration in units of mM.

## Expected Program Behavior

Your `main.py` script will be run to test your code.

There is no set expectation of what your code should print to terminal or export as tables or figures.

You are free to use the libraries found in the course environment to accomplish the required [tasks](#task).

## Submission Requirements

- Submit a link to your GitHub repository through the Brightspace assignment page.

- The timestamp of your submission will be whichever is latest between the timestamp of the last commit in your GitHub
  repository and the timestamp of your Brightspace submission.

## Rubric

| Component               | Weight |
|-------------------------|--------|
| Repository organization | 5%     |
| Code execution          | 20%    |
| Project name            | 5%     |
| Overview                | 5%     |
| Technologies Used       | 5%     |
| Code Design             | 30%    |
| Analysis                | 30%    |



