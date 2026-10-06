# Assignment 2: Predicting Cell Growth Rates with Machine Learning

## Background

The specific growth rate of cells is one of the most important parameters in many bioprocesses because it directly
impacts biomass accumulation, product formation, and process productivity. However, the specific growth rate cannot be
manipulated directly. Instead, it is influenced by environmental and metabolic conditions such as temperature, pH,
nutrient availability, and the accumulation of metabolic byproducts. Machine learning models can be used to identify
relationships between these process variables and cell growth. Such models can provide predictions of specific cell
growth rate under different operating conditions.

## Task

- Your task for this assignment is to develop a machine learning model that can predict the specific growth rate of
  cells based on the temperature, pH, glucose concentration, and lactate concentration of a cell culture.

- In your GitHub repository, add a description in the "About" section. (10 - 20 words)

- You must also include a `README.md` file which has the following structure:
    - **Project Name**
        - Heading 1
        - 40 - 80 words
        - What does your code do?
        - What problem does it solve?
        - Replace heading with your own project name.
        - The project name should resemble your repository name, but it does not need to be identical.
    - **Highlights**
        - Heading 2
        - 3 - 10 words per bullet point
        - Include 5 bullet points in an unordered list.
        - Begin each bullet point with an action verb.
            - E.g., https://capd.mit.edu/resources/resume-action-verbs/
        - Focus on significant technical work, such as what you built, designed, implemented, optimized, tested, or
          learned.
        - Consider what you would want a potential employer to know about your Python skills after reading this section.
    - **Environment**
        - Heading 2
        - List the Python version and all libraries used in the project, including version numbers.
        - Use an unordered list.
        - List Python first, followed by the libraries in alphabetical order.
        - Only include libraries that were actually used in the project. Do not include packages that are installed in
          your environment but were not used.
    - **Code Design**
        - Heading 2
        - 500 - 1000 words
        - Describe the workflow of the program after `main.py` is executed.
        - Explain the purpose of each major stage of the workflow and why it exists.
        - Examples of major stages include (but are not limited to) data loading, preprocessing, feature engineering,
          model training, etc.
        - Do not simply list the functions and classes that are called.
        - Organize this section using Heading 3 subsections.
        - For each subsection, describe:
            - What does this stage of the workflow do?
            - Why is this stage necessary?
            - How does this stage contribute to the overall workflow?
    - **Analysis**
        - Heading 2
        - 1000 - 2000 words
        - Justify your choice of model architecture, including the regression algorithm and hyperparameters.
        - Compare your selected model against reasonable alternatives derived from polynomial and *k*-nearest neighbors
          regression.
        - Provide evidence supporting your conclusions. Evidence may include exported CSV tables, PNG figures, or both.
        - Discuss the advantages and limitations of your model.
        - Explain how temperature, pH, glucose concentration, and lactate concentration influence specific growth rate.
        - Describe cell culture conditions:
            - that result in a high specific growth rate.
            - that result in a low specific growth rate.
            - where the model's predictions may not be reliable.
        - Describe potential applications of the model and how it could be used in practice.
        - Organize this section using Heading 3 subsections.


- Additional Information:
    - There is not a single correct model architecture.
    - You are primarily being evaluated on your ability to justify your choice of model architecture rather than
      developing a model with the lowest possible error.
    - Figures and tables do not count towards the word count.
    - Word limits should be viewed as strong suggestions, rather than hard limits.
    - You can use this website to convert CSV to Markdown, if needed: https://convertcsv.com/csv-to-markdown.htm
    - You should present your repository like a portfolio project rather than an assignment for a course. I would
      encourage you not to make any references to the course (e.g., course code, course name, assignment number, student
      ID, etc.).
    - You can find a guide to the major Markdown syntax elements here: https://www.markdownguide.org/cheat-sheet/

## Setting up your assignment

1. Download the [repository template](https://github.com/Shawn-Chahal/chg4360c-repo-template) for the course.

2. Add [`dataset_cell_growth_4factor.csv`](provided_files/dataset_cell_growth_4factor.csv) to the `datasets` directory
   in your local repository.

3. When you open the project in PyCharm, make sure that you have the course environment selected in the bottom-right
   corner.

4. Make sure to replace any placeholder files from the repository template with the appropriate content.

5. Delete any placeholder files that you are not using.

## Provided files

[`dataset_cell_growth_4factor.csv`](provided_files/dataset_cell_growth_4factor.csv)

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

| Component         | Weight |
|-------------------|--------|
| Code Execution    | 15%    |
| GitHub Repository | 15%    |
| Project Name      | 5%     |
| Highlights        | 5%     |
| Environment       | 5%     |
| Code Design       | 20%    |
| Analysis          | 35%    |



