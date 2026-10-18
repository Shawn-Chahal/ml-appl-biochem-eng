# Assignment 3: Optimizing Monoclonal Antibody Production

## Background

- Monoclonal antibodies (mAbs) are antibodies produced from a uniquely cloned white blood cell and have been used in
  emerging cancer treatments to mark cancer cells for destruction by the immune system.

- Your goal is to maximize the space-time profit of an mAb production process.

- You have the resources to conduct 30 batch runs to collect data that you can use to solve the problem.
    - I.e., You can download 30 batch run datasets of your choice from the [`provided_files`](provided_files) directory.

- The 1 L bioreactors used require 1 day of downtime between batch runs.

- The initial glucose and glutamine concentrations and initial viable cell density (VCD) for each batch are provided
  in [`initial_conditions.csv`](provided_files/initial_conditions.csv).

- The initial lactate, ammonia, and mAb concentrations in the cell culture are zero.

- The initial total cell density (TCD) is equal to the initial VCD.

- The economics of the production are as follows:
    - **Cell culture cost [\$/L]** = 10 $\times$ (Initial Glucose [mM]) + 20 $\times$ (Initial Glutamine [mM]) +
      0.0005 $\times$ (Initial VCD [cells/mL])
    - **Cell culture value [\$/L]** = 1000 $\times$ (mAb [mg/L])
    - **Profit [\$/L]** = (Cell culture value [\$/L]) - (Cell culture cost [\$/L])
    - **Space-time profit [\$ / (L $\cdot$ d)]** = (Profit [\$/L]) / ((Harvest time [d]) + (Downtime [d]))
    - **Space-time yield [mg / (L $\cdot$ d)]** = (mAb [mg/L]) / ((Harvest time [d]) + (Downtime [d]))

## Task

- Your task for this assignment is to determine the initial glucose concentration, glutamine concentration, and viable
  cell density that will maximize the space-time profit of a monoclonal antibody production.

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
        - Justify your choice of initial conditions you selected to conduct your 30 batch runs.
        - Justify your choice of model architecture, including the regression algorithm and hyperparameters.
        - Compare your selected model against at least one other reasonable alternative.
        - Discuss the advantages and limitations of your model.
        - Explain how the initial conditions of the cell culture influence the production cost and yield of monoclonal
          antibodies.
        - Describe initial cell culture conditions that maximize the profit of a batch run on a space time basis.
        - You should report your:
            - recommended initial glucose concentration.
            - recommended initial glutamine concentration.
            - recommended initial viable cell density.
            - recommended harvest time (i.e., when to stop the production).
            - estimated cell culture cost [$/L].
            - estimated cell culture value at harvest [$/L] with 95% prediction intervals.
            - estimated space-time profit [\$ / (L $\cdot$ d)] with 95% prediction intervals.
            - estimated mAb space-time yield [mg / (L $\cdot$ d)] with 95% prediction intervals.
        - Provide evidence supporting your conclusions. Evidence may include exported CSV tables, PNG figures, or both.
        - Organize this section using Heading 3 subsections.


- Additional Information:
    - There is not a single correct model architecture.
    - You are primarily being evaluated on your ability to justify your decisions rather than arriving at the maximum
      possible space-time profit.
    - Figures and tables do not count towards the word count.
    - Word limits should be viewed as strong suggestions, rather than hard limits.
    - You can use this website to convert CSV to Markdown, if needed: https://convertcsv.com/csv-to-markdown.htm
    - You should present your repository like a portfolio project rather than an assignment for a course. I would
      encourage you not to make any references to the course (e.g., course code, course name, assignment number, student
      ID, etc.).
    - You can find a guide to the major Markdown syntax elements here: https://www.markdownguide.org/cheat-sheet/

## Setting up your assignment

1. Download the [repository template](https://github.com/Shawn-Chahal/chg4360c-repo-template) for the course.

2. Add [`initial_conditions.csv`](provided_files/initial_conditions.csv) to the `datasets` directory in your local
   repository.

3. Add 30 `batch_####.csv` of your choice from the [`provided_files`](provided_files) directory to the `datasets`
   directory in your local repository.

4. When you open the project in PyCharm, make sure that you have the course environment selected in the bottom-right
   corner.

5. Make sure to replace any placeholder files from the repository template with the appropriate content.

6. Delete any placeholder files that you are not using.

7. Push the project to a new GitHub repository.

## Provided files

[`initial_conditions.csv`](provided_files/initial_conditions.csv)

- This file contains the initial conditions for simulated mAb production time series with the following columns:
    - `Batch ID`: The identifier of the batch run.
    - `Glucose [mM]`: Initial glucose concentration in units of mM.
    - `Glutamine [mM]`: Initial glutamine concentration in units of mM.
    - `VCD [cells/mL`: Initial viable cell density in units of cells/mL.

[`batch_####.csv`]

- There are 1029 batch run datasets spanning 7 initial glucose, 7 initial glutamine, 7 initial VCD, and 3 replicates of
  each (7 $\times$ 7 $\times$ 7 $\times$ 3 = 1029).
- Each file contains the time series for a simulated mAb production with the following columns:
    - `Time [h]`: Time point of the measurement in units of h.
    - `Glucose [mM]`: Glucose concentration in units of mM.
    - `Glutamine [mM]`: Glutamine concentration in units of mM.
    - `Lactate [mM]`: Lactate concentration in units of mM.
    - `Ammonia [mM]`: Ammonia concentration in units of mM.
    - `VCD [cells/mL]`: Viable cell density in units of cells/mL.
    - `TCD [cells/mL]`: Total cell density in units of cells/mL.
    - `mAb [mg/L]`: Monoclonal antibody concentration in units of mg/L.

## Expected Program Behavior

Your `main.py` script will be run to test your code.

There is no set expectation of what your code should print to terminal or export as tables or figures.

You are free to use the libraries found in the course environment to accomplish the required [tasks](#task).

You are free to use code from the course repository in your project if needed.

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



