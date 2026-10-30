# Group Project: Designing and Optimizing a Deep Neural Network

## Background

- The temperature; pH; glucose, glutamine, lactate, and ammonia concentrations; and viable cell density are among the
  most important factors influencing cell culture dynamics.


- The growth rate and death rate of cells, and the production rate of recombinant proteins are particularly affected by
  these factors.

## Task

- Your main task is to design a fully connected neural network and optimize its hyperparameters to:
    - Predict:
        - Specific Growth Rate [d^-1]
        - Specific Death Rate [d^-1]
        - Specific Production Rate [pg cell^-1 d^-1]
    - Based on:
        - Temperature [C]
        - pH
        - Glucose [mM]
        - Glutamine [mM]
        - Lactate [mM]
        - Ammonia [mM]
        - Viable Cell Density [cells/mL]


- In your GitHub repository, add a description in the "About" section. (10 - 20 words)


- You must also include a `README.md` file which has the following structure:
    - **Project Name**
        - Heading 1
        - 50 - 150 words
        - What does your code do?
        - What problem does it solve?
        - Replace heading with your own project name.
        - The project name should be derived from your repository name, but it does not need to be identical.
    - **Highlights**
        - Heading 2
        - Include 3 - 5 bullet points in an unordered list.
        - Begin each bullet point with an action verb.
            - E.g., Analyzed, Built, Designed, Developed, Evaluated, Extracted, Identified, Optimized, etc.
        - Consider what skills you would want a potential employer to know about.
    - **Environment**
        - Heading 2
        - Include Python and all libraries used in the project, with version numbers, in an unordered list.
        - List Python first, followed by the libraries in alphabetical order.
        - Only include libraries that were actually used in the project. Do not include packages that are installed in
          your environment but were not used.
    - **Code Design**
        - Heading 2
        - 500 - 1500 words
        - Describe the workflow of the program after `main.py` is executed.
        - Explain the purpose of each major stage of the workflow and why it exists.
        - Examples of major stages include (but are not limited to) data loading, preprocessing, feature engineering,
          model training, etc.
        - If `main.py` executes code located elsewhere in your repository (e.g., `src/classes.py`), then that code
          should also be described.
        - Organize this section using Heading 3 subsections.
        - For each subsection, describe:
            - What does this stage of the workflow do?
            - Why is this stage necessary?
            - How does this stage contribute to the overall workflow?
    - **Analysis**
        - Heading 2
        - 1000 - 3000 words
        - Report the optimal values that you found for: `hidden_layers`, `activation`, `learning_rate`, `batch_size`,
          and `weight_decay`.
            - To save on computation time, you may use train-test split instead of cross-validation at this stage.
            - Justify the number of `epochs` that you chose to use for training.
            - Justify the hyperparameter values that you chose to include in your search space (e.g., `param_grid`).
            - State the total number of hyperparameter sets you searched and explain how you decided to distribute them
              across the various hyperparameters.
                - E.g., If you chose 3 levels for `learning_rate`, but only 2 levels for `weight_decay`, explain why.
            - Justify your decision in choosing this set of optimal hyperparameter values.
        - Perform 4-fold cross-validation on the optimal neural network (i.e., the neural network with the optimal
          hyperparameters you reported).
            - Report the mean and standard deviation of the RMSE and NRMSE for each target.
            - Make a figure with 4 subplots (i.e., one for each CV fold) in a 2 x 2 configuration that plots `epoch` on
              the x-axis and `loss_train` and `loss_valid` on the y-axis, with clearly labeled axes and legends. This
              figure must show that the training completed within the number of epochs used in each fold. If not, then
              the number of epochs must be increased.
        - Using the optimal neural network as your baseline, you will examine the impact of each hyperparameter
          individually. For each analysis in this section, you will change the value of one hyperparameter while keeping
          the rest at the same value used in the optimal neural network. Use 3-fold cross-validation for this section.
            - Discuss the influence of `hidden_layers` on model error. You should compare 5 architectures: your optimal,
              2 less complex, and 2 more complex.
            - Discuss the influence of `activation` on model error. You should compare: `"sigmoid"`, `"tanh"`, `"relu"`,
              and `"leakyrelu"`.
            - Discuss the influence of `learning_rate` on model error. You should compare 5 values: your optimal, 2
              lower, and 2 greater.
            - Discuss the influence of `batch_size` on model error. You should compare 5 values: your optimal, 2 lower,
              and 2 greater.
            - Discuss the influence of `weight_decay` on model error. You should compare 5 values: your optimal, 2
              lower, and 2 greater. If your optimal was `weight_decay = 0.0` then you may compare it with 4 values that
              are higher.
        - Based on all the data you have collected on the influence of these hyperparameters on model error, discuss
          what next steps you would take to improve the model if you had access to more computational power.
        - Compare the performance of your optimized neural network to that of an optimized polynomial regression model.
            - You do not need to do anything fancy here, you can recycle old code from the course or past assignments.
            - Discuss the difference in model performance and training time. You do not need to provide exact values for
              training time, just describe the order of magnitude (e.g., seconds vs minutes vs hours, etc.).
        - Always provide evidence supporting your conclusions. Evidence may include exported CSV tables, PNG figures, or
          both.
        - Organize this section using Heading 3 subsections.


- Reminders:
    - You are free to use the libraries found in the course environment and code from the course repository as needed.
    - You are primarily being evaluated on your ability to justify your decisions rather than arriving at the absolute
      optimal solution.
    - Word limits should be viewed as strong suggestions, rather than hard limits.
    - Figures should be embedded in the README. They should not be a link to a file elsewhere in the repository.
    - All figures should be in the `figures` directory.
    - You should present your repository like a portfolio project rather than an assignment for a course. I would
      encourage you not to make any references to the course (e.g., course code, course name, assignment number, student
      ID, etc.).
    - You can find a guide to the major Markdown syntax elements here: https://www.markdownguide.org/cheat-sheet/
    - You can use this website to convert CSV to Markdown, if needed: https://convertcsv.com/csv-to-markdown.htm

## Setting up your assignment

1. Download the [repository template](https://github.com/Shawn-Chahal/chg4360c-repo-template) for the course.


2. Add [`dataset_cell_growth_7factor.csv`](provided_files/dataset_cell_growth_7factor.csv) to the `datasets` directory
   in your local repository.


3. When you open the project in PyCharm, make sure that you have the course environment selected in the bottom-right
   corner.


4. Make sure to replace any placeholder files from the repository template with the appropriate content.


5. Delete any placeholder files that you are not using.


6. Push the project to a new GitHub repository. Keep the repositroy name short (15 - 30 characters or 2 - 4 words).

## Provided files

[`dataset_cell_growth_7factor.csv`](provided_files/dataset_cell_growth_7factor.csv)

- This file contains simulated bioprocess data with the following columns:
    - `Specific Growth Rate [d^-1]`: The specific growth rate of cells in units of d^-1.
    - `Specific Death Rate [d^-1]`: The specific death rate of cells in units of d^-1.
    - `Specific Production Rate [pg cell^-1 d^-1]`: The specific production rate of recombinant protein in units of pg
      cell^-1 d^-1.
    - `Temperature [C]`: The temperature in units of °C.
    - `pH`: The pH.
    - `Glucose [mM]`: Glucose concentration in units of mM.
    - `Glutamine [mM]`: Glutamine concentration in units of mM.
    - `Lactate [mM]`: Lactate concentration in units of mM.
    - `Ammonia [mM]`: Ammonia concentration in units of mM.
    - `Viable Cell Density [cells/mL]`: The concentration of living cells in units of cells/mL.

## Expected Program Behavior

- Your `main.py` script will be run to test your code in the course environment.


- There is no set expectation of what your code should print to terminal or export as tables or figures unless specified
  otherwise in the [Task](#task) section.

## Submission Requirements

- Submit a link to your GitHub repository by email:
    - To: TA that you submitted your group member names to (see Brightspace)
    - CC: All group members
    - Subject: CHG 4360C Group Project Submission


- The timestamp of your submission will be whichever is latest between the timestamp of the last commit in your GitHub
  repository and the timestamp of your Brightspace submission.


- The GitHub repository only needs to be uploaded to one group member's GitHub profile, however each group member is
  encouraged to copy the repository to their respective GitHub profiles for their own portfolio and reference.

## Rubric

| Component         | Weight |
|-------------------|--------|
| Code Execution    | 25%    |
| GitHub Repository | 10%    |
| Introduction      | 5%     |
| Code Design       | 20%    |
| Analysis          | 40%    |

Introduction consists of: Project Name, Highlights, and Environment.