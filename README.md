# [Project Name]

## Project Overview

[write an executive summary for the project]

## Project Structure

```bash
.
├── data                            # [For data privacy add it to .gitignore]
├── models                          # [Store models]
├── notebooks                       # [Notebooks. Depending on the project, notebooks may be added to .gitignore]
│   └── template.ipynb              # [A notebook with usual setups]   
├── src                             # [Store source code used in notebooks]
│   └── __init__.py                 # [Make src a Python module]
├── helpers                         # [Useful ready-to-use helpers]
│   └── __init__.py                 # [Make helpers a Python module]
│   └── sklearn_helper.py           # [Most used Sklearn APIs & useful plotting functions]
│   └── clickhouse_connection.py    # [Connect & run queries on Clickhouse]
│   └── sql_server_connection.py    # [Connect & run queries on SQL Server]
├── configs                         # [Add it to .gitignore]
│   └── config.cfg                  # [Sample config file]
├── docs                            # [Documentation for the project]
├── gradio                          # [Gradio template]
├── artifacts                       # [Store visuals]
├── requirements.txt                # [Requirements (using pipreqs on src directory)]
├── requirements_manual.txt         # [Requirements (manually)]
├── .gitignore                      # [Ignore files that should not be committed to Git]
└── README.md                       # [Describe the project]
```

## Getting Started

1. Name the project
2. Write an executive summary
3. If data is private or too big, add data directory to .gitignore
4. Add configs directory to .gitignore
5. Start developing :blush:

[Provide instructions on how to get the project up and running on a local machine]

## Details

[Provide details of the project]

## Sources

[List all the sources used in this project]
1. [datascience-project-template](https://github.com/DamonRabie/datascience-project-template)

## License

This project is licensed under the [MIT License](LICENSE).

The MIT License is a permissive open source license that allows you to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the software. It also provides an express disclaimer of warranty and liability.

For more information, please refer to the [LICENSE](LICENSE) file.

---