
Installation
============

If you are a user
-----------------

What follows is a detailed installation instruction for non-technical Equinor users.
ADCA is a Command Line Interface (CLI) program and must be run from the terminal.
For users with little or no experience with the terminal, the learning curve might be steep.
However, it's worth having a cursory knowledge of the terminal since it's extremely powerful and will remain so for decades to come.

**Prerequisites.**

Before you get started, make sure you have the following on your local machine:

- Python installed on your computer
- Access to GitHub (search "github" on AccessIT), in particular you should be able to access `the ADCA repository <https://github.com/equinor/decline-curve-analysis>`_
- If you run into issues downloading data from PDM, then ensure that the ODBC Driver for SQL Server is installed. See `installation instructions for pdm-datareader <https://github.com/equinor/pdm-datareader>`_ for more information. Typically you have to install Microsoft SQL Client (search "microsoft sql client" on AccessIT).

To check if you have Python installed, open a Windows PowerShell terminal and type::

   > python --version
   Python 3.11.6
   
To figure out *where* Python is installed, type::

   > (Get-Command python).Path
   C:\Appl\Python\Python311\python.exe
   
It could be the case that you have a Python executable, i.e. a ``python.exe``, on your system, without it being accessible as ``python`` in the terminal.
If so, then that is fine! You can continue with this guide.
If not, then install Python so you have a ``python.exe`` somewhere on your system before continuing with this guide.
   
**Create a virtual environment.**

Create a folder called ``dca`` where you want to run ADCA from::

   > cd C:\Appl\
   > mkdir dca
   > cd C:\Appl\dca
   
Create a `virtual environment <https://docs.python.org/3/library/venv.html>`_ inside your newly created folder.
A virtual environment isolates the Python code related to ADCA from any other Python projects you might have on your system.
You'll need the path to your ``python.exe``::

   > cd C:\Appl\dca
   > C:\Appl\Python\Python311\python.exe -m venv venv
   
**Activate the virtual environment.**

At this point you should have a folder called ``venv`` in ``C:\Appl\dca``.
Change directory to ``C:\Appl\dca`` using the ``cd`` command, then activate the virtual environment::

   > cd C:\Appl\dca
   > .\venv\Scripts\activate
   
You should see ``(venv)`` appear next to the input field on the terminal.
To verify that the Python currently used is the one in the virtual environment::

   (venv) > (Get-Command python).Path
   C:\Appl\dca\venv\Scripts\python.exe
   
**Install ADCA.**

First verify that you can see the repository.
`Click on this link <https://github.com/equinor/decline-curve-analysis>`_ and verify that you do not get a **Page not found**.

To install ADCA, navigate to your ``dca`` project folder using ``cd``, then activate the virtual environment if you have not done so already::

   > cd C:\Appl\dca
   > .\venv\Scripts\activate
   
Then visit the `repository <https://github.com/equinor/decline-curve-analysis>`_ website, 
find the `most recent tag <https://github.com/equinor/decline-curve-analysis/tags>`_ 
and then run::
  
  (venv) > pip install git+https://github.com/equinor/decline-curve-analysis.git@<TAGNAME>
  
where ``<TAGNAME>`` in the command above is e.g. ``v1.0.0``.

To upgrade ADCA in the future, check the `tag list <https://github.com/equinor/decline-curve-analysis/tags>`_ for the most recent tag, then run the same command as above, but with the most recent tag, e.g. ``v1.0.2``::

   (venv) > pip install git+https://github.com/equinor/decline-curve-analysis.git@<TAGNAME>
   
Verify that the installation worked by running::

   (venv) > adca --help
   
You should see a help text.
   
   
**Using ADCA.**

Always start by navigating to your ADCA project folder, where the virtual environment is located and where all your ``.yaml`` config files will be::

   > cd C:\Appl\dca
   
Then activate your virtual environment::

   > .\venv\Scripts\activate
   
Once activated, you can run ``adca`` on ``.yaml`` files located in the directory::

   (venv) > adca run my_config.yaml
   
Your directory structure should look like the following::

    C:\Appl\dca
    ├── venv
    ├── output
    └── my_config.yaml
    
A more complicated setup might have more than one config file, as well as data files::

    C:\Appl\dca
    ├── venv
    ├── output
    ├── data_export_monthly.csv
    ├── config_shortterm_forecast.yaml
    └── config_longterm_forecast.yaml
    
That's it.

- Remember to upgrade ADCA every now and then.
- Keep your working directory and ``output`` folder clean.
- Every time you run ADCA in a new terminal, remember to ``cd`` to your working directory ``C:\Appl\dca`` and activate the virtual environment with ``.\venv\Scripts\activate``.

If you are a developer
----------------------

**If you are a developer**, go to the `repository <https://github.com/equinor/decline-curve-analysis>`_.
To install the package in editable mode, first set up an local isolated Python environment, then run::
   
  git clone https://github.com/equinor/decline-curve-analysis.git
  cd decline-curve-analysis
  pip install -e ".[dev]"

**Versions.**
See the GH actions file and the ``pyproject.toml`` file in the `repository <https://github.com/equinor/decline-curve-analysis>`_ for information about Python and package versions.
This information is not repeated here.

**Running the tests.**
See the GH actions file in the `repository <https://github.com/equinor/decline-curve-analysis>`_ for information about how to run the tests.
This information is not repeated here.

**Command Line Interface.**
A CLI is provided that runs ``adca`` on a set of ``.yaml`` config files::

  adca --help
  
Alternatively, run::

  python -m dca.adca --help

**Technical documentation.**
There are two sources of technical documentation:

* These documentation pages.
* For all the details, reading the source code is recommended. It is extensively documented.
* Questions? Do not hestitate to contact us!
