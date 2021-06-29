---
title: "Python, conda and virtual environments"
exerpt: "The pyhton set up with conda and virtual environments explained"
categories:
    - Blog
    - Python
    - Conda
tags:
    - Python3
    - Anaconda
    - miniconda3
    - virtual environments
---
Setting up python and managing different packages can be really frustraiting at first. Not because there's not enough documentation on it. But because there is just too much about it with so many options to choose from.

This post serves as an ultimate guide to setting up python wether you are a beginner and just wanna learn coding with python or want to set up python for a project but have been away from python for too long to forget how it works.

## Different versions of python
Right now there are many versions of python out there; python2 and python3 being the to major versions. python3 being the most common, have many sub-versions. i.e python3.4 python3.8 with python3.9 being recetly released. in this post we will set up python3 with conda. You can choose the python3 version you prefer.

## Anaconda
beginners tend to get confused about Anaconda, Conda and Miniconda. 

Allow me to explain...

**Conda** is a package manager for python. you can use conda to 

install new packages like `numpy`, `matplotlib` and many more.
list existing packages with `conda list` or just remove them.
Conda can also be used to create virtual environments of python with different pakages or even different python versions! more on this later.
For now just keep in mind that conda is used to manage phython packages.

[**Anaconda**][anaconda] and [**Miniconda**][miniconda] (Miniconda3 for python3) on the other hand, are software distributions. Software distributions are collections of packages pre-configure and pre-built that can be installed on a system(Ubuntu or windows) Just like ubuntu is a distribution of linux hah!

Anaconda and Miniconda are different in some ways from each other. Anaconda comes with many python packages like `numpy` pre-installed and a GUI to easily navigate and configure different setting while miniconda is just the management system without the packages installed. you can install packages by yourself. Miniconda is light and doesn't require as much resources as Anaconda does. So finally when you install either of both, anaconda or minconda you perfrom commands like `conda list` or `conda install [package name]`

According to the official documentation:
### Choose Anaconda if you:

+ Are new to conda or Python
Like the convenience of having Python and over 1500 scientific packages automatically installed at once

+ Have the time and disk space (a few minutes and 3 GB), and/or
Don’t want to install each of the packages you want to use individually.

### Choose Miniconda if you:
+ Do not mind installing each of the packages you want to use individually.

+ Do not have time or disk space to install over 1500 packages at once, and/or Just want fast access to Python and the conda commands, and wish to sort out the other programs later

You can download either of them  for windows, MAC OS or Linux from the links given above. Follow the installation instruction given on the page
with miniconda you have to choose the version of the python3. I would recommend to go with python3.8 or python3.4 

[anaconda]: https://www.anaconda.com/products/individual#Downloads
[miniconda]: https://docs.conda.io/en/latest/miniconda.html

Anaconda comes with the GUI interface as well as the command prompt. while minconda only comes with the command prompt/power shell prompt 

From here on you can use the GUI interface to search and install packages or open the anaconda command prompt for windows. Once you open it you'll see something like this:
```shell
(base) C:\users\snawar hussain\> 
```
the `(base)` means that conda base environment is activated. On linux it shows up on the default terminal after you install miniconda or anaconda. On Windows you need to open the Anaconda Prompt. search for it in the newly installed programs section of the windows.
## Virtual Environments
Python ,being a popular programming language, has many versions. few years back there was a major update that split the language into two major versions being python2 and python3. There are many projects running that use either python2 or python3. on top of that, python has hundreds of packages that can be installed as per requirement of a specific project. and then ON TOP OF THAT, these packages also have many versions.

<img src="https://media.giphy.com/media/3o6Yg4GUVgIUg3bf7W/giphy.gif" width="400" height="200" />

sometimes one package is compatible with a specific version of other package while another package is compatible with another version of thes same package.
or the other scenerio can be when you are working on different projects or want to run different projects that require different versions of the same package.

This is where virtual environment comes in to play. With conda (the package manager),you can create virtual environments with different versions of python and packages that work in isolation to other environment. so when you activate one environment then the system will only see that version of python and packages installed in it. in this way you can have many different version of python and packages all on a single system.
### Creating a new Conda Environment for Python.
In this section we will create a new conda virtual environment for python and install matplotlib in it.

with the anaconda prompt open for on windows, type following in the terminal

```shell
(base) C:\users\snawar hussain\> conda env list
```
this will print out all the existing virtual environments on your system with the path to it. the `*` sign shows the current activated environment.  Right now we only have base environment.
to creat a new virtual environment with specific python version use:
```bash
conda create --name new_environment python = 3.8
```
with the `--name` argument the next text will be the name of the environment which is `new_environment` in this case. you can change it to any name you like.
`python3.8` will tell the conda to make tha environment with python verison 3.8
after typing this command hit ENTER and accept all the prompts. conda will install all the necessary packages.
once its finished doing its thing.
now if you do a `conda env list` it will show you the newly made environment as well with `*` still on base. 
in order to activate the new environement type
```bash
conda activate new_environment
```
the `conda activate` command accepts the env name to be activated which is `new_environment` in our case.
once you hit ENTER. now it will show you something like 
```bash
(new_environment) c:\users\snawar hussain>
```
indicated that new environment is activated from here on you can do a 
```bash
conda list
```
to see all the packages installed.
now to install `matplotlib` we will do a 
```bash 
conda install matplotlib
```
this command will install an appropriate matplotlib version for us in that environment
to see if matplotlib is installed do a `conda list` again or type `python` in the terminal with the new environment activated it will open up python (python3.8 in this case) and then type
`import matplotlib` and hit ENTER. if this command doesn't give any errors. that means that matplotlib is installed 
you check the verison by typing 
```python
print(matplotlib.__version__)
```
this will print the version number.
press ctrl-Z or type `exit()` to exit python.

from here on you can play around a little more and do `conda -h` or `conda install -h` to see learn about different commands and arguments give to conda. or use can just use the GUI version to make new environment, install and unintall packages. it is pretty straight forward in GUI panel of anaconda.

in this post we learned about Anaconda, Miniconda , Conda and why we use virtual environment. we then we ahead and created a new virtual environment for python and installed a package in that environment.

Now writing few lines of python like this in command prompt is ok but if you want to do work on a project then we need something else to write code , edit it, debug it and run it. for that we need a python code editor. and that is a post for another day :)
