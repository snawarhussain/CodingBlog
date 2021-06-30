---
title: "Python, conda and virtual environments"
excerpt: "The python set up with conda and virtual environments explained"
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
Setting up python and managing different packages can be frustrating at first. Not because there's not enough documentation on it. But precisely because there is just too much about it with so many options to choose from.

This post serves as an ultimate guide to setting up python whether you are a beginner and just wanna learn coding with python or want to set up python for a project but have been away from python for too long to forget how it works.

## Different versions of python
There are many versions of python out there; python2 and python3 are the two major versions. python3 being the most common, have many sub-versions. i.e python3.4 python3.8 with python3.9 being recently released. in this post, we will set up python3 with conda. You can choose the python3 version you prefer.

## Anaconda
beginners tend to get confused about Anaconda, Conda and Miniconda. 

Allow me to explain...

**Conda** is a package manager for python. you can use conda to 

install new packages like `numpy`, `matplotlib` and many more.
list existing packages with `conda list` or just remove them.
Conda can also be used to create virtual environments of python with different packages or even different python versions! more on this later.
For now, just keep in mind that conda is used to manage python packages.

[**Anaconda**][anaconda] and [**Miniconda**][miniconda] (Miniconda3 for python3) on the other hand, are software distributions. Software distributions are collections of packages pre-configured and pre-built that can be installed on a system(Ubuntu or windows).

 Just like Ubuntu is a distribution of Linux hah!

Anaconda and Miniconda are different in some ways from each other. Anaconda comes with many python packages like `numpy` pre-installed and a GUI to easily navigate and configure different settings while miniconda is just the management system without the packages installed. you can install packages by yourself. Miniconda is light and doesn't require as many resources as Anaconda does. So finally when you install either of both, anaconda or miniconda you perform commands like `conda list`  or  `conda install [package name]`

According to the official documentation:
### Choose Anaconda if you:

+ Are new to conda or Python

+ Like the convenience of having Python and over 1500 scientific packages automatically installed at once

+ Have the time and disk space (a few minutes and 3 GB), and/or
Don’t want to install each of the packages you want to use individually.

### Choose Miniconda if you:
+ Do not mind installing each of the packages you want to use individually.

+ Do not have time or disk space to install over 1500 packages at once, and/or Just want fast access to Python and the conda commands, and wish to sort out the other programs later

You can download either of them for windows, MAC OS or Linux from the links given above. Follow the installation instruction given on the page
with miniconda you have to choose the version of the python3. I would recommend going for python3.8 or python3.7 

[anaconda]: https://www.anaconda.com/products/individual#Downloads
[miniconda]: https://docs.conda.io/en/latest/miniconda.html

Anaconda comes with the GUI interface as well as the command prompt. while miniconda only comes with the command prompt/power shell prompt 

From here on you can use the GUI interface to search and install packages or open the anaconda command prompt for windows. Once you open it you'll see something like this:
```s
(base) C:\users\snawar hussain\> 
```
the `(base)` means that conda base environment is activated. On Linux, it shows up on the default terminal after you install miniconda or anaconda. On Windows, you need to open the Anaconda Prompt. Search for it in the newly installed programs section of the windows.
## Virtual Environments
Python, being a popular programming language, has many versions. A few years back there was a major update that split the language into two major versions being python2 and python3. Many working projects use either python2 or python3. on top of that, python has hundreds of packages that can be installed as per the requirement of a specific project. and then ON TOP OF THAT, these packages also have many versions.

<img src="https://media.giphy.com/media/3o6Yg4GUVgIUg3bf7W/giphy.gif" width="400" height="200" />

sometimes one package is compatible with a specific version of another package while another package is compatible with another version of the same package.
or the other scenario can be when you are working on different projects or want to run different projects that require different versions of the same package.

This is where the virtual environment comes into play. With conda (the package manager), you can create virtual environments with different versions of python and packages that work in isolation from other environments. so when you activate one environment then the system will only see that version of python and packages installed in it. in this way, you can have many different versions of python and packages all on a single system.
### Creating a new Conda Environment for Python.
In this section, we will create a new conda virtual environment for python and install matplotlib in it.

with the anaconda prompt open for on windows, type following in the terminal

```s
conda env list
```
this will print out all the existing virtual environments on your system with the path to it. the `*` sign shows the current activated environment.  Right now we only have a base environment.
to create a new virtual environment with specific python version use:
```s
conda create --name new_environment python = 3.8
```
with the `--name` argument the next text will be the name of the environment which is `new_environment` in this case. you can change it to any name you like.
`python3.8` will tell the conda to make the environment with python version 3.8
after typing, this command hit ENTER and accept all the prompts. conda will install all the necessary packages.
once it is finished doing its thing.
now if you do a `conda env list` it will show you the newly made environment as well with `*` still on base. 
to activate the new environment type
```s
conda activate new_environment
```
the `conda activate` command accepts the env name to be activated which is `new_environment` in our case.
once you hit ENTER. now it will show you something like 
```s
(new_environment) c:\users\snawar hussain>
```
indicated that the new environment is activated. from here on you can type `python` to envoke python in the command prompt. you can also type `conda list` 
to see all the packages installed.
now to install `matplotlib` we will do a 
```s 
conda install matplotlib
```
this command will install an appropriate matplotlib version for us in that environment
to see if matplotlib is installed do a `conda list` again or type `python` in the terminal with the new environment activated it will open up python (python3.8 in this case) and then type
`import matplotlib` and hit ENTER. if this command doesn't give any errors. that means that matplotlib is installed 
you check the version by typing 
```python
print(matplotlib.__version__)
```
this will print the version number.
press ctrl-Z or type `exit()` to exit python.

from here on, you can play around a little more and do `conda -h` or `conda install -h` to learn about different commands and arguments given to conda. or use can just use the GUI version to make new environments, install and uninstall packages. it is pretty straightforward in the GUI panel of anaconda.

in this post, we learned about Anaconda, Miniconda, Conda and why we use virtual environments. we then we ahead and created a new virtual environment for python and installed a package in that environment.

Now writing few lines of python like this in the command prompt is ok but if you want to do work on a project then we need something else to write code, edit it, debug it and run it. that we need a python code editor. and that is a post for another day :)
