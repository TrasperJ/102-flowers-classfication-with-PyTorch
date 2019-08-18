1. Linux General Info:<br>
Linux OS is actually a combination of the GNU project (providing all the quirks and features wrapping around the kernel such as file systems and package managers) and the Linux shell.<br>
So, when referring to Linux, it generally means the Linux shell + GNU features. <br>

2. The linux hierarchy: Forks and distributions <br>
Forks are on a higher level, different forks each contain their own hierarchy of distros.<br>
Main forks originates from the very early GNU Linux are: Debian, Red Hat, Packman, Gentoo, Slackware<br>
The most popular forks, i.e. Debian & Red Hat, contains distros of:<br>
(1) Debian: <br>
Ubuntu, Mint, BackTrack, Kali<br>
(2) Red Hat: <br>
CentOS, Fedora, OpenSUSE, ManDrake;<br>
Different forks may have different commands. For instance, for Debians (Ubuntu) the packages are installed with apt-get, while for Red Hat, it is yum. 

3. The Permissions of files (users, groups):<br>
Groups in linux are groups of users, each group specify a special set of permissions, users in the group conform to that set of permissions of the group.<br>
Each file or folder has a _ _ _ three digits, each ranging from 1 to 7. They denote the permission for the file owner, the group owner, and everybody, respectively;<br>
1 - 7 is actually the sum of:<br>
1= execute;<br>
2= write;<br>
4= read;<br>
So, if a file is marked as 7, then 7 = 1 + 2 + 4 meaning u can do anything to this file;<br>
To change the permission on a file,<br>
$ sudo chmod 7 file_A,   give file_A the maximum permision<br>

4. Specify the path within Linux file system:<br>
Firstly of all, think of the path as a string, which is a concatenation of characters & directory names.<br>
There are four important characters which denote specific directories:<br>
(1) /  denotes the root directory (cd / will take your to the root directory, if cd /some_directory, then this directory must be directly located within the root directory) (NOTED: if / appears in front, then it means root: /home means the home directory which directly locates in the root directory; yet if it appears in the back, i.e. home/, it means the home directories and all its sub-directories)<br>
(2) ~  the user's home directory (cd ~ will take me to /home/jasper/ on my machine)<br>
(3) .  the current directory (cd . will do nothing, just stay in the current directory) (NOTED: if ./, it means the current directory and all its subdirectories, yet if .file_name, it means this file is hidden, need to be shown with $ls -a)<br>
(4) .. the directory one level higher (cd .. will take you one-step higher into the file hierarchy)<br>
For instance, ~/.bashrc means the hidden bashrc file stored directly in the home directory<br>

5. About the .bashrc and .bash_profile files:<br>
These two files are both hidden files, containing executable shell commands to configure your bash (Bourne Again Shell, when you open up a terminal on a Linux machine, a shell is activated. A shell is a program (with its own commands) to help you communicate with the linux kernel.)<br>
.bashrc is interactive non-login shells while .bash_profile is for login shells, meaning whenever you log-into your system with username and passwords, shell commands stored in the .bash_profile will be sequentially executed. While for .bashrc, commands contained within need to be executed via type in the following command into your terminal $ source ~/.bashrc (.bash_profile can be executed similarly as well)<br>
The most common usage of .bashrc and .bash_profile files are to modify the $PATH (use command $ echo $PATH to see the PATH of your own system) of the system. $PATH is a list of directories that contains binary executables of commands(applications) (such as ls, python, python3, etc). The shell will check over directories in $PATH to look for the executable to execute a particular command(application) when it is called.<br>
Applications can be installed in different locations in the system (all kinds of /bin directories that are for binary executable, such as /bin, /usr/bin, /usr/local/bin/, /home/user_name/anaconda3/bin, etc), through different methods (sudo apt-get install, pip install, conda install, install with downloader installers, etc)<br>
If one application (say you have multiple python installed on your machine through different sources) has multiple executables exist on your machine, then the one whose directory appears more earlier in the $PATH will be called, things come after will not be examined. <br>
So, whenever a new application is installed, the $PATH should be modified using .bashrc or .bash_profile. paths that are exported later near the bottom within the .bashrc file will appear more early in $PATH, thus the corresponding executables havehigher priority to be called.<br>

6. The top command usage:<br>
the top command is a linux buit-in application to monitor the processes being run on the machine. this command has various arguments. It can be called directly with arguments, such as top -u user_name,  or start-up by calling top, then directly type in the argument u, then type in the following parameters.<br>
(1) top -u user_name：shows the processes of one particular user;<br>
(2) top -c ：show absolute path of processes;<br>
(3) top -k PID  then hit Enter twice, kill a process with the given PID;<br>
(4) top -p PID: shows details of a process with PID;<br>
(5) = : return normal state within top after execute a particular argument, without exiting top altogether. <br>
