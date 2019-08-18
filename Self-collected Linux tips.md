1. Linux General Info:<br>
Linux OS is actually a combination of the GNU project (providing all the quirks and features warapping around the kernel such as file systems and package managers) and the Linux shell.<br>
So, when referring to Linux, it generally means the Linux shell + GNU features. <br>

2. The linux hierachy: Forks and distributions <br>
Forks are on a higher level, different forks each contain their own hierarchy of distros.<br>
Main forks origniiates from the very early GNU Linux are: Debian, Red Hat, Packman, Gentoo, Slackware<br>
The most popular forks, i.e. Debian & Red Hat, contains distros of:<br>
(1) Debian: <br>
Ubuntu, Mint, BackTrack, Kali<br>
(2) Red Hat: <br>
CentOS, Fedora, OpenSUSE, ManDrake;<br>
Differet forks may have different commands. For instance, for Debians (Ubuntu) the package are installed with apt-get, while for Red Hat, it is yum. 

3. The Permissions of files (users, groups):<br>
Groups in linux are groups of users, each group specify a special set of permissions, users in the group conform to that set of permissions of the group.<br>
Each file or folder has a _ _ _ three digits, each ranging from 1 to 7. They denote the permission for the file owner, the group owner, and everybody, respectively;<br>
1 - 7 is actually the sum of:<br>
1= execute;<br>
2= write;<br>
4= read;<br>
So, if a file is marked as 7, then 7 = 1 + 2 + 4 meaning u can do anything to this file;<br>
To change the permission on a file,<br>
$ sudo chmod 7 file_A,   give file_A the maximum permision<br>

4. Specify the path within Linux file system:<br>
FIrstly of all, think of the path as a string, which is a concatenation of characters & directory names.<br>
There are four important characters which denote specific directories:<br>
(1) /  denotes the root directory (cd / will take your to the root directory, if cd /some_directory, then this directory must be directly locates within the root directory) (NOTED: if / appears in front, then it means root: /home means the home directory which directly locates in the root directory; yet if it appears in the back, i.e. home/, it means the home directories and all its sub-directories)<br>
(2) ~  the user's home directory (cd ~ will take me to /home/jasper/ on my machine)<br>
(3) .  the current directory (cd . will do nothing, just stay in the current directory) (NOTED: if ./, it means the current directory and all its subdirectories, yet if .file_name, it means this file is hidden, need to be shown with $ls -a)<br>
(4) .. the directory one level higher (cd .. will take you one-step higher into the file hiearhcy)<br>
For instance, ~/.bashrc means the hidden bashrc file stored directly in the home directory<br>

5. About the .bashrc and .bash_profile files:<br>
These two files are both hidden files, containing executable shell commands to configurate your bash (Bourne Again Shell, when you open up a terminal on a Linux machine, a shell is activated. A shell is a program (with its own commands) to help your communicate with the linux kernel.)<br>
.bashrc is interactive non-login shells while .bash_profile is for login shells, meaning whenever you log-into your system with username and passwords, shell commands stored in the .bash_profile will be sequentially executed. While for .bashrc, commands contained within need to be executed via type in the following command into your terminal $ source ~/.bashrc (.bash_profile can be executed similarly as well)<br>
The most common usage of .bashrc and .bash_profile files are to modfiied the $PATH (use command $ echo $PATH to see the PATH of your own system) of the system. $PATH is a list of directories which contains bianry executables of commands(applications) (such as ls, python, python3, etc). The shell will check over directories in $PATH to look for the executable to execute a particular command(applciation) when it is called.<br>
Applciations can be installed in different locations in the system (all kinds of /bin directories that are for binary exectuable, such as /bin, /usr/bin, /usr/local/bin/, /home/user_name/anaconda3/bin, etc), through different methods (sudo apt-get install, pip install, conda install, install with downloader installers, etc)


