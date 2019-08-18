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


