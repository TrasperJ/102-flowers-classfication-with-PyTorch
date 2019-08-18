Jupyter notebook is a sequential collection of cells. Cell can be in either Edit (green colored) & Command (blue colored) mode, and
cell has Raw(R), Markdown(M), and Code(Y) types. <br>
<br>
Cell types can be selected when the cell is in Command mode. To escape from Edit mode to Command mode, hit the ESC key. Once in Command mode, can select the cell type with R, M, Y key.<br>

0. How to use Jupyter Notebook Remotely: Run your codes on a remote server, while displaying the Jupyter notebook on your local machine<br>
(1) On the server: cd into the project directory<br>
$ jupyter notebook --no-browser --port=8889<br>
(2) On your local machine:<br>
$ ssh -N -f -L localhost:8888:localhost:8889 username@your_remote_host_name<br>
(3) On your local machine, open a browser and go to http://localhost:8888<br>
(4) The first-time connection between the local and remote machines may require a token verification:<br>
The browser will display a prompt-up window asking for a token, and the token can be found in your remote machine terminal after executing step (1).<br>
This token verification is only needed for the first time connection. Later connections just go over steps (1) - (3).<br>

1. Run cell:<br>
(1) If want to run all cells from the top to bottom altogether, go to the Toolbar-->Cell-->Run All;<br>
(2) If want to run a single cell:<br>
  a. Shift + Enter: run the cell and initialize a new cell below;
  b. Ctrl + Enter: run the cell without initializing a new cell below.<br>

2. Insert a new cell:<br>
Select the cell above and escape to command mode with ESC, then hit B.<br>

3. Split a cell:<br>
Toolbar-->Edit-->Split Cell<br>

4. Clean up the running outputs of cells:<br>
(1) Cleaning all cells to prepare for a whole new run (if connect remotely, doing this will refresh the GPUs on the remote server):<br>
  Toolbar--> Restart & Clear Output
(2) Cleaning just one cell:<br>
  In command mode (use ESC to get in command mode), hit R to change to Raw type, then switch back to code type with Y.<br>

5. Display line number in cells:<br>
Toolbar-->View-->Toggle Line Number<br>
 
6. Sometimes if directly compy&paste codes from a text editor, the indentation may be troublesome:<br>
Select the cell with problematic indentations, hit Tab to indent, then hit Shift + Tab to inverse indentation.<br> 
