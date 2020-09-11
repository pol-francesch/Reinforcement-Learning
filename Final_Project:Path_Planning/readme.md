# Final Report for AE 4350
This report was created as part of a final project AE 4350 at TU Delft (Bio-inspired intelligence and learning for Aerospace Applications). I created a path planning algorithm using a Monte Carlo Off-Policy control. You can find more detailed information in my report.

# Code for final report
I am including two codes in this report. The first is the one from Rastogi - though slightly modified
to better meet my needs -, and the second one is my implementation of their code, which is heavily
modified. The following subsections are made for running the latter.

## Libraries
To run this code I am using the newest version of Python 3 along with some libraries:
numpy - for array math
matplotlib - for plotting
tqdm - for a progress bar when running RL
imageio - for creating gifs
pygame - used for the map


## Running Code
I developed the code with the PyCharm IDE, however it is not recommended to run it with that.
This is as the PyCharm IDE can cause issues with Matplotlib. It is better to just run the code from the
terminal. You must only run the main.py file, as the rest are simply helper files.
Once you have run the code, it will first ask you to select a map. It does this by showing you maps
until you find one you like. Once it shows you a map, the code stops. You must press space bar while
selecting the map window, and then the terminal will have you input whether you like this map, and
hence perform the algorithm on it, or would like to see other maps.

After you have selected your map, sit back and relax as it takes a while to run. Please note that
although tqdm is quite good, it does not give very accurate estimates at first. So even if it says it will
take 200 hours to complete, know that on my desktop PC usually takes around twenty minutes, though
this of course depends on your settings and hardware. Finally, the code will end, and you will find two
new plots in the images directory.


## Options
There are some options found in the main.py file which can easily be changed if you would like to
check versatility. The first two are the size of the map and the resolution. I would keep the size values
between 2 and 6, anything smaller and the rudimentary math I used to create maps begins to break
down. The same is true for anything bigger. The resolution can be changed at will, though it is not a
very useful metric.
The next important value is the cell edge. This is used by pygame to determine the size of the
window. If you cannot see the full size of the window, reduce this value.
16
Finally, if you would not like to remake graphs, you have the choice to do this. You can also rename
your images/plots such that they will not overwrite each other.
There is also the option to change the epsilon value.


## Rastogi’s Code
This code is significantly faster to run the set up as there are not as many states. From our end, the
procedure is somewhat similar. You can simply run racetrack_problem.py and it will run exactly the
same racetrack I have shown in the report.
If you would like to get a random racetrack, you must simply change the options in line 646. Note
that if you have already run this code, and you re-run it, it will overwrite save files and the plots.
