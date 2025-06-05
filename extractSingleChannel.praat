#determine praat version
ver1$ = left$(praatVersion$, (rindex(praatVersion$, ".")-1));
ver1 = 'ver1$'
if ver1 < 5.2
	exit Please download a more recent version of Praat
endif

if ver1 == 5.2
	ver2$ = right$(praatVersion$, length(praatVersion$) - (rindex(praatVersion$, ".")));
	ver2 = 'ver2$'
	if ver2 < 4
		exit Please download a more recent version of Praat (minor)
	endif
endif

beginPause: "Instructions"
	comment: "Select which channel to extract. 1 = left, 2 = right"
	positive: "Channel", 1
	comment: "Click 'Next' to continue"
clicked = endPause("Next", 1);


beginPause ("Instructions")
	comment ("Please select input folder.")
	comment ("Click 'Next' to continue")
clicked = endPause("Next", 1);

#wavefile folder path
directory$ = chooseDirectory$ ("Select folder containing wave files")
if directory$ == ""
	exit Script exited. You did not select a folder.
else
	directory$ = directory$ + "/";
endif

beginPause ("Instructions")
	comment ("Please select output folder.")
	comment ("Click 'Next' to continue")
clicked = endPause("Next", 1);

#wavefile folder path
output_directory$ = chooseDirectory$ ("Select folder to save extracted wave files")
if output_directory$ == ""
	exit Script exited. You did not select a folder.
else
	output_directory$ = output_directory$ + "/";
endif


Create Strings as file list... list 'directory$'
numberOfFiles = Get number of strings

clearinfo
printline Processing...
for ifile to numberOfFiles

	select Strings list
	fileName$ = Get string... ifile
	Read from file... 'directory$''fileName$'

	Extract one channel... channel

	Write to WAV file... 'output_directory$''fileName$'

	select all
	minus Strings list
	Remove

endfor

select all
Remove

clearinfo
printline Done


