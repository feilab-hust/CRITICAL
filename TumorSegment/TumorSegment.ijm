// main function
function TumorSegment(ImagePath)
{
	open(ImagePath);
	
	// set image properties
	Stack.setXUnit("um");
	run("Properties...", "pixel_width=20.64 pixel_height=20.64 voxel_depth=20");
	
	// convert to 8-bit
	setMinAndMax(110, 4000);
	run("8-bit");
	
	run("Subtract Background...", "rolling=100 stack"); // Subtract Background
	run("Top Hat...", "radius=1.5 don't stack"); // Top Hat Filtering

	// get threshold 
	setAutoThreshold("Otsu dark stack");
	getThreshold(lower, upper);
	threshold = 0.65*lower;
	if (threshold < 40) {
		setAutoThreshold("MaxEntropy dark stack");
		getThreshold(lower, upper);
		threshold = 1.1*lower;
		}

	// Scale 0.5 for quick calculate
	//run("Scale...", "x=0.5 y=0.5 z=0.5 interpolation=None process");
	
	// make binary
	setMinAndMax(threshold, threshold);
	run("Apply LUT", "stack");
	
	// distance watershed
	run("Distance Transform Watershed 3D", "distances=[Borgefors (3,4,5)] output=[16 bits] normalize dynamic=1 connectivity=6");
	
	setMinAndMax(0, 0.1);
	run("8-bit");
	run("Gray Morphology", "radius=1 type=circle operator=[fast erode]");
	
	// segment by Ostu threshold & do statistics
	run("3D Objects Counter", "threshold=" + 128 +" min.=6 objects statistics summary");

	// save Statistics information
	excel_path = path2 + dirList2[n];
	csvPath = substring(excel_path, 0, lengthOf(excel_path)-1) + ".csv";
	Table.save(csvPath);
	run("Close"); 
	rename("Object");
	run("8-bit");

	open(ImagePath);
	run("Scale...", "x=1 y=1 z=1 interpolation=None process");
	Stack.getStatistics(area, mean, min, max, std, histogram);
	setMinAndMax(min, max);
	run("8-bit");
	rename("originds");
	
	run("Merge Channels...", "c1=originds c2=Object create ignore");
	
	// set image properties
	Stack.setXUnit("um");
	run("Properties...", "pixel_width=20.64 pixel_height=20.64 voxel_depth=20");
	
	CompositeName = substring(ImagePath, 0, lengthOf(ImagePath)-4) + "_Merge_v5.tif";
	run("Save", "save=" + CompositeName);
	close("*");
	
}

parentDir = getDirectory("Choose the parent directory ");
dirList1 = getFileList(parentDir);

for(m = 0; m < dirList1.length; m++)
{
	if (endsWith(dirList1[m], "/") == false) continue;
	path1 = parentDir + dirList1[m];
	dirList2 = getFileList(path1);
	for(n = 0; n < dirList2.length; n++) 
	{   
	    if (endsWith(dirList2[n], "/") == false) continue;
		path2 = path1 + dirList2[n];
		dirList3 = getFileList(path2);
	    for (i = 0; i < dirList3.length; i++) 
	    {	
				if (endsWith(dirList3[i], "stack.tif") == false) continue;
				ImagePath = path2 + dirList3[i];
				print(">now processing: " + ImagePath);
				TumorSegment(ImagePath);
			
	    }
	}
}

