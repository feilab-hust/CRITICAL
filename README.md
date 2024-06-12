# TumorSegment-ImageJMacrocode

This code is used to segment tumors in the lung lobes of mice and obtain information about the number, size and location of tumors. Prior to this, our data had been simply preprocessed.

If you want to test our code, download the contents of the "TumorSegment_testdata" folder. Our test data has been uploaded to google drive: https://drive.google.com/drive/folders/1nCuSpTXkkto2ZSKuN_eA7z6JRkfVCyiA?usp=sharing
Please run this code through ImageJ, we also provide a compressed package of "ImageJ" on google drive for direct use: https://drive.google.com/drive/folders/1nCuSpTXkkto2ZSKuN_eA7z6JRkfVCyiA?usp=sharing

Code usage: Run this code directly through ImageJ. After running, select the "TumorSegment_testdata" folder in "Choose the parent directory", and the macro code will run automatically. The path format for the data needs to be the same as our "TumorSegment_testdata". The code automatically recognizes "stack.tif" in the subfolder for processing, "stack_Merge_v5.tif" is the processed image result, and the ".csv" file is the processed statistics result.
