# Setting up the Database
Open the xlsx files and save them as csv files in this directory. Then, run _setupDB.py_. The xlsx, csv, and db files
are already on the .gitignore for this directory so there is no need to worry about accidentally pushing the data
to the repo.

## To whom it may concern 
When I wrote the above instructions, I thought that they were sufficient. I have been informed that I was wrong and 
that I should write more specific instructions to setup the database. If you found the above instructions too vague
and the below instructions too specific, then your reading comprehension skills lie somewhere between what is required
for each.
If you find the below instructions too complex, pray to your diety of choice for answers or ask Ben for help. 
My number text is the best way to reach me, and you can ask someone for my number.

## _**Very**_ specific instructions for setting up the database:
For each of the files _SWaT_Dataset_Attack_v0.xlsx_, _SWaT_Dataset_Normal_v0.xlsx_, and _SWaT_Dataset_Normal_v1.xlsx_
which wil hereafter collectively be revered to as _spreadsheet.xlsx_:

1. Move _spreadsheet.xlsx_ into the _Data_ directory
2. Open the file in _Microsoft Excel_
3. Go to the _Save As_ menu

    3a. Near the upper left corner of the window, click the button labeled _File_

    3b. On the lift side of the window click the button labeled _Save As_
4. Change the file format to CSV

    4a. Below the name of the spreadsheet, open the dropdown menu for file formats. Click the button labeled 
_Excel Workbook (*.xlsx)_ to open the dropdown menu.

    4b. Here you have many options to choose from. Click on one of the following options:

         i. CSV UTF-8 (Comma delimited) (*.csv)
        ii. CSV (Comma delimited) (*.csv)
       iii. CSV (Macintosh) (*.csv)
        iv. CSV (MS-DOS) (*.csv)
5. Click the _Save_ button to the right of the dropdown menu.
6. Repeat steps 1-5 for each spreadsheet.
7. Run _setupDB.py_

If you messed up any of these steps, you must get a new computer and start over.